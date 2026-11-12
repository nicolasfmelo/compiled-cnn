#!/usr/bin/env python3
"""Negative compiler tests: valid ONNX must not silently bypass unsupported semantics."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import re
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('engine_generate', ROOT/'engine/gen/generate.py')
generator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = generator
spec.loader.exec_module(generator)


class UnsupportedGraphTests(unittest.TestCase):
    def reject(self, mutate, message):
        model = onnx.load(ROOT/'artifacts/engine/yolov10s_raw.onnx')
        mutate(model)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/engine') as tmp:
            path = Path(tmp)/'test.onnx'
            onnx.save(model, path)
            with self.assertRaisesRegex(ValueError, message):
                generator.compile_graph(path, Path(tmp)/'generated', 4)
            self.assertFalse((Path(tmp)/'generated/weights.bin').exists())

    def test_unknown_operator(self):
        def mutate(model):
            node = next(n for n in model.graph.node if n.op_type == 'Sigmoid')
            node.op_type = 'LeakyRelu'  # Valid ONNX, intentionally not in our compiler.
        self.reject(mutate, 'unsupported op/attributes LeakyRelu')

    def test_resize_coordinate_semantics(self):
        def mutate(model):
            node = next(n for n in model.graph.node if n.op_type == 'Resize')
            attribute = next(a for a in node.attribute if a.name == 'coordinate_transformation_mode')
            attribute.s = b'half_pixel'
        self.reject(mutate, 'unsupported Resize')

    def test_unsupported_attribute(self):
        def mutate(model):
            node = next(n for n in model.graph.node if n.op_type == 'Resize')
            node.attribute.append(helper.make_attribute('exclude_outside', 1))
        self.reject(mutate, 'unsupported op/attributes Resize')

    def test_custom_domain(self):
        self.reject(lambda m: setattr(m.graph.node[0], 'domain', 'not_supported'), 'custom ONNX domains')


class SyntheticGraphTests(unittest.TestCase):
    def compile(self, nodes, output, shape, initializers=(), error=None):
        graph = helper.make_graph(nodes, 'synthetic',
            [helper.make_tensor_value_info('images', onnx.TensorProto.FLOAT, [1,3,640,640])],
            [helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, shape)],
            [numpy_helper.from_array(np.ones((3,3,1,1), np.float32), 'w'), *initializers])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',18)], ir_version=10)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/engine') as tmp:
            path, dest = Path(tmp)/'synthetic.onnx', Path(tmp)/'generated'
            onnx.save(model,path)
            if error:
                with self.assertRaisesRegex(ValueError,error):
                    generator.compile_graph(path,dest,4)
                self.assertFalse(dest.exists())
                return None
            generator.compile_graph(path,dest,4)
            return (dest/'model.cpp').read_text()

    def conv(self, output, **attrs):
        return helper.make_node('Conv',['images','w'],[output],name=output,
                                strides=[640,640], **attrs)

    @staticmethod
    def activation():
        return [helper.make_node('Sigmoid',['A'],['sig'],name='sig'),
                helper.make_node('Mul',['A','sig'],['act'],name='act')]

    @staticmethod
    def call(code, name):
        return next(line for line in code.splitlines() if line.endswith('// '+name))

    def activation_flag(self, code):
        params = re.search(r'\{([0-9, ]+)\}',self.call(code,'A'))
        return int(params[1].split(',')[7])

    def test_residual_future_producer_not_fused(self):
        code = self.compile([self.conv('A'),self.conv('B'),
            helper.make_node('Add',['A','B'],['C'],name='sum')], 'C',[1,3,1,1])
        self.assertNotIn('&t_',self.call(code,'A'))
        self.assertIn('&t_',self.call(code,'B'))  # B can read A, never the reverse.
        self.assertNotIn('// sum',code)

    def test_residual_earlier_producer_and_constant_fused(self):
        constant = numpy_helper.from_array(np.ones((1,3,1,1),np.float32),'residual')
        code = self.compile([self.conv('A'),helper.make_node('Add',['A','residual'],['C'],name='sum')],
                            'C',[1,3,1,1],[constant])
        self.assertIn('&t_',self.call(code,'A'))
        self.assertNotIn('// sum',code)

    def test_asymmetric_padding_rejected(self):
        for pads in ([0,0,1,0],[0,0,0,1]):
            with self.subTest(pads=pads):
                self.compile([self.conv('A',pads=pads)],'A',[1,3,2 if pads[2] else 1,2 if pads[3] else 1],
                             error='asymmetric Conv padding unsupported')

    def test_symmetric_padding_accepted(self):
        code = self.compile([self.conv('A',pads=[1,1,1,1])],'A',[1,3,2,2])
        self.assertIn('// A',code)

    def test_input_rooted_aliases_rejected(self):
        shape = numpy_helper.from_array(np.array([1,3,640,640],np.int64),'shape')
        starts = numpy_helper.from_array(np.array([0],np.int64),'starts')
        ends = numpy_helper.from_array(np.array([1],np.int64),'ends')
        axes = numpy_helper.from_array(np.array([1],np.int64),'axes')
        sizes = numpy_helper.from_array(np.array([1,2],np.int64),'sizes')
        cases = [([helper.make_node('Reshape',['images','shape'],['alias'])],[1,3,640,640],[shape]),
                 ([helper.make_node('Slice',['images','starts','ends','axes'],['alias'])],[1,1,640,640],[starts,ends,axes]),
                 ([helper.make_node('Split',['images','sizes'],['alias','unused'],axis=1)],[1,1,640,640],[sizes])]
        for nodes, dims, constants in cases:
            with self.subTest(op=nodes[0].op_type):
                self.compile(nodes,'alias',dims,constants,error='input-rooted aliases unsupported')

    def test_arena_rooted_alias_accepted(self):
        shape = numpy_helper.from_array(np.array([1,3,1,1],np.int64),'shape')
        code = self.compile([self.conv('A'),helper.make_node('Reshape',['A','shape'],['alias'])],
                            'alias',[1,3,1,1],[shape])
        self.assertIn('// A',code)

    def test_silu_alias_consumers_preserved(self):
        shape = numpy_helper.from_array(np.array([1,3,1,1],np.int64),'shape')
        starts = numpy_helper.from_array(np.array([0],np.int64),'starts')
        ends = numpy_helper.from_array(np.array([1],np.int64),'ends')
        axes = numpy_helper.from_array(np.array([1],np.int64),'axes')
        sizes = numpy_helper.from_array(np.array([1,2],np.int64),'sizes')
        cases = [(helper.make_node('Reshape',['A','shape'],['alias']),[shape],6),
                 (helper.make_node('Slice',['A','starts','ends','axes'],['alias']),[starts,ends,axes],4),
                 (helper.make_node('Split',['A','sizes'],['alias','unused'],axis=1),[sizes],4)]
        for alias, constants, channels in cases:
            with self.subTest(op=alias.op_type):
                code = self.compile([self.conv('A'),alias,*self.activation(),
                    helper.make_node('Sigmoid',['alias'],['other'],name='other'),
                    helper.make_node('Concat',['act','other'],['C'],axis=1,name='cat')],
                    'C',[1,channels,1,1],constants)
                self.assertEqual(0,self.activation_flag(code))
                self.assertIn('unary(',self.call(code,'sig'))

    def test_silu_graph_output_preserved(self):
        code = self.compile([self.conv('A'),*self.activation()],'A',[1,3,1,1])
        self.assertEqual(0,self.activation_flag(code))
        self.assertIn('unary(',self.call(code,'sig'))

    def test_silu_sole_consumer_fused(self):
        code = self.compile([self.conv('A'),*self.activation()],'act',[1,3,1,1])
        self.assertEqual(1,self.activation_flag(code))
        self.assertNotIn('unary(',code)


class Int8GraphTests(unittest.TestCase):
    def test_fp32_byte_identical_to_pbi097(self):
        expected = {
            'model.cpp': '292fee13b74e5d04bfe1396e1e58efe419e1c53eb6cd31d58954bd6ffe080ee8',
            'model.h': '221b6fac81b09598653f6cc5133b0f3037ce5397eb167212898625379bac2496',
            'plan.txt': 'c4ce1a51346fce1fc9e889038403c3d520c0c851a0e31d7c60c6441df6f42dfc',
            'weights.S': 'b47ae8fb60d9e3b89abb73888b01e2f7631829da26b4cf32edad82c8be501603',
            'weights.bin': '1d9c8bcbf5e6ce64ed9cf8a6774dcfd812a98d16b74a387929b265a90f4f680b'}
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/engine') as tmp:
            destination = Path(tmp)/'generated'
            generator.compile_graph(ROOT/'artifacts/engine/yolov10s_raw.onnx', destination, 5)
            self.assertEqual(expected, {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in destination.iterdir()})

    def test_depthwise_stays_fp32(self):
        nodes = [helper.make_node('Conv',['images','w'],['A'],name='dense',strides=[640,640]),
                 helper.make_node('Conv',['A','dw'],['B'],name='depthwise',group=3)]
        graph = helper.make_graph(nodes, 'depthwise',
            [helper.make_tensor_value_info('images',onnx.TensorProto.FLOAT,[1,3,640,640])],
            [helper.make_tensor_value_info('B',onnx.TensorProto.FLOAT,[1,3,1,1])],
            [numpy_helper.from_array(np.ones((3,3,1,1),np.float32),'w'),
             numpy_helper.from_array(np.ones((3,1,1,1),np.float32),'dw')])
        model = helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)],ir_version=10)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/engine') as tmp:
            tmp=Path(tmp)
            onnx.save(model,tmp/'graph.onnx')
            (tmp/'cal.json').write_text(json.dumps(dict(tensors=dict(images=dict(min=0.,max=1.)))))
            generator.compile_graph(tmp/'graph.onnx',tmp/'generated',5,tmp/'cal.json')
            code=(tmp/'generated/model.cpp').read_text()
            self.assertEqual(code.count('if (ctx.int8)'),1)
            self.assertNotIn('conv_int8',SyntheticGraphTests.call(code,'depthwise'))

    def test_int8_emission_fused_input_fallback_and_determinism(self):
        # B consumes SiLU's original ONNX tensor, not Conv A's alias root.
        nodes = [helper.make_node('Conv', ['images','w'], ['A'], name='first', strides=[640,640]),
                 helper.make_node('Sigmoid', ['A'], ['sig']),
                 helper.make_node('Mul', ['A','sig'], ['act']),
                 helper.make_node('Conv', ['act','w'], ['B'], name='second')]
        # All-zero channels exercise scale guard; +/- .5 and 1.5 exercise even rounding.
        w = np.array([[[[127]],[[.5]],[[1.5]]], [[[0]],[[0]],[[0]]],
                      [[[-127]],[[-.5]],[[-1.5]]]], np.float32)
        graph = helper.make_graph(nodes, 'int8',
            [helper.make_tensor_value_info('images', onnx.TensorProto.FLOAT, [1,3,640,640])],
            [helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT, [1,3,1,1])],
            [numpy_helper.from_array(w,'w')])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',18)], ir_version=10)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/engine') as tmp:
            tmp = Path(tmp)
            onnx.save(model, tmp/'graph.onnx')
            stats = dict(min=-1.,max=2.,p9999=[-.8,1.8],p99999=[-.9,1.9])
            (tmp/'cal.json').write_text(json.dumps(dict(tensors=dict(images=stats,act=stats))))
            for method in ('minmax','p99999','p9999'):
                generator.compile_graph(tmp/'graph.onnx',tmp/'one',5,tmp/'cal.json',method)
                generator.compile_graph(tmp/'graph.onnx',tmp/'two',5,tmp/'cal.json',method)
                self.assertEqual({p.name:p.read_bytes() for p in (tmp/'one').iterdir()},
                                 {p.name:p.read_bytes() for p in (tmp/'two').iterdir()})
                code = (tmp/'one/model.cpp').read_text()
                self.assertEqual(code.count('if (ctx.int8)'),2)
                self.assertIn('conv_int8(',code)
                # Decode packed bytes at the offset emitted in the C++ call.
                offset = int(re.search(r'reinterpret_cast<const int8_t\*>\(engine_weights \+ (\d+)\)',code)[1])*4
                packed = np.frombuffer((tmp/'one/weights.bin').read_bytes(),np.int8,24,offset).reshape(6,4)
                np.testing.assert_array_equal(packed[:3],[[127,0,2,0],[0,0,0,0],[-127,0,-2,0]])
                np.testing.assert_array_equal(packed[3:],0)
                params = re.search(r'engine_weights \+ (\d+), engine_weights \+ (\d+), ([0-9.e+-]+)f, (\d+)\}',code)
                blob=(tmp/'one/weights.bin').read_bytes()
                products=np.frombuffer(blob,'<f4',3,int(params[1])*4)
                correction=np.frombuffer(blob,'<f4',3,int(params[2])*4)
                scale=np.float32(params[3]); zp=int(params[4])
                # All channels have scale_w == 1, including the guarded zero channel.
                np.testing.assert_array_equal(products,[scale]*3)
                expected=(-products.astype(np.float64)*zp*packed[:3].astype(np.int64).sum(axis=1)).astype(np.float32)
                np.testing.assert_array_equal(correction,expected)
            generator.compile_graph(tmp/'graph.onnx',tmp/'one',5,tmp/'cal.json','minmax',('first',))
            self.assertEqual((tmp/'one/model.cpp').read_text().count('if (ctx.int8)'),1)
            with self.assertRaisesRegex(ValueError,'unknown FP32 layers'):
                generator.compile_graph(tmp/'graph.onnx',tmp/'one',5,tmp/'cal.json','minmax',('typo',))


if __name__ == '__main__':
    unittest.main()
