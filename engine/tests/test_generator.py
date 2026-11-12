#!/usr/bin/env python3
"""Negative compiler tests: valid ONNX must not silently bypass unsupported semantics."""
import importlib.util
from pathlib import Path
import sys
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
                self.assertIn(', 0}',self.call(code,'A'))
                self.assertIn('unary(',self.call(code,'sig'))

    def test_silu_graph_output_preserved(self):
        code = self.compile([self.conv('A'),*self.activation()],'A',[1,3,1,1])
        self.assertIn(', 0}',self.call(code,'A'))
        self.assertIn('unary(',self.call(code,'sig'))

    def test_silu_sole_consumer_fused(self):
        code = self.compile([self.conv('A'),*self.activation()],'act',[1,3,1,1])
        self.assertIn(', 1}',self.call(code,'A'))
        self.assertNotIn('unary(',code)


if __name__ == '__main__':
    unittest.main()
