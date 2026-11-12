#!/usr/bin/env python3
"""Negative compiler tests: valid ONNX must not silently bypass unsupported semantics."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import onnx
from onnx import helper

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


if __name__ == '__main__':
    unittest.main()
