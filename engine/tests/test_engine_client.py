#!/usr/bin/env python3
"""Exercise the real executable's binary protocol, error paths and strict FP32 output."""
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cpu.common import compare, read_binary
from engine.engine_client import EngineClient

BINARY = ROOT/'engine/build/yolo_engine'


class EngineProtocolTests(unittest.TestCase):
    def run_cli(self, *args, data=b''):
        return subprocess.run([str(BINARY),*args],input=data,capture_output=True,cwd=ROOT)

    def test_clean_eof(self):
        result = self.run_cli('serve')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,b'')

    def test_truncated_eof(self):
        result = self.run_cli('serve',data=b'\0'*4)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(result.stdout,b'')
        self.assertIn(b'truncated serve input',result.stderr)

    def test_bad_precision_and_int8_validation_redirect(self):
        result = self.run_cli('bench','--precision','fp16')
        self.assertNotEqual(result.returncode,0)
        self.assertIn(b'precision must be fp32 or int8',result.stderr)
        result = self.run_cli('validate','artifacts/ref','--precision','int8')
        self.assertNotEqual(result.returncode,0)
        self.assertIn(b'engine/validate_int8.py',result.stderr)

    def test_multiple_inputs_and_exact_binary_output(self):
        x = read_binary(ROOT/'artifacts/ref/bus.input.bin',(1,3,640,640))
        reference = read_binary(ROOT/'artifacts/ref/bus.eager_fp32.bin',(1,300,6))
        for precision in ('fp32','int8'):
            with EngineClient(precision) as client:
                first,second = client.infer(x),client.infer(x)
                self.assertEqual(first.shape,(1,300,6))
                np.testing.assert_array_equal(first,second)
                if precision == 'fp32':
                    self.assertTrue(compare(first,reference)['ok'])
                with self.assertRaisesRegex(ValueError,'expected finite'):
                    client.infer(x[0])


if __name__ == '__main__':
    unittest.main()
