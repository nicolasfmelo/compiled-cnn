"""Synchronous binary-pipe client for the generated CPU engine (no inference runtime)."""
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class EngineClient:
    def __init__(self, precision='fp32', threads=6, cpus='0,2,4,6,8,10', binary=None):
        self.process = subprocess.Popen(
            [str(binary or ROOT/'build/bin/yolo_engine'), 'serve', '--precision', precision,
             '--threads', str(threads), '--cpus', cpus],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def infer(self, x):
        x = np.asarray(x, dtype='<f4', order='C')
        if x.shape != (1,3,640,640) or not np.isfinite(x).all():
            raise ValueError('expected finite float32 [1,3,640,640] input')
        try:
            self.process.stdin.write(x.tobytes())
            self.process.stdin.flush()
            data = bytearray()
            while len(data) < 300*6*4:
                chunk = self.process.stdout.read(300*6*4-len(data))
                if not chunk:
                    raise RuntimeError(f'engine closed output pipe (exit {self.process.poll()})')
                data.extend(chunk)
        except BrokenPipeError as error:
            raise RuntimeError(f'engine closed input pipe (exit {self.process.poll()})') from error
        output = np.frombuffer(data, dtype='<f4').reshape(1,300,6).copy()
        if not np.isfinite(output).all():
            raise ValueError('non-finite engine output')
        return output

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait()
        self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
