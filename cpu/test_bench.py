"""Exercise summary arithmetic and the shell protocol without real sleeps/inference."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bench import summarize
from common import ARTIFACTS, ROOT, VARIANTS


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        ARTIFACTS.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ARTIFACTS)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_summary_uses_per_run_medians_and_spread(self):
        directory = self.root / 'runs'
        directory.mkdir()
        for variant in VARIANTS:
            for r, mean, p95, fps, ghz in ((1, 10, 11, 100, 4), (2, 20, 40, 50, 3), (3, 30, 33, 33, 2)):
                row = dict(variant=variant, mean_ms=mean, p50_ms=mean, p95_ms=p95,
                           throughput=fps, frequency_ghz=ghz, even_cpu_frequency_ghz=ghz,
                           rss_mib=100, load_ms=10, first_ms=20, model_mib=1, model_gflops=20)
                (directory / f'round-{r}-{variant}.json').write_text(json.dumps(row))
        with contextlib.redirect_stdout(io.StringIO()):
            result = summarize(directory)
        self.assertEqual(len(result), 5)
        for row in result:
            self.assertEqual((row['mean_ms'], row['p95_ms'], row['throughput'], row['frequency_ghz']), (20, 33, 50, 3))
            self.assertEqual((row['min_mean_ms'], row['max_mean_ms']), (10, 30))
            self.assertEqual(row['peak_fp32_gflops'], 576)
            self.assertAlmostEqual(row['percent_peak'], 1000 / 576 * 100)
        self.assertTrue((self.root / 'round3-summary.txt').exists())
        (directory / f'round-3-{VARIANTS[0]}.json').unlink()
        with self.assertRaises(FileNotFoundError):
            summarize(directory)

    def run_shell(self, mode):
        native = self.root / 'native'
        native.mkdir()
        script = native / 'bench_cpu.sh'
        script.write_text((ROOT / 'native/bench_cpu.sh').read_text())
        binary = self.root / 'bin'
        binary.mkdir()
        stubs = {
            'sleep': '#!/usr/bin/env bash\necho "sleep:$*" >> "$TEST_EVENTS"\n',
            'taskset': '#!/usr/bin/env bash\nshift 2\nexec "$@"\n',
            'uv': f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['TEST_EVENTS'], 'a') as f:
    f.write(json.dumps(args) + '\\n')
if '--variant' in args:
    state = Path(os.environ['TEST_STATE'])
    n = int(state.read_text()) if state.exists() else 0
    state.write_text(str(n + 1))
    mode = os.environ['TEST_MODE']
    if mode == 'error': sys.exit(2)
    if mode == 'exhaust' or mode == 'retry' and n == 0: sys.exit(75)
'''
        }
        for name, content in stubs.items():
            path = binary / name
            path.write_text(content)
            path.chmod(0o755)
        events = self.root / 'events'
        env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", TEST_MODE=mode,
                   TEST_EVENTS=str(events), TEST_STATE=str(self.root / 'state'))
        completed = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
        lines = events.read_text().splitlines()
        commands = [json.loads(line) for line in lines if line.startswith('[')]
        variants = [args[args.index('--variant') + 1] for args in commands if '--variant' in args]
        sleeps = [line for line in lines if line.startswith('sleep:')]
        self.assertTrue(all(line == 'sleep:60' for line in sleeps))
        self.assertEqual(len(sleeps), len(variants))
        return completed, variants, commands

    def test_rotating_rounds_and_contention_retry(self):
        completed, variants, commands = self.run_shell('retry')
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        expected = [VARIANTS[0]]  # discarded first attempt, then the full schedule
        for shift in (0, 2, 4):
            expected.extend(VARIANTS[shift:] + VARIANTS[:shift])
        self.assertEqual(variants, expected)
        self.assertTrue(any('--summarize' in args for args in commands))

    def test_contention_retry_limit(self):
        completed, variants, commands = self.run_shell('exhaust')
        self.assertEqual(completed.returncode, 75)
        self.assertEqual(len(variants), 6)  # initial attempt + five retries
        self.assertIn('five retries', completed.stdout)
        self.assertFalse(any('--summarize' in args for args in commands))

    def test_non_contention_errors_do_not_retry(self):
        completed, variants, _ = self.run_shell('error')
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(len(variants), 1)
        self.assertIn('not a contention discard', completed.stdout)


if __name__ == '__main__':
    unittest.main()
