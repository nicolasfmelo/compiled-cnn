#!/usr/bin/env python3
"""CPU inference-only benchmark: 20 warm-up and 200 measured calls."""
import argparse
import json
import os
from pathlib import Path
import resource
import threading
import time

import numpy as np
from common import ARTIFACTS, PCORES, VARIANTS, load_runtime, read_binary


class FrequencySampler:
    """Sample six P-core cpufreq readings at 10 Hz during measured iterations."""
    def __init__(self):
        self.values = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            values = []
            for cpu in sorted(PCORES):
                path = Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq')
                try:
                    values.append(int(path.read_text()) / 1e6)
                except OSError:
                    # /proc MHz fallback if cpufreq is unavailable.
                    sections = Path('/proc/cpuinfo').read_text().split('\n\n')
                    fields = dict(line.split(':', 1) for line in sections[cpu].splitlines() if ':' in line)
                    values.append(float(next(v for k, v in fields.items() if k.strip() == 'cpu MHz')) / 1000)
            self.values.append(float(np.mean(values)))
            self.stop.wait(.1)


def profile():
    import torch
    from aoti.export import load_model
    from ultralytics.utils.torch_utils import get_flops
    os.sched_setaffinity(0, PCORES)
    torch.set_num_threads(6)
    model = load_model('fp32').cpu().float()
    assert model.is_fused() and model.model[-1].end2end
    import thop
    with torch.inference_mode():
        flops = get_flops(model, imgsz=640)
        direct = thop.profile(model, inputs=(torch.zeros(1, 3, 640, 640),), verbose=False)[0] * 2 / 1e9
    if flops <= 0 or not np.isclose(flops, direct):
        raise RuntimeError(f'Ultralytics FLOPs verification failed: {flops} vs {direct}')
    info = dict(gflops=flops, direct_640_gflops=direct, fused=True, imgsz=[640, 640],
                source='Ultralytics get_flops (THOP MACs x2), verified by full 640 forward')
    (ARTIFACTS / 'cpu').mkdir(exist_ok=True)
    (ARTIFACTS / 'cpu/flops.json').write_text(json.dumps(info, indent=2) + '\n')
    print('model: ' + json.dumps(info))


class ContentionMonitor:
    """Detect bursty builds as well as sustained external CPU work."""
    BUILDERS = {'cargo', 'rustc', 'clippy-driver', 'cc1', 'cc1plus', 'nvcc', 'trtexec', 'ninja'}

    def __init__(self):
        import psutil
        self.psutil = psutil
        self.previous = {}
        self.heavy = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        last = time.monotonic()
        while not self.stop.is_set():
            now = time.monotonic()
            current = {}
            total = 0
            contributors = []
            for process in self.psutil.process_iter(['pid', 'name', 'cpu_times', 'create_time']):
                try:
                    if process.pid == os.getpid():
                        continue
                    info = process.info
                    key = (process.pid, info['create_time'])
                    cpu = info['cpu_times'].user + info['cpu_times'].system
                    current[key] = cpu
                    if info['name'] in self.BUILDERS:
                        self.heavy.append((process.pid, info['name'], 'build process present'))
                    # Newly spawned short-lived jobs must not disappear unobserved.
                    baseline = self.previous.get(key, 0 if time.time() - info['create_time'] < now - last else cpu)
                    delta = max(0, cpu - baseline)
                    total += delta
                    if delta:
                        contributors.append((process.pid, info['name'], round(delta / max(now - last, .001) * 100, 1)))
                except self.psutil.Error:
                    pass
            if self.previous and total / max(now - last, .001) > .75:
                self.heavy.append(('aggregate external CPU', round(total / (now - last) * 100, 1),
                                   sorted(contributors, key=lambda x: x[2], reverse=True)[:5]))
            self.previous, last = current, now
            self.stop.wait(.5)

    def finish(self):
        self.stop.set()
        self.thread.join()
        if self.heavy:
            raise RuntimeError('Discard this benchmark: external workload detected: ' + str(self.heavy[:10]))


def check_host():
    print('loadavg: ' + Path('/proc/loadavg').read_text().strip(), flush=True)
    monitor = ContentionMonitor()
    monitor.thread.start()
    time.sleep(5)
    monitor.finish()
    print('host check: no compiler/build processes or >75% aggregate external CPU during 5 s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--check-host', action='store_true')
    args = parser.parse_args()
    if args.check_host:
        return check_host()
    if args.profile:
        return profile()
    if args.variant is None:
        parser.error('--variant is required')
    contention = ContentionMonitor()
    contention.thread.start()
    gflops = json.loads((ARTIFACTS / 'cpu/flops.json').read_text())['gflops']
    x = read_binary(ARTIFACTS / 'ref/bus.input.bin', (1, 3, 640, 640))
    infer, settings, size, load_ms = load_runtime(args.variant)
    start = time.perf_counter()
    infer(x)
    first_ms = (time.perf_counter() - start) * 1000
    for _ in range(20):
        infer(x)
    sampler = FrequencySampler()
    sampler.thread.start()
    times = []
    try:
        for _ in range(200):
            start = time.perf_counter()
            infer(x)
            times.append((time.perf_counter() - start) * 1000)
    finally:
        sampler.stop.set()
        sampler.thread.join()
    contention.finish()
    mean = float(np.mean(times))
    frequency = float(np.mean(sampler.values))
    peak = 6 * 32 * frequency
    achieved = gflops * 1000 / mean
    print(f'variant: {args.variant}')
    print('settings: ' + settings + ' warmup=20 iterations=200')
    print(f'latency ms: mean {mean:.2f} | p50 {np.percentile(times, 50):.2f} | p95 {np.percentile(times, 95):.2f}')
    print(f'throughput: {1000 / mean:.1f} it/s')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'peak RSS MiB: {rss:.1f}')
    print(f'load ms: {load_ms:.2f} | first run ms: {first_ms:.2f}')
    print(f'model MiB: {size / 2**20:.3f}')
    print(f'frequency: P-core mean {frequency:.3f} GHz | samples {len(sampler.values)}')
    print(f'compute: model {gflops:.3f} GFLOPs | achieved {achieved:.2f} GFLOPS | peak FP32 {peak:.2f} GFLOPS | {achieved / peak * 100:.2f}% of peak')
    result = dict(variant=args.variant, settings=settings, times_ms=times, mean_ms=mean,
                  p50_ms=float(np.percentile(times, 50)), p95_ms=float(np.percentile(times, 95)),
                  throughput=1000/mean, rss_mib=rss, load_ms=load_ms, first_ms=first_ms,
                  model_mib=size/2**20, frequency_ghz=frequency, frequency_samples=len(sampler.values),
                  model_gflops=gflops, achieved_gflops=achieved, peak_fp32_gflops=peak, percent_peak=achieved/peak*100)
    (ARTIFACTS / f'cpu/bench-{args.variant}.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
