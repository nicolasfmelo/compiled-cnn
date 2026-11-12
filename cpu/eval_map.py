#!/usr/bin/env python3
"""COCO mAP on val2017 image-id ranks 1–500, disjoint from calibration."""
import argparse
import json
import subprocess
import sys
import time

import numpy as np
from common import ARTIFACTS, COCO_ROOT, coco_images, load_runtime, read_image

EVAL_VARIANTS = ('eager-cpu-fp32', 'ov-cpu-fp32', 'ov-cpu-int8')


def original_boxes(output, width, height):
    gain = min(640 / height, 640 / width)
    left = round((640 - round(width * gain)) / 2 - 0.1)
    top = round((640 - round(height * gain)) / 2 - 0.1)
    rows = np.asarray(output[0], dtype=np.float32)
    rows = rows[rows[:, 4] >= .001].copy()
    rows[:, [0, 2]] = np.clip((rows[:, [0, 2]] - left) / gain, 0, width)
    rows[:, [1, 3]] = np.clip((rows[:, [1, 3]] - top) / gain, 0, height)
    rows[:, 2:4] -= rows[:, 0:2]  # xyxy -> COCO xywh
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=EVAL_VARIANTS)
    args = parser.parse_args()
    if args.variant is None:
        for variant in EVAL_VARIANTS:
            subprocess.run([sys.executable, __file__, '--variant', variant], check=True)
        results = {v: json.loads((ARTIFACTS / f'cpu/map-{v}.json').read_text()) for v in EVAL_VARIANTS}
        drop = (results['ov-cpu-fp32']['map50_95'] - results['ov-cpu-int8']['map50_95']) * 100
        print(f"INT8 drop: {drop:.4f} points (limit 1.0): {'PASS' if drop <= 1 else 'FAIL'}", flush=True)
        (ARTIFACTS / 'cpu/map-summary.json').write_text(json.dumps(dict(results=results, int8_drop_points=drop, accepted=drop <= 1), indent=2) + '\n')
        return int(drop > 1)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    info, images = coco_images()
    # COCO category IDs have gaps; ascending category order is YOLO's 0–79 order.
    categories = sorted(c['id'] for c in info['categories'])
    assert len(categories) == 80
    infer, settings, _, _ = load_runtime(args.variant)
    print(f'variant: {args.variant}\nsettings: {settings}', flush=True)
    detections = []
    start = time.perf_counter()
    for n, image in enumerate(images, 1):
        output = infer(read_image(image))
        if output.shape != (1, 300, 6) or not np.isfinite(output).all():
            raise ValueError('invalid output')
        for row in original_boxes(output, image['width'], image['height']):
            detections.append(dict(image_id=image['id'], category_id=categories[int(row[5])],
                                   bbox=row[:4].tolist(), score=float(row[4])))
        if n % 50 == 0:
            print(f'{args.variant}: {n}/500 images', flush=True)
    destination = ARTIFACTS / 'cpu'
    destination.mkdir(exist_ok=True)
    (destination / f'predictions-{args.variant}.json').write_text(json.dumps(detections) + '\n')
    coco = COCO(str(COCO_ROOT / 'annotations/instances_val2017.json'))
    predicted = coco.loadRes(detections)
    evaluator = COCOeval(coco, predicted, 'bbox')
    evaluator.params.imgIds = [im['id'] for im in images]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    result = dict(variant=args.variant, images=500, ranks=[1, 500], image_ids=[int(i) for i in evaluator.params.imgIds],
                  score_threshold=.001, max_outputs=300, coco_max_dets=evaluator.params.maxDets,
                  map50_95=float(evaluator.stats[0]), map50=float(evaluator.stats[1]),
                  seconds=time.perf_counter() - start)
    (destination / f'map-{args.variant}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"mAP {args.variant}: mAP50-95={result['map50_95'] * 100:.4f} mAP50={result['map50'] * 100:.4f}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
