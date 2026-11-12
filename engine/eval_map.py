#!/usr/bin/env python3
"""Compare engine FP32/INT8 on the unchanged 500-image COCO round-3 protocol."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'cpu'))  # cpu.eval_map's existing `from common` import
sys.path.insert(0, str(ROOT))
from cpu.common import ARTIFACTS, COCO_ROOT
from cpu.eval_map import original_boxes, coco_images, read_image
from engine.engine_client import EngineClient

VARIANTS = ('engine-fp32','engine-int8')


def evaluate(variant, binary):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    info, images = coco_images()
    categories = sorted(c['id'] for c in info['categories'])
    assert len(categories) == 80
    detections = []
    start = time.perf_counter()
    with EngineClient(variant.removeprefix('engine-'), binary=binary) as client:
        for n, image in enumerate(images, 1):
            output = client.infer(read_image(image))
            if output.shape != (1,300,6) or not np.isfinite(output).all():
                raise ValueError('invalid output')
            for row in original_boxes(output, image['width'], image['height']):
                detections.append(dict(image_id=image['id'], category_id=categories[int(row[5])],
                                       bbox=row[:4].tolist(), score=float(row[4])))
            if n % 50 == 0:
                print(f'{variant}: {n}/500 images', flush=True)
    destination = ARTIFACTS/'engine/int8'
    destination.mkdir(parents=True, exist_ok=True)
    (destination/f'predictions-{variant}.json').write_text(json.dumps(detections)+'\n')
    coco = COCO(str(COCO_ROOT/'annotations/instances_val2017.json'))
    evaluator = COCOeval(coco, coco.loadRes(detections), 'bbox')
    evaluator.params.imgIds = [im['id'] for im in images]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    result = dict(variant=variant, images=500, ranks=[1,500], image_ids=[int(i) for i in evaluator.params.imgIds],
                  score_threshold=.001, max_outputs=300, coco_max_dets=evaluator.params.maxDets,
                  map50_95=float(evaluator.stats[0]), map50=float(evaluator.stats[1]),
                  seconds=time.perf_counter()-start, threads=6, cpus=[0,2,4,6,8,10])
    (destination/f'map-{variant}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f"mAP {variant}: mAP50-95={result['map50_95']*100:.4f} mAP50={result['map50']*100:.4f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, help='evaluate one variant for method-selection experiments')
    parser.add_argument('--binary', type=Path)
    args = parser.parse_args()
    if args.variant:
        evaluate(args.variant, args.binary)
        return 0
    results = {v:evaluate(v, args.binary) for v in VARIANTS}
    drop = (results['engine-fp32']['map50_95']-results['engine-int8']['map50_95'])*100
    accepted = drop <= 1.
    (ARTIFACTS/'engine/int8/map-summary.json').write_text(json.dumps(
        dict(results=results, int8_drop_points=drop, accepted=accepted), indent=2)+'\n')
    print(f"INT8 drop: {drop:.4f} points (limit 1.0): {'PASS' if accepted else 'FAIL'}", flush=True)
    return int(not accepted)


if __name__ == '__main__':
    sys.exit(main())
