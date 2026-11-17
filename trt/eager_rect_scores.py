# Best eager FP16 score per class, 640x640 vs 384x640 references (decodes *.eager_fp16.bin, incl. rows below 0.25).
# Usage (repo root): uv run python trt/eager_rect_scores.py
import numpy as np
for name, cls in [("webcam_03", 59), ("webcam_04", 59), ("webcam_05", 61), ("zidane", 27)]:
    row = []
    for ref in ("ref", "ref_384x640"):
        out = np.fromfile(f"artifacts/{ref}/{name}.eager_fp16.bin", "<f4").reshape(300, 6)
        row.append(float(out[out[:, 5] == cls][:, 4].max()))
    print(f"{name} class {cls}: eager 640x640 {row[0]:.4f} -> 384x640 {row[1]:.4f}")
