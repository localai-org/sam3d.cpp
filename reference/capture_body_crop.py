#!/usr/bin/env python3
"""Capture actual pinned Body bbox operations; no checkpoint or model inference.

Imports the standalone reviewed upstream bbox_utils file, not a rewritten oracle
or the package __init__. This establishes only bbox/affine operation fixtures.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform

import cv2
import numpy as np
from safetensors.numpy import save_file


UPSTREAM_REVISION = "b5c765a0d89d789985e186d396315e7590887b94"
SOURCE_PATH = "sam_3d_body/data/transforms/bbox_utils.py"
SOURCE_SHA256 = "0f49a3f857a09f98d1fd9c609df42f875899515f5e10a0ea6b25acd7a9197d79"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.upstream / SOURCE_PATH
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("reviewed upstream bbox source hash mismatch")
    spec = importlib.util.spec_from_file_location("sam3d_official_bbox_capture", source)
    bbox = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bbox)
    # Parameters/boxes are deliberately represented in F32 as in native API.
    cases = [
        [0, 512, 512, 1.25, .75, 0, 0, 0, 120, 240],
        [1, 512, 512, 1.25, .75, 0, 0, 0, 240, 120],
        [2, 512, 512, 1.25, .75, 0, -80, -31, 400, 700],
        [3, 512, 512, .9, .75, 0, 101.25, 102.75, 160.125, 172.5],
        [4, 192, 256, 1.25, .75, 90, 0, 0, 120, 240],
        [5, 512, 384, 1.25, .75, -45, 401.1, -17.4, 902.7, 631.2],
        [6, 511, 513, 1.25, .75, 180, .25, .75, 1.25, 2.25],
    ]
    rng = np.random.default_rng(391)
    for index in range(7, 135):
        x, y = rng.uniform(-500, 1500, 2)
        w, h = rng.uniform(1, 1800, 2)
        out = [(512, 512), (192, 256), (513, 511)][index % 3]
        cases.append([index, *out, [.9, 1.25][index % 2], .75,
                      float(rng.uniform(-180, 180)), x, y, x+w, y+h])
    tensors = {}
    rules = {"schema_version": 1,
             "boundary": "official Body bbox center/scale/aspect/affine operations only; no model parity",
             "tensors": []}
    lines = ["SAM3D_CROP_CASES_V1"]
    for case in cases:
        index, width, height = case[:3]
        padding, prior, rot = [float(np.float32(v)) for v in case[3:6]]
        box = np.asarray(case[6:], dtype=np.float32)
        center, scale = bbox.bbox_xyxy2cs(box, padding)
        prior_scale = bbox.fix_aspect_ratio(scale, prior)
        final_scale = bbox.fix_aspect_ratio(prior_scale, width / height)
        affine = bbox.get_warp_matrix(center, final_scale, rot, (width, height))
        values = {"center": center, "padded_scale": scale, "prior_scale": prior_scale,
                  "scale": final_scale, "affine": affine}
        for name, value in values.items():
            key = f"case.{index:04d}.{name}"
            tensors[key] = np.ascontiguousarray(value)
            # Arithmetic F32 boundaries are expected exact. OpenCV's generic LU
            # and a direct double affine solve can differ at rounding precision.
            rule = {"name": key, "mode": "exact"}
            if name == "affine":
                rule = {"name": key, "mode": "float", "max_abs": 1e-8,
                        "relative_l2": 1e-10, "zero_reference_floor": 1e-12}
            rules["tensors"].append(rule)
        row = [index, width, height, padding, prior, rot, *box.tolist()]
        lines.append(" ".join(str(v) for v in row))
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output/"upstream.safetensors")
    (args.output/"cases.txt").write_text("\n".join(lines)+"\n")
    (args.output/"rules.json").write_text(json.dumps(rules, indent=2)+"\n")
    manifest = {"boundary": rules["boundary"], "upstream_revision": UPSTREAM_REVISION,
                "upstream_file": SOURCE_PATH, "upstream_file_sha256": SOURCE_SHA256,
                "oracle": "actual upstream standalone functions invoked unchanged",
                "not_asserted": ["TopdownAffine wrapper equivalence", "image resampling",
                                 "normalization", "model inference", "full R0 reference gate"],
                "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
                "cases": len(cases), "tensors": len(tensors), "artifacts": {}}
    for name in ["upstream.safetensors", "cases.txt", "rules.json"]:
        manifest["artifacts"][name] = hashlib.sha256((args.output/name).read_bytes()).hexdigest()
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
