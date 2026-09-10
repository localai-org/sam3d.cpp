#!/usr/bin/env python3
"""Pack native crop diagnostic text into safetensors for the strict comparator."""
import argparse
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open() as stream:
        if stream.readline().strip() != "SAM3D_CROP_RESULTS_V1":
            raise ValueError("invalid native result header")
        tensors = {}
        ids = set()
        for line in stream:
            fields = line.split()
            if len(fields) != 15:
                raise ValueError("native result row must have case ID plus 14 values")
            index = int(fields[0])
            if index < 0 or index in ids:
                raise ValueError("invalid/duplicate case ID")
            ids.add(index)
            values = [float(v) for v in fields[1:]]
            for offset, name in enumerate(["center", "padded_scale", "prior_scale", "scale"]):
                tensors[f"case.{index:04d}.{name}"] = np.asarray(values[offset*2:offset*2+2], dtype=np.float32)
            tensors[f"case.{index:04d}.affine"] = np.asarray(values[8:], dtype=np.float64).reshape(2, 3)
        if not ids:
            raise ValueError("empty native result")
    save_file(tensors, args.output)


if __name__ == "__main__":
    main()
