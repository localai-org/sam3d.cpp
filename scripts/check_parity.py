#!/usr/bin/env python3
"""Compare ordered layer/operation taps stored as safetensors, never pickle.

Rules are explicit per-boundary tolerances frozen by reference capture. No
default 'close enough' thresholds and no silent intersection of tensor names.
This report proves only the declared tensor boundary, never entire-model parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from safetensors import safe_open


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rules(rules: dict) -> list[dict]:
    if rules.get("schema_version") != 1:
        raise ValueError("unsupported rules schema")
    if not isinstance(rules.get("boundary"), str) or not rules["boundary"].strip():
        raise ValueError("rules must name the asserted boundary")
    entries = rules.get("tensors")
    if not isinstance(entries, list) or not entries:
        raise ValueError("rules need a nonempty ordered tensor list")
    seen = set()
    for entry in entries:
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("tensor names must be nonempty and unique")
        seen.add(name)
        mode = entry.get("mode")
        if mode not in ("exact", "float"):
            raise ValueError(f"{name}: mode must be exact or float")
        if mode == "float":
            for field in ("max_abs", "relative_l2", "zero_reference_floor"):
                value = entry.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name}: invalid/missing {field}")
            if entry["zero_reference_floor"] == 0:
                raise ValueError(f"{name}: zero_reference_floor must be positive")
    return entries


def compare_array(reference: np.ndarray, candidate: np.ndarray, rule: dict) -> dict:
    report = {"name": rule["name"], "pass": False,
              "reference_shape": list(reference.shape),
              "candidate_shape": list(candidate.shape),
              "reference_dtype": str(reference.dtype),
              "candidate_dtype": str(candidate.dtype)}
    if reference.shape != candidate.shape:
        return report | {"error": "shape mismatch (axis layout is part of the contract)"}
    if reference.dtype != candidate.dtype:
        return report | {"error": "dtype mismatch; explicitly normalize capture dtypes first"}
    if reference.size == 0:
        return report | {"error": "empty tensor cannot establish a numerical boundary"}
    if reference.dtype.kind not in "biuf":
        return report | {"error": "unsupported tensor dtype"}
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return report | {"error": "non-finite reference or candidate"}
    if rule["mode"] == "exact":
        count = int(np.count_nonzero(reference != candidate))
        return report | {"pass": count == 0, "mismatched_elements": count}
    if reference.dtype.kind != "f":
        return report | {"error": "integer/bool tensors require exact comparison"}
    # Chunk the float64 reductions; do not create two full-size F64 activation copies.
    ref_flat, got_flat = reference.reshape(-1), candidate.reshape(-1)
    max_abs, error_norm, reference_norm = 0.0, 0.0, 0.0
    worst_index = 0
    for begin in range(0, reference.size, 65536):
        ref = ref_flat[begin:begin + 65536].astype(np.float64)
        got = got_flat[begin:begin + 65536].astype(np.float64)
        delta = got - ref
        local = int(np.argmax(np.abs(delta)))
        local_max = float(abs(delta[local]))
        if local_max > max_abs:
            max_abs, worst_index = local_max, begin + local
        # hypot accumulation avoids overflow for finite F32 input tensors.
        error_norm = math.hypot(error_norm, float(np.linalg.norm(delta)))
        reference_norm = math.hypot(reference_norm, float(np.linalg.norm(ref)))
    relative = error_norm / max(reference_norm, rule["zero_reference_floor"])
    if not all(math.isfinite(x) for x in (max_abs, relative)):
        return report | {"error": "non-finite error metric"}
    return report | {"pass": max_abs <= rule["max_abs"] and relative <= rule["relative_l2"],
                     "max_abs": max_abs, "relative_l2": relative,
                     "reference_l2": reference_norm, "worst_flat_index": worst_index,
                     "limits": {k: rule[k] for k in
                                ("max_abs", "relative_l2", "zero_reference_floor")}}


def compare_files(reference: Path, candidate: Path, rules: dict) -> dict:
    entries = validate_rules(rules)
    names = {x["name"] for x in entries}
    results = []
    with safe_open(reference, framework="numpy") as ref, safe_open(candidate, framework="numpy") as got:
        for label, actual in (("reference", set(ref.keys())), ("candidate", set(got.keys()))):
            if actual != names:
                raise ValueError(f"{label} tensor set mismatch: missing={sorted(names - actual)}, "
                                 f"unexpected={sorted(actual - names)}")
        for entry in entries:
            name = entry["name"]
            results.append(compare_array(ref.get_tensor(name), got.get_tensor(name), entry))
    failed = [item["name"] for item in results if not item["pass"]]
    return {"schema_version": 1, "boundary": rules["boundary"], "pass": not failed,
            "first_failed_boundary": failed[0] if failed else None,
            "tensor_count": len(results), "failed_count": len(failed),
            "reference_sha256": sha256_file(reference),
            "candidate_sha256": sha256_file(candidate), "tensors": results}


def read_rules(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate rules key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true", help="also print every tensor result")
    args = parser.parse_args()
    try:
        result = compare_files(args.reference, args.candidate, read_rules(args.rules))
        result["rules_sha256"] = sha256_file(args.rules)
    except Exception as exc:
        result = {"pass": False, "error": str(exc)}
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(serialized)
    if args.verbose:
        print(serialized, end="")
    else:
        summary = {k: v for k, v in result.items() if k != "tensors"}
        summary["report"] = str(args.report)
        print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
