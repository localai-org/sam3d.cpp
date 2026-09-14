#!/usr/bin/env python3
"""Generate the inference lookup header from the pinned NVIDIA tables."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path


def assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(), filename=str(path))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"dmc_table", "num_vd_table", "check_table"}:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    return values


def flatten(value):
    if isinstance(value, list):
        for child in value:
            yield from flatten(child)
    else:
        yield value


def emit(name: str, ctype: str, dims: str, value) -> str:
    body = ",".join(str(v) for v in flatten(value))
    return f"inline constexpr {ctype} {name}{dims} = {{{body}}};\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    value = assignments(args.source)
    if set(value) != {"dmc_table", "num_vd_table", "check_table"}:
        raise ValueError("pinned FlexiCubes table file has an unexpected schema")
    text = """// Generated from the pinned SAM 3D Objects FlexiCubes tables.
// Copyright (c) Meta Platforms, Inc. and affiliates.
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// Licensed under the Apache License, Version 2.0.
// Do not edit: run scripts/objects/generate_flexicubes_tables.py.
#pragma once
#include <cstdint>
namespace sam3d::flexicubes_tables {
"""
    text += emit("dmc", "int8_t", "[256][4][7]", value["dmc_table"])
    text += emit("num_dual", "int8_t", "[256]", value["num_vd_table"])
    text += emit("check", "int16_t", "[256][5]", value["check_table"])
    text += "}  // namespace sam3d::flexicubes_tables\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)


if __name__ == "__main__":
    main()

