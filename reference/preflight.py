#!/usr/bin/env python3
"""Validate pinned source and checkpoint bytes without importing upstream/PyTorch.

This is only a source/artifact preflight, not an environment or parity gate.
Legacy checkpoint formats are treated as opaque bytes and never deserialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath


COMPONENTS = {
    "body": ("official_body_oracle", "facebook/sam-3d-body-dinov3"),
    "objects": ("official_objects_oracle", "facebook/sam-3d-objects"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    return json.loads(path.read_text(), object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def pinned_identity(value, length: int) -> str:
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"expected a lowercase {length}-digit hash")
    return value


def confined_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("artifact path must be a relative POSIX path")
    part = PurePosixPath(relative)
    if not part.parts or part.is_absolute() or ".." in part.parts:
        raise ValueError("artifact path must not escape its directory")
    base = root.resolve(strict=True)
    path = base.joinpath(*part.parts).resolve(strict=True)
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError("artifact must be a regular file inside its directory")
    return path


def check_source(root: Path, revision: str) -> dict:
    pinned_identity(revision, 40)
    def git(*args):
        return subprocess.check_output(
            ["git", "--no-optional-locks", "-c", "core.hooksPath=/dev/null",
             "-c", "core.fsmonitor=false", "-C", str(root), *args],
            text=True, stderr=subprocess.PIPE,
        ).strip()
    actual = git("rev-parse", "HEAD")
    if actual != revision:
        raise ValueError(f"source revision mismatch: expected {revision}, got {actual}")
    dirty = git("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ValueError("upstream source is modified or contains untracked files")
    return {"revision": actual, "clean": True}


def check_artifact(root: Path, entry: dict) -> dict:
    expected_hash = pinned_identity(entry["sha256"], 64)
    expected_bytes = entry["bytes"]
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError("expected artifact size must be a positive integer")
    path = confined_file(root, entry["path"])
    size = path.stat().st_size
    if size != expected_bytes:
        raise ValueError(f"size mismatch: expected {expected_bytes}, got {size}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"SHA-256 mismatch: expected {expected_hash}, got {actual_hash}")
    return {"path": entry["path"], "bytes": size, "sha256": actual_hash,
            "verified_local_bytes": True}


def one_match(entries: list, key: str, value: str) -> dict:
    matches = [entry for entry in entries if entry.get(key) == value]
    if len(matches) != 1:
        raise ValueError(f"manifest needs exactly one {key}={value}")
    return matches[0]


def preflight(manifest: dict, component: str, source: Path,
              model_directory: Path | None) -> dict:
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported source manifest schema")
    role, repository = COMPONENTS[component]
    src = one_match(manifest["source_revisions"], "role", role)
    model = one_match(manifest["model_repositories"], "repository", repository)
    pinned_identity(model["revision"], 40)
    report = {"schema_version": 1, "component": component,
              "boundary": "source_and_selected_artifact_bytes_only",
              "source_only": model_directory is None,
              "source_repository": src["repository"],
              "model_repository": repository, "model_revision": model["revision"],
              "dependency_closure_complete": manifest.get("dependency_closure_complete") is True,
              "reference_ready": False, "artifacts": [], "errors": []}
    try:
        report["source"] = check_source(source, src["revision"])
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        report["errors"].append(f"source: {exc}")
    entries = model["selected_files"]
    if not entries:
        raise ValueError("selected artifact list must not be empty")
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate artifact paths")
    if model_directory is not None:
        for entry in entries:
            try:
                report["artifacts"].append(check_artifact(model_directory, entry))
            except (ValueError, OSError, KeyError) as exc:
                report["errors"].append(f"{entry['path']}: {exc}")
    report["pass"] = not report["errors"]
    # Environment/auxiliary-asset/capture validation is a separate future check.
    # Even all selected bytes verified must never imply the complete R0 gate.
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("sources.json"))
    parser.add_argument("--component", choices=COMPONENTS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-directory", type=Path)
    group.add_argument("--source-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = preflight(read_json(args.manifest), args.component, args.source,
                           args.model_directory)
        result["manifest_sha256"] = sha256_file(args.manifest)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result = {"pass": False, "reference_ready": False, "errors": [str(exc)]}
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized)
    print(serialized, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
