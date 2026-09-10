#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 sam3d.cpp contributors.
"""Verify the pinned Body GGUF bundle; dry-run by default, no network imports.

Inspired by the explicit-file publishing workflows in skin-tokens.cpp/kimodo.cpp;
this implementation is original. Never walks a model or working directory.
--upload requires --accept-model-licenses and a clean source commit.
--create-repo separately authorizes creating a public HF model repository.
Keep the input files immutable throughout validation/upload.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
from typing import BinaryIO

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "LocalAI-io/sam-3d-body-dinov3-GGUF"
COMPONENTS = ("body-dinov3-f32.gguf", "body-pose-branch-f32.gguf", "mhr-lod1-f32.gguf")
DOCUMENTS = {
    "README.md": "distribution/body/README.md",
    "NOTICE": "distribution/body/NOTICE",
    "LICENSE": "LICENSES/SAM.txt",
    "LICENSES/DINOv3.md": "LICENSES/DINOv3.md",
    "LICENSES/MHR-Apache-2.0.txt": "LICENSES/MHR-Apache-2.0.txt",
    "LICENSES/Momentum.txt": "LICENSES/Momentum.txt",
    "LICENSES/Apache-2.0.txt": "LICENSE",
}
CARD_HEADER = """---
license: other
license_name: sam-license
license_link: LICENSE
base_model: facebook/sam-3d-body-dinov3
library_name: gguf
pipeline_tag: image-to-3d
inference: false
tags:
  - gguf
  - ggml
  - sam-3d-body
  - human-pose-estimation
  - 3d-human-mesh-recovery
---
"""


def validate_repo(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValueError("repository must be a namespace/name, not a URL or path")
    if ".." in value or "--" in value or value.endswith((".git", ".", "-")):
        raise ValueError("invalid repository name")
    if value.split("/")[0].lower() in {"facebook", "facebookresearch", "meta"}:
        raise ValueError("refusing to upload a conversion into an upstream namespace")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest key: " + key)
        result[key] = value
    return result


def source_state(root: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True).strip()
    try:
        revision = git("rev-parse", "--verify", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": True}
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("invalid source commit identity")
    return {"commit": revision, "dirty": dirty}


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def open_regular(path: Path, stack: ExitStack) -> BinaryIO:
    # Nonblocking rejects FIFOs without hanging; no-follow rejects symlink files.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    stream = stack.enter_context(os.fdopen(fd, "rb"))
    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
        raise ValueError("publication inputs must be regular files")
    return stream


@dataclass
class ModelFile:
    stream: BinaryIO
    before: tuple

    def unchanged(self):
        if identity(os.fstat(self.stream.fileno())) != self.before:
            raise ValueError("model file changed during publication; inputs must be immutable")


def validate_card(text: str) -> None:
    # A deliberately fixed, reviewed metadata header, not a general YAML parser.
    if not text.startswith(CARD_HEADER):
        raise ValueError("model card metadata differs from the reviewed Body contract")
    for value in (*COMPONENTS, "11aaa346c7204874a1cbafe3d39a979080b2c55a",
                  "LICENSES/DINOv3.md", "LICENSES/MHR-Apache-2.0.txt",
                  "## Supported scope and limitations", "## License and responsible use"):
        if value not in text:
            raise ValueError("model card is missing required provenance/license/scope content")
    if "{{" in text or "}}" in text:
        raise ValueError("unresolved model-card template")


def prepare(root: Path, paths: dict[str, Path], repo: str, stack: ExitStack):
    validate_repo(repo)
    if set(paths) != set(COMPONENTS):
        raise ValueError("exactly the three Body components must be supplied")
    specification = json.loads((root / "distribution/body/artifacts.json").read_text(), object_pairs_hook=unique_object)
    if (set(specification) != {"schema_version", "upstream", "files"}
            or type(specification["schema_version"]) is not int or specification["schema_version"] != 1
            or set(specification["files"]) != set(COMPONENTS)):
        raise ValueError("invalid reviewed artifact specification")
    models, payloads, records = {}, {}, []
    for destination in COMPONENTS:
        expected = specification["files"][destination]
        if (set(expected) != {"bytes", "sha256", "precision", "licenses"}
                or type(expected["bytes"]) is not int or expected["bytes"] < 24
                or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])):
            raise ValueError("invalid reviewed artifact identity")
        stream = open_regular(paths[destination], stack)
        before = identity(os.fstat(stream.fileno()))
        if before[2] != expected["bytes"]:
            raise ValueError("unexpected file size: " + destination)
        if stream.read(8) != b"GGUF" + struct.pack("<I", 3):
            raise ValueError("not a GGUF v3 component: " + destination)
        stream.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError("unapproved or corrupt GGUF: " + destination)
        model = ModelFile(stream, before)
        model.unchanged()
        stream.seek(0)
        models[destination] = model
        records.append({"path": destination, **expected})
    for destination, source in DOCUMENTS.items():
        stream = open_regular(root / source, stack)
        if not 1 <= os.fstat(stream.fileno()).st_size <= 128 * 1024:
            raise ValueError("missing/oversized publication document: " + destination)
        data = stream.read()
        if destination == "README.md":
            card = data.decode("utf-8")
            validate_card(card)
            if REPOSITORY not in card:
                raise ValueError("model card download repository is missing")
            data = card.replace(REPOSITORY, repo).encode("utf-8")
        payloads[destination] = data
        records.append({"path": destination, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    state = source_state(root)
    manifest = {"format": "sam3d-body-gguf-release-v1", "repository": repo,
                "conversion_source": state, "upstream": specification["upstream"], "files": records}
    payloads["MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    checksums = records + [{"path": "MANIFEST.json", "sha256": hashlib.sha256(payloads["MANIFEST.json"]).hexdigest()}]
    payloads["SHA256SUMS"] = "".join(f"{item['sha256']}  {item['path']}\n" for item in checksums).encode()
    return manifest, models, payloads


def upload(root, repo, manifest, models, payloads, create_repo=False):
    if manifest["conversion_source"]["commit"] is None or manifest["conversion_source"]["dirty"]:
        raise ValueError("upload requires a committed, clean source checkout")
    if source_state(root) != manifest["conversion_source"]:
        raise ValueError("source checkout changed after validation")
    for model in models.values():
        model.unchanged()
    # Import/network activity only occurs after all local checks and opt-in flags.
    from huggingface_hub import HfApi, CommitOperationAdd, ModelCard
    from huggingface_hub.errors import RepositoryNotFoundError
    ModelCard(payloads["README.md"].decode()).validate()  # HF's metadata validation
    api = HfApi()
    try:
        parent = api.model_info(repo).sha
    except RepositoryNotFoundError:
        if not create_repo:
            raise ValueError("repository is missing/inaccessible; creation requires --create-repo") from None
        api.create_repo(repo, repo_type="model", private=False, exist_ok=False)
        parent = api.model_info(repo).sha
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=data) for name, data in payloads.items()]
    for name, model in models.items():
        model.unchanged()
        model.stream.seek(0)
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=model.stream))
    # One commit keeps the card, checksums and matching components together.
    # No directory uploads and no deletion/visibility changes of existing repos.
    api.create_commit(repo_id=repo, repo_type="model", operations=operations,
                      parent_commit=parent, commit_message="Publish verified SAM 3D Body F32 GGUF bundle")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--branch", type=Path, required=True)
    parser.add_argument("--mhr", type=Path, required=True)
    parser.add_argument("--repo-id", default=REPOSITORY)
    parser.add_argument("--upload", action="store_true", help="perform HF validation and upload (default: local dry run)")
    parser.add_argument("--accept-model-licenses", action="store_true", help="confirm authority/compliance with all SAM, DINO and MHR terms")
    parser.add_argument("--create-repo", action="store_true", help="also authorize creating a PUBLIC HF model repository if missing")
    args = parser.parse_args(argv)
    if args.upload and not args.accept_model_licenses:
        parser.error("--upload requires --accept-model-licenses")
    if args.create_repo and not args.upload:
        parser.error("--create-repo requires --upload")
    try:
        with ExitStack() as stack:
            paths = dict(zip(COMPONENTS, (args.backbone, args.branch, args.mhr)))
            manifest, models, payloads = prepare(ROOT, paths, args.repo_id, stack)
            print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
            print("Additional generated files: MANIFEST.json, SHA256SUMS", flush=True)
            if not args.upload:
                print("Dry run: verified locally; no repository created, no network requests, no upload.")
                return 0
            if manifest["conversion_source"]["commit"] is None or manifest["conversion_source"]["dirty"]:
                raise ValueError("upload requires a committed, clean source checkout")
            try:
                upload(ROOT, args.repo_id, manifest, models, payloads, args.create_repo)
            except ImportError:
                print("upload requires: uv sync --frozen --extra download", file=sys.stderr)
                return 1
            except Exception as error:
                # HTTP exceptions can inherit OSError. Keep them out of the
                # local-file error handler below; never echo their response body.
                print("upload failed (" + type(error).__name__ + "); check clean inputs, HF access and connectivity", file=sys.stderr)
                return 1
            print("Uploaded verified bundle to https://huggingface.co/" + args.repo_id)
            return 0
    except (OSError, ValueError) as error:
        print("publication rejected: " + str(error), file=sys.stderr)
        return 1
    except Exception as error:
        # Do not echo third-party HTTP exceptions that might include credentials.
        print("publication failed (" + type(error).__name__ + "); check HF access/connectivity", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
