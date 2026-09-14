#!/usr/bin/env python3
"""Capture the pinned upstream SAM 3D Objects mesh decoder stage by stage.

This is the oracle for the native geometry port.  It deliberately executes
the original model classes, sparse convolutions, feature aggregation and
FlexiCubes implementation.  Captures use SAMT files with ggml dimension order
so C++ can consume them without layout conversion.

The default fixture is small enough for quick decoder development while still
crossing a Swin-window boundary and exercising both sparse subdivisions.  A
production SLat can instead be supplied with --features and --coords.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import types

import numpy as np
import torch


GGML_TYPE_F32 = 0
GGML_TYPE_I32 = 26
SAMT_MAGIC = b"SAMT"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_samt(path: Path) -> tuple[np.ndarray, tuple[int, ...], int]:
    with path.open("rb") as stream:
        if stream.read(4) != SAMT_MAGIC:
            raise ValueError(f"{path}: invalid SAMT magic")
        ndims = struct.unpack("<i", stream.read(4))[0]
        if not 1 <= ndims <= 8:
            raise ValueError(f"{path}: invalid SAMT rank {ndims}")
        ne = struct.unpack("<" + "q" * ndims, stream.read(8 * ndims))
        tensor_type = struct.unpack("<i", stream.read(4))[0]
        dtype = {GGML_TYPE_F32: np.dtype("<f4"), GGML_TYPE_I32: np.dtype("<i4"), 30: np.dtype("<i4")}.get(tensor_type)
        if dtype is None:
            raise ValueError(f"{path}: unsupported SAMT type {tensor_type}")
        count = int(np.prod(ne, dtype=np.int64))
        data = np.frombuffer(stream.read(), dtype=dtype)
        if data.size != count:
            raise ValueError(f"{path}: expected {count} elements, found {data.size}")
        # ne is ggml order.  The byte stream is the C-order representation of
        # the corresponding upstream tensor with reversed dimensions.
        return data.reshape(tuple(reversed(ne))), tuple(ne), tensor_type


class Capture:
    def __init__(self, output: Path, compact: bool = False):
        self.output = output
        self.compact = compact
        output.mkdir(parents=True, exist_ok=True)
        self.tensors: dict[str, dict] = {}

    def save(self, name: str, value) -> None:
        if self.compact:
            sparse_roots = {"base.block11", "upsample0.output", "upsample1.output", "decoder.raw"}
            if name not in sparse_roots and not name.startswith((
                "input.", "base.block11.", "upsample0.output.", "upsample1.output.",
                "decoder.raw.", "aggregate.", "mesh.", "output.",
            )):
                return
        if hasattr(value, "feats") and hasattr(value, "coords"):
            self.save(name + ".feats", value.feats)
            self.save(name + ".coords", value.coords)
            return
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous().numpy()
        value = np.asarray(value)
        if value.dtype == np.bool_:
            value = value.astype(np.int32)
        if np.issubdtype(value.dtype, np.integer):
            value = np.ascontiguousarray(value, dtype="<i4")
            tensor_type = GGML_TYPE_I32
        else:
            value = np.ascontiguousarray(value, dtype="<f4")
            tensor_type = GGML_TYPE_F32
        logical_shape = tuple(int(v) for v in value.shape) or (1,)
        if not value.shape:
            value = value.reshape(1)
        ne = tuple(reversed(logical_shape))
        path = self.output / (name + ".samt")
        with path.open("wb") as stream:
            stream.write(SAMT_MAGIC)
            stream.write(struct.pack("<i", len(ne)))
            stream.write(struct.pack("<" + "q" * len(ne), *ne))
            stream.write(struct.pack("<i", tensor_type))
            stream.write(value.tobytes(order="C"))
        self.tensors[name] = {
            "file": path.name,
            "logical_shape": list(logical_shape),
            "samt_ne": list(ne),
            "dtype": "i32" if tensor_type == GGML_TYPE_I32 else "f32",
            "sha256": sha256(path),
        }


def stub_kaolin() -> None:
    """The bundled FlexiCubes only uses Kaolin's shape assertion helper."""
    kaolin = types.ModuleType("kaolin")
    utils = types.ModuleType("kaolin.utils")
    testing = types.ModuleType("kaolin.utils.testing")

    def check_tensor(value, shape, throw=False):
        valid = value.ndim == len(shape) and all(
            expected is None or expected == actual
            for expected, actual in zip(shape, value.shape)
        )
        if throw and not valid:
            raise ValueError(f"tensor {tuple(value.shape)} does not match {shape}")
        return valid

    testing.check_tensor = check_tensor
    sys.modules.update({
        "kaolin": kaolin,
        "kaolin.utils": utils,
        "kaolin.utils.testing": testing,
    })


def synthetic_input() -> tuple[np.ndarray, np.ndarray]:
    # A 3x3x3 block straddling coordinate 32 exercises shifted and unshifted
    # Swin buckets.  Values are deterministic and cover all eight channels.
    xyz = np.stack(np.meshgrid(
        np.arange(30, 33), np.arange(30, 33), np.arange(30, 33), indexing="ij"
    ), axis=-1).reshape(-1, 3)
    coords = np.concatenate((np.zeros((len(xyz), 1), dtype=np.int32), xyz.astype(np.int32)), axis=1)
    index = np.arange(len(coords) * 8, dtype=np.float32).reshape(len(coords), 8)
    feats = np.sin(index * np.float32(0.173)) * np.float32(0.7) + np.cos(index * np.float32(0.037)) * np.float32(0.2)
    return feats.astype(np.float32), coords


def load_input(features: Path | None, coords: Path | None) -> tuple[np.ndarray, np.ndarray, dict]:
    if features is None and coords is None:
        feats, indices = synthetic_input()
        return feats, indices, {"kind": "synthetic-window-boundary-v1"}
    if features is None or coords is None:
        raise ValueError("--features and --coords must be supplied together")
    feats, _, ft = read_samt(features)
    indices, _, ct = read_samt(coords)
    if ft != GGML_TYPE_F32 or ct not in (GGML_TYPE_I32, 30):
        raise ValueError("features must be F32 and coordinates must be I32")
    if feats.ndim != 2 or feats.shape[1] != 8 or indices.shape != (feats.shape[0], 4):
        raise ValueError(f"expected upstream layouts features=[N,8], coords=[N,4], got {feats.shape}, {indices.shape}")
    return feats, indices, {
        "kind": "captured-slat",
        "features": str(features),
        "features_sha256": sha256(features),
        "coords": str(coords),
        "coords_sha256": sha256(coords),
    }


def install_flexicubes_taps(fc, capture: Capture) -> None:
    original_surface = fc._identify_surf_cubes
    original_cases = fc._get_case_id
    original_edges = fc._identify_surf_edges
    original_weights = fc._normalize_weights
    original_vd = fc._compute_vd
    original_tri = fc._triangulate

    def surface(scalar, cube_idx):
        mask, occupancy = original_surface(scalar, cube_idx)
        capture.save("mesh.surface_cube_indices", torch.nonzero(mask).flatten())
        capture.save("mesh.corner_occupancy", occupancy[mask])
        return mask, occupancy

    def cases(occupancy, surface_mask, resolution):
        result = original_cases(occupancy, surface_mask, resolution)
        capture.save("mesh.case_ids", result)
        return result

    def edges(scalar, cube_idx, surface_mask):
        result = original_edges(scalar, cube_idx, surface_mask)
        for name, value in zip(("surface_edges", "edge_index_map", "edge_counts", "surface_edge_mask"), result):
            capture.save("mesh." + name, value)
        return result

    def weights(beta, alpha, gamma, surface_mask, scale):
        result = original_weights(beta, alpha, gamma, surface_mask, scale)
        for name, value in zip(("beta", "alpha", "gamma"), result):
            capture.save("mesh.normalized_" + name, value)
        return result

    def dual(*args, **kwargs):
        result = original_vd(*args, **kwargs)
        for name, value in zip(("dual_vertices", "deviation", "dual_gamma", "dual_index_map", "dual_attributes"), result):
            if value is not None:
                capture.save("mesh." + name, value)
        return result

    def triangulate(*args, **kwargs):
        result = original_tri(*args, **kwargs)
        for name, value in zip(("vertices", "faces", "oriented_surface_edges", "edge_indices", "vertex_attributes"), result):
            if value is not None:
                capture.save("mesh." + name, value)
        return result

    fc._identify_surf_cubes = surface
    fc._get_case_id = cases
    fc._identify_surf_edges = edges
    fc._normalize_weights = weights
    fc._compute_vd = dual
    fc._triangulate = triangulate


def extract_mesh(model, raw, capture: Capture):
    # Spell out SparseFeatures2Mesh.__call__ so the shared-vertex aggregation
    # is itself a parity boundary.  The actual bundled FlexiCubes object still
    # performs all topology decisions and QEF work.
    from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.utils_cube import (
        get_defomed_verts, get_dense_attrs, sparse_cube2verts,
    )

    extractor = model.mesh_extractor
    cube_coords = raw.coords[:, 1:]
    cube_feats = raw.feats
    sdf, deform, color, weights = [extractor.get_layout(cube_feats, name) for name in ("sdf", "deform", "color", "weights")]
    sdf = sdf + extractor.sdf_bias
    vertex_pos, vertex_attrs, _ = sparse_cube2verts(
        cube_coords, torch.cat((sdf, deform, color), dim=-1), training=False
    )
    capture.save("aggregate.vertex_coords", vertex_pos)
    capture.save("aggregate.vertex_attributes", vertex_attrs)
    capture.save("aggregate.cube_coords", cube_coords)
    capture.save("aggregate.cube_weights", weights)

    dense_vertex_attrs = get_dense_attrs(vertex_pos, vertex_attrs, res=extractor.res + 1, sdf_init=True)
    dense_weights = get_dense_attrs(cube_coords, weights, res=extractor.res, sdf_init=False)
    scalar = dense_vertex_attrs[..., 0]
    deformation = dense_vertex_attrs[..., 1:4]
    colors = dense_vertex_attrs[..., 4:]
    deformed_vertices = get_defomed_verts(extractor.reg_v, deformation, extractor.res)

    install_flexicubes_taps(extractor.mesh_extractor, capture)
    vertices, faces, _, attributes = extractor.mesh_extractor(
        voxelgrid_vertices=deformed_vertices,
        scalar_field=scalar,
        cube_idx=extractor.reg_c,
        resolution=extractor.res,
        beta=dense_weights[:, :12],
        alpha=dense_weights[:, 12:20],
        gamma_f=dense_weights[:, 20],
        voxelgrid_colors=colors,
        training=False,
    )
    # The tap records these too; the explicit captures make the public output
    # contract clear even if upstream later reorganizes private helpers.
    capture.save("output.vertices", vertices)
    capture.save("output.faces", faces)
    capture.save("output.vertex_attributes", attributes)
    return vertices, faces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--coords", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract-mesh", action="store_true")
    parser.add_argument("--compact", action="store_true",
                        help="omit large internal activations but retain stage boundaries and extraction taps")
    args = parser.parse_args()

    os.environ.setdefault("LIDRA_SKIP_INIT", "1")
    os.environ.setdefault("SPARSE_BACKEND", "spconv")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    sys.path.insert(0, str(args.upstream))
    stub_kaolin()

    from sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_mesh import SLatMeshDecoder
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")

    model = SLatMeshDecoder(
        resolution=64,
        model_channels=768,
        latent_channels=8,
        num_blocks=12,
        num_heads=12,
        mlp_ratio=4,
        attn_mode="swin",
        window_size=8,
        representation_config={"use_color": True},
        use_fp16=True,
        device=device,
    ).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)

    features_np, coords_np, input_manifest = load_input(args.features, args.coords)
    features = torch.from_numpy(features_np.copy()).to(device=device, dtype=torch.float32)
    coords = torch.from_numpy(coords_np.copy()).to(device=device, dtype=torch.int32)
    sparse = sp.SparseTensor(features, coords)
    capture = Capture(args.output, compact=args.compact)
    capture.save("input.features", features)
    capture.save("input.coords", coords)

    h = model.input_layer(sparse)
    capture.save("base.input_layer", h)
    ape = model.pos_embedder(sparse.coords[:, 1:])
    capture.save("base.ape", ape)
    h = h + ape
    capture.save("base.ape_added", h)
    h = h.type(model.dtype)
    for index, block in enumerate(model.blocks):
        h = block(h)
        capture.save(f"base.block{index:02d}", h)

    for level, block in enumerate(model.upsample):
        prefix = f"upsample{level}"
        norm1 = block.act_layers[0](h)
        capture.save(prefix + ".norm1", norm1)
        activated = block.act_layers[1](norm1)
        capture.save(prefix + ".silu1", activated)
        branch = block.sub(activated)
        skip_input = block.sub(h)
        capture.save(prefix + ".subdivided", branch)
        conv1 = block.out_layers[0](branch)
        capture.save(prefix + ".conv1", conv1)
        norm2 = block.out_layers[1](conv1)
        capture.save(prefix + ".norm2", norm2)
        silu2 = block.out_layers[2](norm2)
        capture.save(prefix + ".silu2", silu2)
        conv2 = block.out_layers[3](silu2)
        capture.save(prefix + ".conv2", conv2)
        skip = block.skip_connection(skip_input)
        capture.save(prefix + ".skip", skip)
        h = conv2 + skip
        capture.save(prefix + ".output", h)

    h = h.type(sparse.dtype)
    raw = model.out_layer(h)
    capture.save("decoder.raw", raw)
    vertices = faces = None
    if args.extract_mesh:
        vertices, faces = extract_mesh(model, raw, capture)

    source_files = [
        args.upstream / "sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_vae/decoder_mesh.py",
        args.upstream / "sam3d_objects/model/backbone/tdfy_dit/representations/mesh/cube2mesh.py",
        args.upstream / "sam3d_objects/model/backbone/tdfy_dit/representations/mesh/flexicubes/flexicubes.py",
        args.upstream / "sam3d_objects/model/backbone/tdfy_dit/representations/mesh/flexicubes/tables.py",
    ]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "scope": "pinned upstream SLat mesh decoder and optional inference-only FlexiCubes extraction",
        "capture_policy": "compact-stage-boundaries" if args.compact else "all-stage-taps",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "spconv": __import__("spconv").__version__,
        "precision": "upstream mixed f32/f16; captures normalized to f32",
        "tf32": False,
        "input": input_manifest,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_sha256": {str(path.relative_to(args.upstream)): sha256(path) for path in source_files},
        "mesh_extracted": args.extract_mesh,
        "vertices": None if vertices is None else int(vertices.shape[0]),
        "faces": None if faces is None else int(faces.shape[0]),
        "tensors": capture.tensors,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "captured": len(capture.tensors),
        "vertices": manifest["vertices"],
        "faces": manifest["faces"],
        "manifest_sha256": sha256(manifest_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
