#!/usr/bin/env python3
# Shared schema helpers for SAM 3D Objects -> GGUF conversion.
# See convert_sam3d_to_gguf.py for the naming contract with cpp_ggml/src.
import ctypes
import os
import re
from pathlib import Path

import numpy as np
import torch
import gguf


def quantize_q8_0(w: np.ndarray) -> np.ndarray:
    """Q8_0: blocks of 32 floats -> fp16 scale + 32 int8 quants (34 bytes)."""
    w = w.astype(np.float32).reshape(-1, 32)
    amax = np.max(np.abs(w), axis=1).astype(np.float32)
    amax[amax == 0] = 1e-12
    scale = (amax / 127.0).astype(np.float16)
    q = np.clip(np.rint(w / scale[:, None].astype(np.float32)), -127, 127).astype(np.int8)
    out = np.empty((w.shape[0], 34), dtype=np.uint8)
    out[:, :2] = scale.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8).reshape(-1, 32)
    return out.reshape(-1)


def quantize_q4_0(w: np.ndarray) -> np.ndarray:
    """Q4_0: blocks of 32 floats -> fp16 scale + 16 packed nibbles.

    This mirrors ggml's ``quantize_row_q4_0_ref`` byte for byte.  In
    particular, the signed value with the largest absolute magnitude selects
    the scale sign, the scale is stored as F16 only after quantization, and
    values 0..15 for elements 0..15/16..31 occupy the low/high nibble.
    """
    blocks = np.ascontiguousarray(w, dtype=np.float32).reshape(-1, 32)
    max_index = np.argmax(np.abs(blocks), axis=1)
    max_value = blocks[np.arange(blocks.shape[0]), max_index]
    scale_f32 = max_value / np.float32(-8.0)
    reciprocal = np.divide(
        np.float32(1.0), scale_f32, out=np.zeros_like(scale_f32),
        where=scale_f32 != 0,
    )
    quants = np.minimum(
        15,
        np.trunc(blocks * reciprocal[:, None] + np.float32(8.5)).astype(np.int16),
    ).astype(np.uint8)

    out = np.empty((blocks.shape[0], 18), dtype=np.uint8)
    out[:, :2] = scale_f32.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = quants[:, :16] | (quants[:, 16:] << 4)
    return out.reshape(-1)


def quantize_q4_0_activation_aware(
        w: np.ndarray, activations: np.ndarray,
        scale_multipliers: tuple[float, ...] = (0.75, 0.875, 1.0, 1.125, 1.25),
        max_output_rows: int | None = None, max_tokens: int | None = None,
        ) -> tuple[np.ndarray, dict[str, float]]:
    """Quantize Q4_0 blocks against observed input activations.

    Q4_0's on-disk format stores one F16 scale and 32 four-bit codes per
    block.  The reference quantizer selects that scale from the largest
    weight.  For a linear projection, the relevant error instead is
    ``||(Wq - W) X||``.  This routine evaluates a small deterministic set of
    legal scales per block against the covariance of an observed ``X`` and
    selects the lowest-error encoding.  It does not change the GGUF type,
    block layout, or runtime compute path.

    The search is coordinate descent over input blocks.  It maintains the
    complete projected residual for each calibrated output row, so the choice
    for a block includes cross-block error terms instead of assuming their
    errors are independent.  ``max_*`` make conversion cost bounded for a
    calibration trace while retaining deterministic evenly spaced samples.
    """
    data = np.ascontiguousarray(w, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] % 32:
        raise ValueError("activation-aware Q4_0 requires a rank-2 tensor with a 32-aligned input")
    x = np.ascontiguousarray(activations, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != data.shape[1]:
        raise ValueError(
            "activation shape must be (input_features, tokens), got "
            f"{x.shape} for a {data.shape} weight")
    multipliers = np.asarray(scale_multipliers, dtype=np.float32)
    if multipliers.ndim != 1 or not len(multipliers) or np.any(multipliers <= 0):
        raise ValueError("scale multipliers must be a non-empty sequence of positive values")

    rows, cols = data.shape
    output_rows = min(rows, max_output_rows or rows)
    token_count = min(x.shape[1], max_tokens or x.shape[1])
    token_indices = np.linspace(0, x.shape[1] - 1, token_count, dtype=np.int64)
    x = x[:, token_indices]
    nblocks = cols // 32
    standard = quantize_q4_0(data).reshape(rows, nblocks, 18)
    scales = standard[:, :, :2].copy().reshape(-1).view(np.float16).reshape(rows, nblocks)
    scales = scales.astype(np.float32)
    packed_q = standard[:, :, 2:]
    quants = np.concatenate((packed_q & 0x0F, packed_q >> 4), axis=2).astype(np.uint8)
    dequantized = scales[:, :, None] * (quants.astype(np.float32) - np.float32(8.0))
    weight_blocks = data.reshape(rows, nblocks, 32)
    x_blocks = x.reshape(nblocks, 32, token_count)
    residual = np.einsum(
        "rbk,bkt->rt", dequantized[:output_rows] - weight_blocks[:output_rows], x_blocks,
        optimize=True,
    )
    baseline_cost = float(np.sum(residual * residual, dtype=np.float64))

    for block_index in range(nblocks):
        block = weight_blocks[:output_rows, block_index]
        x_block = x_blocks[block_index]
        old_error = dequantized[:output_rows, block_index] - block
        residual_without_block = residual - old_error @ x_block
        max_index = np.argmax(np.abs(block), axis=1)
        max_value = block[np.arange(output_rows), max_index]
        base_scale = max_value / np.float32(-8.0)
        # Match ggml's Q4_0 contract: choose codes using the F32 scale, then
        # store that scale as F16.  The stored F16 value is what dequantizes at
        # runtime and therefore what the residual objective evaluates.
        candidate_scale_f32 = base_scale[None, :] * multipliers[:, None]
        candidate_scale = candidate_scale_f32.astype(np.float16)
        dequant_scale_f32 = candidate_scale.astype(np.float32)
        reciprocal = np.divide(
            np.float32(1.0), candidate_scale_f32,
            out=np.zeros_like(candidate_scale_f32), where=candidate_scale_f32 != 0,
        )
        candidate_q = np.minimum(
            15,
            np.trunc(block[None, :, :] * reciprocal[:, :, None] + np.float32(8.5)).astype(np.int16),
        ).astype(np.uint8)
        candidate_error = dequant_scale_f32[:, :, None] * (
            candidate_q.astype(np.float32) - np.float32(8.0)
        ) - block[None, :, :]
        candidate_projection = (candidate_error.reshape(-1, 32) @ x_block).reshape(
            len(multipliers), output_rows, token_count)
        costs = np.sum((residual_without_block[None, :, :] + candidate_projection) ** 2,
                       axis=2, dtype=np.float64)
        best = np.argmin(costs, axis=0)
        selected_q = candidate_q[best, np.arange(output_rows)]
        selected_scale = candidate_scale[best, np.arange(output_rows)]
        residual = residual_without_block + candidate_projection[best, np.arange(output_rows)]
        quants[:output_rows, block_index] = selected_q
        scales[:output_rows, block_index] = selected_scale.astype(np.float32)

    optimized_cost = float(np.sum(residual * residual, dtype=np.float64))
    packed = standard.copy()
    selected_scale_bytes = scales.astype(np.float16).reshape(-1).view(np.uint8).reshape(rows, nblocks, 2)
    packed[:, :, :2] = selected_scale_bytes
    packed[:, :, 2:] = quants[:, :, :16] | (quants[:, :, 16:] << 4)
    report = {
        "activation_weighted_sse": optimized_cost,
        "baseline_activation_weighted_sse": baseline_cost,
        "relative_sse": optimized_cost / baseline_cost if baseline_cost else 1.0,
        "activation_tokens": float(token_count),
        "calibrated_output_rows": float(output_rows),
    }
    return packed.reshape(-1), report


def quantize_q4_1(w: np.ndarray) -> np.ndarray:
    """Q4_1: blocks of 32 floats -> fp16 scale/min + 16 nibbles."""
    blocks = np.ascontiguousarray(w, dtype=np.float32).reshape(-1, 32)
    lo = blocks.min(axis=1)
    hi = blocks.max(axis=1)
    scale = (hi - lo) / np.float32(15.0)
    reciprocal = np.divide(np.float32(1.0), scale, out=np.zeros_like(scale), where=scale != 0)
    quants = np.minimum(
        15,
        np.trunc((blocks - lo[:, None]) * reciprocal[:, None] + np.float32(0.5)).astype(np.int16),
    ).astype(np.uint8)
    out = np.empty((blocks.shape[0], 20), dtype=np.uint8)
    out[:, :2] = scale.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:4] = lo.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 4:] = quants[:, :16] | (quants[:, 16:] << 4)
    return out.reshape(-1)


def quantize_q4_k(w: np.ndarray) -> np.ndarray:
    """Q4_K using the version-matched ggml reference quantizer.

    Q4_K has a 256-element super-block whose packing is intentionally owned
    by ggml.  Calling its exported reference routine avoids maintaining a
    duplicate binary format in this converter.
    """
    data = np.ascontiguousarray(w, dtype=np.float32)
    if data.shape[-1] % 256:
        raise ValueError("Q4_K requires the innermost tensor dimension to be divisible by 256")
    override = os.environ.get("SAM3D_GGML_BASE_LIB")
    candidates = ([Path(override)] if override else []) + list(
        Path(__file__).resolve().parents[1].glob("build-*/lib/libggml-base.so.0")
    )
    library = next((path for path in candidates if path.is_file()), None)
    if library is None:
        raise RuntimeError(
            "Q4_K conversion requires a built ggml base library; set SAM3D_GGML_BASE_LIB "
            "or build cpp_ggml/build-cuda or cpp_ggml/build-vulkan first"
        )
    quantize = ctypes.CDLL(str(library)).quantize_row_q4_K_ref
    quantize.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
    quantize.restype = None
    packed = np.empty(data.size // 256 * 144, dtype=np.uint8)
    quantize(data.ctypes.data, packed.ctypes.data, data.size)
    return packed


def convert_tensor(w, dtype: str, *, q4_activations: np.ndarray | None = None,
                   q4_scale_multipliers: tuple[float, ...] | None = None,
                   q4_output_rows: int | None = None,
                   q4_token_limit: int | None = None) -> np.ndarray:
    """Accepts torch.Tensor or np.ndarray; returns the stored numpy array."""
    is_torch = hasattr(w, "detach")
    ndim = w.dim() if is_torch else w.ndim
    # keep small per-channel params (bias, norms) in F32: binary ops between
    # the F32 activations and F16 biases are not supported by ggml backends
    if ndim <= 1:
        if is_torch:
            return w.detach().float().numpy()
        return np.asarray(w, dtype=np.float32)
    # Q4_0/Q4_1/Q8_0 require the fastest dim (ggml ne[0], i.e. in-features) to be a
    # multiple of the 32-wide quant block; otherwise fall back to F16
    block_size = 256 if dtype == "q4_k" else 32
    if dtype in ("q4_0", "q4_1", "q4_k", "q8_0") and w.shape[-1] % block_size != 0:
        dtype = "f16"
    if is_torch:
        w = w.detach()
        if dtype == "f32":
            return w.float().numpy()
        if dtype == "f16":
            return w.float().half().numpy()
        if dtype == "q4_0":
            if q4_activations is not None:
                packed, _ = quantize_q4_0_activation_aware(
                    w.float().numpy(), q4_activations,
                    q4_scale_multipliers or (0.75, 0.875, 1.0, 1.125, 1.25),
                    q4_output_rows, q4_token_limit,
                )
                return packed
            return quantize_q4_0(w.float().numpy())
        if dtype == "q4_1":
            return quantize_q4_1(w.float().numpy())
        if dtype == "q4_k":
            return quantize_q4_k(w.float().numpy())
        return quantize_q8_0(w.float().numpy())
    if dtype == "f32":
        return w.astype(np.float32)
    if dtype == "f16":
        return w.astype(np.float16)
    if dtype == "q4_0":
        if q4_activations is not None:
            packed, _ = quantize_q4_0_activation_aware(
                w.astype(np.float32), q4_activations,
                q4_scale_multipliers or (0.75, 0.875, 1.0, 1.125, 1.25),
                q4_output_rows, q4_token_limit,
            )
            return packed
        return quantize_q4_0(w.astype(np.float32))
    if dtype == "q4_1":
        return quantize_q4_1(w.astype(np.float32))
    if dtype == "q4_k":
        return quantize_q4_k(w.astype(np.float32))
    return quantize_q8_0(w.astype(np.float32))


def write_tensor(w, name, t, dtype, *, q4_activations: np.ndarray | None = None,
                 q4_scale_multipliers: tuple[float, ...] | None = None,
                 q4_output_rows: int | None = None,
                 q4_token_limit: int | None = None):
    """convert_tensor + add_tensor, with explicit quantized raw dtype/shape.

    gguf-py >= 0.19 rejects raw uint8 payloads unless raw_dtype is passed;
    raw_shape must be the per-row *byte* shape, which gguf-py converts back
    to the logical element shape when writing the GGUF header."""
    arr = convert_tensor(t, dtype, q4_activations=q4_activations,
                         q4_scale_multipliers=q4_scale_multipliers,
                         q4_output_rows=q4_output_rows, q4_token_limit=q4_token_limit)
    if arr.dtype == np.uint8:
        logical = tuple(t.shape) if hasattr(t, "detach") else np.asarray(t).shape
        if dtype == "q4_0":
            bytes_per_block = 18
            raw_dtype = gguf.GGMLQuantizationType.Q4_0
        elif dtype == "q4_1":
            bytes_per_block = 20
            raw_dtype = gguf.GGMLQuantizationType.Q4_1
        elif dtype == "q4_k":
            bytes_per_block = 144
            raw_dtype = gguf.GGMLQuantizationType.Q4_K
        elif dtype == "q8_0":
            bytes_per_block = 34
            raw_dtype = gguf.GGMLQuantizationType.Q8_0
        else:
            raise ValueError(f"packed tensor without a quantized dtype: {name}")
        block_size = 256 if dtype == "q4_k" else 32
        byte_shape = tuple(logical[:-1]) + (logical[-1] // block_size * bytes_per_block,)
        w.add_tensor(name, arr, raw_shape=byte_shape,
                     raw_dtype=raw_dtype)
    else:
        w.add_tensor(name, arr)


CEMB_PREFIX = "_base_models.condition_embedder."
DIT_PREFIX = "_base_models.generator.reverse_fn.backbone."


def rewrite_key(key: str):
    """Map a checkpoint state_dict key to its GGUF name (or None to skip)."""
    if key.startswith(CEMB_PREFIX + "idx_emb"):
        return "cemb.idx_emb"
    if key.startswith(CEMB_PREFIX):
        rest = key[len(CEMB_PREFIX):]
        m = re.match(r"module_list\.(\d+)\.(.*)", rest)
        if m:
            return f"cemb.emb{m.group(1)}.{m.group(2)}"
        m = re.match(r"projection_nets\.(\d+)\.0\.(.*)", rest)
        if m:
            return f"cemb.emb{m.group(1)}.proj_ln.{m.group(2)}"
        m = re.match(r"projection_nets\.(\d+)\.1\.(.*)", rest)
        if m:
            return f"cemb.emb{m.group(1)}.proj.{m.group(2)}"
        return None
    if key.startswith(DIT_PREFIX):
        return "dit." + key[len(DIT_PREFIX):]
    if key.startswith("_base_models."):
        return None  # decoder copies / unrelated entries
    # plain checkpoints (ss_decoder / slat_decoder_gs): handled per model
    return key


def sparse_conv_weight(w: torch.Tensor) -> torch.Tensor:
    """spconv SubMConv3d weight (Cout, KD, KH, KW, Cin) -> (Cout, 27*Cin).

    The checkpoint stores kernel offsets before in-channels, so the
    C-contiguous flatten already yields each output row as
    [off0_c0..off0_cCin, off1_c0..] with the 27 offsets iterating in the
    same row-major (D, H, W) order as torch."""
    cout, k, cin = w.shape[0], w.shape[1], w.shape[4]
    return w.reshape(cout, k * k * k * cin)


def load_state_dict(ckpt_path: str) -> dict:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise RuntimeError(f"unexpected checkpoint format: {ckpt_path}")
    return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}


def sval(s: str) -> gguf.GGUFValue:
    return gguf.GGUFValue(s, gguf.GGUFValueType.STRING)


def add_common_kv(w, model, dtype):
    w.add_uint32("sam3d.graph_format", 1)
    w.add_string("sam3d.model", model)
    w.add_string("sam3d.dtype", dtype)


def detect_embedders(yconf):
    cemb = yconf["module"]["condition_embedder"]["backbone"]
    out = []
    pos_groups = []
    for entry in cemb["embedder_list"]:
        opts, kwargs_info = entry[0], entry[1]
        target = opts["_target_"]
        etype = "dino" if target.endswith("dino.Dino") else "pointmap"
        groups = [g for _, g in kwargs_info]
        for g in groups:
            if g not in pos_groups:
                pos_groups.append(g)
        out.append({"type": etype, "opts": opts,
                    "kwargs": [k for k, _ in kwargs_info], "groups": groups})
    return out, pos_groups, cemb


def write_embedder_kv(w, embedders, pos_groups, cemb_conf):
    w.add_uint32("cemb.n_embedders", len(embedders))
    w.add_uint32("cemb.n_pos_groups", len(pos_groups))
    w.add_array("cemb.pos_groups", pos_groups)
    w.add_float32("cemb.proj_multiplier",
                  float(cemb_conf.get("projection_net_hidden_dim_multiplier", 4.0)))
    for i, e in enumerate(embedders):
        w.add_string(f"cemb.emb{i}.type", e["type"])
        w.add_array(f"cemb.emb{i}.inputs", e["kwargs"])
        w.add_array(f"cemb.emb{i}.pos_groups", e["groups"])
        if e["type"] == "dino":
            w.add_uint32(f"cemb.emb{i}.input_size", int(e["opts"].get("input_size", 224)))
            w.add_uint32(f"cemb.emb{i}.normalize_images",
                         int(e["opts"].get("normalize_images", True)))
            w.add_uint32(f"cemb.emb{i}.prenorm_features",
                         int(e["opts"].get("prenorm_features", False)))
            # dinov2_vitl14_reg geometry
            w.add_uint32(f"cemb.emb{i}.embed_dim", 1024)
            w.add_uint32(f"cemb.emb{i}.depth", 24)
            w.add_uint32(f"cemb.emb{i}.num_heads", 16)
            w.add_uint32(f"cemb.emb{i}.patch_size", 14)
            w.add_uint32(f"cemb.emb{i}.reg_tokens", 4)
        else:
            w.add_uint32(f"cemb.emb{i}.input_size", int(e["opts"].get("input_size", 256)))
            w.add_uint32(f"cemb.emb{i}.patch_size", int(e["opts"].get("patch_size", 8)))
            w.add_uint32(f"cemb.emb{i}.embed_dim", int(e["opts"].get("embed_dim", 512)))
            w.add_uint32(f"cemb.emb{i}.num_heads", 16)
            w.add_string(f"cemb.emb{i}.remap", e["opts"].get("remap_output", "exp"))
            w.add_uint32(f"cemb.emb{i}.n_blocks", 1)
            w.add_float32(f"cemb.emb{i}.mlp_ratio", 2.0)


def finish_gguf(w):
    """Write header + kv + tensors to disk (gguf >= 0.19 explicit protocol)."""
    w.open_output_file()
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
