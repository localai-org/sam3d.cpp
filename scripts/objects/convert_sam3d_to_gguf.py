#!/usr/bin/env python3
# Convert official SAM 3D Objects checkpoints (PyTorch) to GGUF for the
# cpp_ggml C++ inference engine.
#
# Supported models:
#   ss_generator       EmbedderFuser conditioners + MOT sparse-structure DiT
#   ss_decoder         dense 3D-conv occupancy decoder
#   slat_generator     EmbedderFuser conditioners + sparse structured-latent DiT
#   slat_decoder_gs    windowed sparse transformer -> Gaussian splat parameters
#   slat_decoder_gs_4  same arch as slat_decoder_gs, num_gaussians=4 variant
#   slat_decoder_mesh  windowed sparse transformer -> mesh vertex/face params
#   moge_vitl          MoGe ViT-L image -> point-map model used by pipeline.yaml
#
# Usage:
#   python3 convert_sam3d_to_gguf.py --checkpoint-dir checkpoints/hf \
#       --model all --dtype f16 --output cpp_ggml/models/gguf
#
# GGUF tensor naming schema consumed by cpp_ggml/src:
#   cemb.emb{i}.*               condition embedder backbones (DINO / PointPatch)
#   cemb.emb{i}.proj_ln.*       fuser projection pre-norm
#   cemb.emb{i}.proj.{w1,w2,w3} fuser llama3 FeedForward projection
#   cemb.idx_emb                learned modality positional embedding
#   dit.*                       generator backbone (dense / sparse DiT)
#   dec.*                       SS decoder (dense 3D conv)
#   gsdec.*                     Gaussian decoder (windowed sparse transformer)
#   meshdec.*                   Mesh decoder (windowed sparse transformer)
#   moge.*                      MoGe ViT-L backbone and point/mask heads
import argparse
import hashlib
import json
import os
import re
import struct

import numpy as np

import yaml
import gguf
import torch

from gguf_schema import (convert_tensor, rewrite_key, load_state_dict, sval,
                         write_tensor, add_common_kv, detect_embedders,
                         write_embedder_kv, finish_gguf, quantize_q4_0,
                         sparse_conv_weight)


def load_q4_calibrations(config_path):
    """Load named Q4_0 activation samples from a reproducible JSON config."""
    if not config_path:
        return {}
    config_path = os.path.abspath(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    tensors = config.get("tensors", {})
    if not isinstance(tensors, dict):
        raise ValueError("Q4 calibration config requires a 'tensors' object")
    calibrations = {}
    for name, spec in tensors.items():
        if not isinstance(spec, dict) or "activation_samt" not in spec:
            raise ValueError(f"Q4 calibration entry for {name} requires activation_samt")
        sample_path = spec["activation_samt"]
        if not os.path.isabs(sample_path):
            sample_path = os.path.join(os.path.dirname(config_path), sample_path)
        with open(sample_path, "rb") as f:
            header = f.read(8)
            magic, ndim = struct.unpack("<Ii", header)
            if magic != 0x544D4153 or ndim != 2:
                raise ValueError(f"{sample_path} is not a rank-2 SAMT tensor")
            shape = struct.unpack("<2q", f.read(16))
            (tensor_type,) = struct.unpack("<i", f.read(4))
            if tensor_type != 0:
                raise ValueError(f"{sample_path} must contain F32 activations, got type {tensor_type}")
            flat = np.frombuffer(f.read(), dtype="<f4", count=shape[0] * shape[1])
        if flat.size != shape[0] * shape[1]:
            raise ValueError(f"truncated activation sample: {sample_path}")
        # ggml stores ne[0] as the contiguous feature dimension.  A linear
        # weight is (out_features, in_features), so expose X as (in, tokens).
        activations = flat.reshape(shape[1], shape[0]).T.copy()
        multipliers = tuple(float(v) for v in spec.get(
            "scale_multipliers", (0.75, 0.875, 1.0, 1.125, 1.25)))
        output_rows = spec.get("output_rows")
        token_limit = spec.get("token_limit")
        if output_rows is not None and int(output_rows) <= 0:
            raise ValueError(f"output_rows for {name} must be positive")
        if token_limit is not None and int(token_limit) <= 0:
            raise ValueError(f"token_limit for {name} must be positive")
        calibrations[name] = (activations, multipliers, os.path.abspath(sample_path),
                              int(output_rows) if output_rows is not None else None,
                              int(token_limit) if token_limit is not None else None)
    return calibrations


def apply_weight_overrides(state_dict, override_path):
    """Overlay a sparse checkpoint while preserving the official checkpoint contract.

    QAT emits only the trained master weights.  Keeping that artifact sparse
    avoids copying a multi-gigabyte source checkpoint and prevents an export
    from silently using a stale full checkpoint.  Every override must name an
    existing tensor with an identical shape.
    """
    if not override_path:
        return state_dict
    overrides = load_state_dict(override_path)
    if not overrides:
        raise ValueError(f"weight override checkpoint is empty: {override_path}")
    merged = dict(state_dict)
    for name, value in overrides.items():
        reference = merged.get(name)
        if reference is None:
            raise ValueError(f"weight override does not exist in source checkpoint: {name}")
        if tuple(reference.shape) != tuple(value.shape):
            raise ValueError(
                f"weight override shape mismatch for {name}: "
                f"{tuple(value.shape)} vs {tuple(reference.shape)}")
        merged[name] = value
    print(f"[overrides] {len(overrides)} tensor(s) from {override_path}", flush=True)
    return merged


def load_q4_low_rank_residuals(config_path):
    """Load explicit GGUF weight names and ranks for Q4 residual factors."""
    if not config_path:
        return {}
    config_path = os.path.abspath(config_path)
    with open(config_path, encoding="utf-8") as stream:
        config = json.load(stream)
    tensors = config.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("low-rank residual config requires a non-empty 'tensors' object")
    result = {}
    for name, spec in tensors.items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            raise ValueError("each low-rank residual entry must be a tensor name and object")
        rank = spec.get("rank")
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"low-rank residual {name!r} requires a positive integer rank")
        result[name] = rank
    return result


def load_q4_low_rank_residual_overrides(checkpoint_path):
    """Load trainable residual factors emitted by q4_ss_trajectory_qat.py."""
    if not checkpoint_path:
        return {}
    state_dict = load_state_dict(checkpoint_path)
    prefix = "_base_models.generator."
    factors: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix) or not (key.endswith(".lora_a") or key.endswith(".lora_b")):
            raise ValueError(f"invalid Q4 low-rank residual factor name: {key}")
        suffix = ".lora_a" if key.endswith(".lora_a") else ".lora_b"
        gname = rewrite_key(prefix + key[len(prefix):-len(suffix)])
        if gname is None or not gname.endswith(".weight"):
            raise ValueError(f"low-rank residual factor does not name a generator weight: {key}")
        factors.setdefault(gname, {})[suffix[-1]] = value.detach().float().cpu().contiguous()
    missing = sorted(name for name, pair in factors.items() if set(pair) != {"a", "b"})
    if missing:
        raise ValueError("low-rank residual checkpoint has an incomplete factor pair: " + ", ".join(missing))
    if not factors:
        raise ValueError(f"low-rank residual checkpoint is empty: {checkpoint_path}")
    return factors


def q4_0_low_rank_residual_factors(weight: torch.Tensor, rank: int):
    """Factor the exact Q4_0 encoding residual as ``B @ A`` in F16.

    The decomposition is derived from the bytes that the converter writes,
    not an approximate fake quantizer.  ``A`` has shape ``(rank, in)`` and
    ``B`` has shape ``(out, rank)`` so the graph can evaluate
    ``Wq @ x + B @ (A @ x)`` without dequantizing ``Wq``.
    """
    if weight.ndim != 2:
        raise ValueError(f"low-rank residual requires a rank-2 weight, got {tuple(weight.shape)}")
    source = weight.detach().float().cpu().contiguous()
    out_features, in_features = source.shape
    if in_features % 32:
        raise ValueError(f"low-rank residual requires a 32-aligned input, got {in_features}")
    if rank > min(out_features, in_features):
        raise ValueError(
            f"low-rank residual rank {rank} exceeds {min(out_features, in_features)} for "
            f"{tuple(source.shape)}")

    packed = quantize_q4_0(source.numpy()).reshape(out_features, in_features // 32, 18)
    scales = np.ascontiguousarray(packed[..., :2]).view(np.float16).reshape(out_features, -1)
    nibbles = packed[..., 2:]
    codes = np.concatenate((nibbles & 0x0F, nibbles >> 4), axis=-1).astype(np.float32)
    decoded = (scales.astype(np.float32)[..., None] * (codes - np.float32(8.0))).reshape(source.shape)
    residual = source - torch.from_numpy(decoded)
    u, singular, vh = torch.linalg.svd(residual, full_matrices=False)
    root = singular[:rank].sqrt()
    factor_a = (root[:, None] * vh[:rank]).contiguous()
    factor_b = (u[:, :rank] * root[None, :]).contiguous()
    return factor_a.half(), factor_b.half()


def q4_low_rank_factor_names(weight_name: str) -> tuple[str, str]:
    """Choose GGUF-safe names for optional Q4 residual factors.

    ggml rejects tensor names of 64 bytes or longer.  Preserve the descriptive
    historical suffixes whenever they fit, and use the compact aliases only
    for the long modality names in this model.
    """
    long_names = (weight_name + ".lora_a", weight_name + ".lora_b")
    if all(len(name.encode("utf-8")) < 64 for name in long_names):
        return long_names
    short_names = (weight_name + ".ra", weight_name + ".rb")
    if not all(len(name.encode("utf-8")) < 64 for name in short_names):
        raise ValueError(f"low-rank residual tensor name exceeds GGUF limit: {weight_name}")
    return short_names


def convert_generator(name, ckpt_dir, out_dir, dtype, is_ss, keep_f16_patterns=(),
                      quantize_matrix_only=False, q4_calibration_config=None,
                      weight_overrides=None, q4_low_rank_residuals=None,
                      q4_low_rank_residual_overrides=None):
    sd = load_state_dict(os.path.join(ckpt_dir, name + ".ckpt"))
    sd = apply_weight_overrides(sd, weight_overrides)
    yconf = yaml.safe_load(open(os.path.join(ckpt_dir, name + ".yaml")))
    embedders, pos_groups, cemb_conf = detect_embedders(yconf)
    rf = yconf["module"]["generator"]["backbone"]["reverse_fn"]["backbone"]

    arch = "sam3d.ss" if is_ss else "sam3d.slat"
    out_path = os.path.join(out_dir, f"{name}-{dtype}.gguf")
    w = gguf.GGUFWriter(out_path, arch)
    add_common_kv(w, name, dtype)
    if dtype == "q4_k":
        w.add_string("sam3d.quantization.q4_k_policy",
                     "matrix .weight tensors with a 256-aligned input dimension; other tensors are F16")
    if quantize_matrix_only:
        w.add_string("sam3d.quantization.matrix_only_policy",
                     "only rank-2-or-greater .weight tensors are quantized; other tensors are F16")
    if keep_f16_patterns:
        w.add_array("sam3d.quantization.keep_f16_regex",
                    [pattern.pattern for pattern in keep_f16_patterns])
    calibrations = load_q4_calibrations(q4_calibration_config)
    low_rank_residuals = q4_low_rank_residuals or {}
    low_rank_residual_overrides = q4_low_rank_residual_overrides or {}
    if (low_rank_residuals or low_rank_residual_overrides) and not is_ss:
        raise ValueError("low-rank residuals are currently consumed only by the SS flow graph")
    if calibrations:
        if dtype != "q4_0":
            raise ValueError("activation-aware calibration is only valid for Q4_0 output")
        w.add_array("sam3d.quantization.activation_aware_q4_tensors", sorted(calibrations))
    write_embedder_kv(w, embedders, pos_groups, cemb_conf)

    # ---- DiT metadata (shared keys)
    w.add_uint32("dit.model_channels", int(rf["model_channels"]))
    w.add_uint32("dit.num_blocks", int(rf["num_blocks"]))
    w.add_uint32("dit.num_heads", int(rf["num_heads"]))
    w.add_float32("dit.mlp_ratio", float(rf["mlp_ratio"]))
    w.add_uint32("dit.cond_channels", int(rf["cond_channels"]))
    w.add_uint32("dit.qk_rms_norm", int(rf["qk_rms_norm"]))
    w.add_uint32("dit.pe_ape", int(rf["pe_mode"] == "ape"))
    w.add_uint32("dit.in_channels", int(rf["in_channels"]))
    w.add_uint32("dit.out_channels", int(rf["out_channels"]))

    if is_ss:
        # MOT metadata
        w.add_uint32("dit.is_shortcut", int(rf.get("is_shortcut_model", False)))
        res, ps = int(rf["resolution"]), int(rf["patch_size"])
        w.add_uint32("dit.shape_tokens", (res // ps) ** 3)
        merged = rf.get("latent_share_transformer", {})
        w.add_array("dit.merged_names", list(merged))
        members = [m for ms in merged.values() for m in ms]
        w.add_array("dit.merged_members", members)
        singles = [n for n in rf["latent_mapping"] if n not in members]
        w.add_array("dit.single_names", singles)
        for lname, lat in rf["latent_mapping"].items():
            pe = lat.get("pos_embedder", {})
            tl = int(pe["token_len"]) if "token_len" in pe else (res // ps) ** 3
            w.add_uint32(f"dit.latent.{lname}.token_len", tl)
            w.add_uint32(f"dit.latent.{lname}.in_channels", int(lat["in_channels"]))
            w.add_uint32(f"dit.latent.{lname}.pos_learnt",
                         int(str(pe.get("_target_", "")).endswith("LearntPositionEmbedder")))
    else:
        # sparse SLat metadata
        w.add_uint32("dit.io_block_channels", int(rf["io_block_channels"][0]))
        w.add_uint32("dit.num_io_res_blocks", int(rf["num_io_res_blocks"]))
        w.add_uint32("dit.use_skip_connection", int(rf.get("use_skip_connection", True)))
        w.add_uint32("dit.resolution", int(rf["resolution"]))

    # ---- tensors
    n = 0
    matched_keep_f16: list[str] = []
    written_low_rank_residuals: dict[str, int] = {}
    for key, tensor in sd.items():
        gname = rewrite_key(key)
        if gname is None:
            continue
        # PointPatchEmbed pos_embed (1, D, H, W): pre-permute to token-major
        # (H*W, D) rows (token = h*H_n + w) so the C++ graph adds it directly
        # without a transposing permute.
        if gname.endswith(".pos_embed") and tensor.ndim == 4:
            tensor = tensor[0].permute(1, 2, 0).reshape(-1, tensor.shape[1]).contiguous()
        # rewrite spconv weights to (Cout, 27*Cin) row format; the sparse
        # conv sits inside a SparseSequential, hence the .conv.conv.weight keys
        if (gname.startswith("dit.input_blocks.") or gname.startswith("dit.out_blocks.")) \
                and (gname.endswith(".conv1.weight") or gname.endswith(".conv2.weight")
                     or gname.endswith(".conv1.conv.weight")
                     or gname.endswith(".conv2.conv.weight")):
            tensor = sparse_conv_weight(tensor)
        matches_keep_f16 = any(pattern.search(gname) for pattern in keep_f16_patterns)
        if matches_keep_f16:
            matched_keep_f16.append(gname)
        force_f16 = matches_keep_f16
        if quantize_matrix_only:
            # Normalization vectors, learned tables and biases feed
            # elementwise operations. Preserving them isolates GEMM
            # quantization error and avoids avoidable diffusion drift.
            force_f16 = force_f16 or not (gname.endswith(".weight") and tensor.ndim >= 2)
        if dtype == "q4_k":
            # Keep scalar/vector parameters and non-projection tables in F16.
            # They feed elementwise graph operators, while only aligned matrix
            # weights can use Q4_K's 256-element super-block GEMM directly.
            shape = tensor.shape
            is_q4_k_matrix = (
                gname.endswith(".weight")
                and len(shape) >= 2
                and shape[-1] % 256 == 0
            )
            force_f16 = force_f16 or not is_q4_k_matrix
        tensor_dtype = "f16" if force_f16 else dtype
        calibration = calibrations.get(gname)
        if calibration and tensor_dtype != "q4_0":
            raise ValueError(f"calibrated tensor is not Q4_0: {gname} ({tensor_dtype})")
        if calibration:
            activations, multipliers, sample_path, output_rows, token_limit = calibration
            if tensor.ndim != 2 or tensor.shape[1] != activations.shape[0]:
                raise ValueError(
                    f"activation sample {sample_path} does not match {gname}: "
                    f"weight {tuple(tensor.shape)}, activations {activations.shape}")
            print(f"[calibrate] {gname}: {activations.shape[1]} tokens, "
                  f"scales={multipliers}, rows={output_rows or tensor.shape[0]}, "
                  f"sampled_tokens={token_limit or activations.shape[1]}", flush=True)
            write_tensor(w, gname, tensor, tensor_dtype, q4_activations=activations,
                         q4_scale_multipliers=multipliers, q4_output_rows=output_rows,
                         q4_token_limit=token_limit)
        else:
            write_tensor(w, gname, tensor, tensor_dtype)
        n += 1
        rank = low_rank_residuals.get(gname)
        learned_factors = low_rank_residual_overrides.get(gname)
        if rank is not None or learned_factors is not None:
            if dtype != "q4_0" or tensor_dtype != "q4_0":
                raise ValueError(
                    f"low-rank residual {gname} requires a Q4_0 base tensor, got {tensor_dtype}")
            if calibration:
                raise ValueError(
                    f"low-rank residual {gname} cannot be combined with activation-aware Q4 scaling")
            if learned_factors is None:
                factor_a, factor_b = q4_0_low_rank_residual_factors(tensor, rank)
            else:
                factor_a, factor_b = learned_factors["a"], learned_factors["b"]
                if factor_a.ndim != 2 or factor_b.ndim != 2 or \
                        factor_a.shape[1] != tensor.shape[1] or \
                        factor_b.shape[0] != tensor.shape[0] or \
                        factor_a.shape[0] != factor_b.shape[1]:
                    raise ValueError(
                        f"low-rank factor shape mismatch for {gname}: A={tuple(factor_a.shape)}, "
                        f"B={tuple(factor_b.shape)}, weight={tuple(tensor.shape)}")
                rank = int(factor_a.shape[0])
            factor_a_name, factor_b_name = q4_low_rank_factor_names(gname)
            write_tensor(w, factor_a_name, factor_a, "f16")
            write_tensor(w, factor_b_name, factor_b, "f16")
            written_low_rank_residuals[gname] = rank
            n += 2
            print(f"[q4-residual] {gname}: rank={rank}", flush=True)
    if keep_f16_patterns:
        w.add_uint32("sam3d.quantization.keep_f16_tensor_count", len(matched_keep_f16))
        w.add_array("sam3d.quantization.keep_f16_tensor_names", matched_keep_f16)
        print(f"[keep-f16] {name}: {len(matched_keep_f16)} tensor(s) matched", flush=True)
    requested_low_rank_residuals = set(low_rank_residuals) | set(low_rank_residual_overrides)
    missing_low_rank_residuals = sorted(requested_low_rank_residuals - set(written_low_rank_residuals))
    if missing_low_rank_residuals:
        raise ValueError(
            "low-rank residual tensor(s) not exported: " + ", ".join(missing_low_rank_residuals))
    if written_low_rank_residuals:
        names = sorted(written_low_rank_residuals)
        w.add_uint32("sam3d.quantization.low_rank_residual_tensor_count", len(names))
        w.add_array("sam3d.quantization.low_rank_residual_tensor_names", names)
        w.add_array("sam3d.quantization.low_rank_residual_ranks",
                    [written_low_rank_residuals[name] for name in names])
    finish_gguf(w)
    return out_path, n


def convert_ss_decoder(ckpt_dir, out_dir, dtype):
    name = "ss_decoder"
    yconf = yaml.safe_load(open(os.path.join(ckpt_dir, name + ".yaml")))
    sd = load_state_dict(os.path.join(ckpt_dir, name + ".ckpt"))
    out_path = os.path.join(out_dir, f"{name}-{dtype}.gguf")
    w = gguf.GGUFWriter(out_path, "sam3d.ssdec")
    add_common_kv(w, name, dtype)
    w.add_uint32("dec.out_channels", int(yconf["out_channels"]))
    w.add_uint32("dec.latent_channels", int(yconf["latent_channels"]))
    w.add_uint32("dec.num_res_blocks", int(yconf["num_res_blocks"]))
    w.add_uint32("dec.num_res_blocks_middle", int(yconf["num_res_blocks_middle"]))
    w.add_array("dec.channels", [int(c) for c in yconf["channels"]])
    w.add_string("dec.norm_type", yconf.get("norm_type", "layer"))
    w.add_string("dec.conv_layout", "im2col3d-v1")
    max_conv_oc_ic = max(
        int(t.shape[0] * t.shape[1]) for t in sd.values()
        if t.ndim == 5 and t.shape[2:] == (3, 3, 3)
    )
    w.add_uint32("dec.max_conv_oc_ic", max_conv_oc_ic)

    def conv3d_weight(t):
        # PyTorch is (OC, IC, KD, KH, KW).  The C++ IM2COL_3D lowering emits
        # a row [IC, KD, KH, KW], so each output-channel row is contiguous
        # [IC * 27].  GGUF reverses numpy dimensions, yielding ggml ne
        # {27*IC, OC}, which is directly consumable by ggml_mul_mat.
        oc, ic = t.shape[0], t.shape[1]
        return np.ascontiguousarray(t.reshape(oc, ic * 27))

    # Conv biases remain whole; pixel shuffle is built from ggml reshape and
    # permute operators at runtime.
    n = 0
    for key, tensor in sd.items():
        t = tensor
        if key.endswith(".weight") and tensor.ndim == 5:
            t = np.ascontiguousarray(conv3d_weight(tensor.numpy().astype(np.float32)))
        if not isinstance(t, np.ndarray):
            t = t.numpy()
        write_tensor(w, "dec." + key, t, dtype)
        n += 1
    finish_gguf(w)
    return out_path, n


def convert_slat_decoder_gs(ckpt_dir, out_dir, dtype, name="slat_decoder_gs"):
    yconf = yaml.safe_load(open(os.path.join(ckpt_dir, name + ".yaml")))
    sd = load_state_dict(os.path.join(ckpt_dir, name + ".ckpt"))
    out_path = os.path.join(out_dir, f"{name}-{dtype}.gguf")
    w = gguf.GGUFWriter(out_path, "sam3d.gsdec")
    add_common_kv(w, name, dtype)
    w.add_uint32("gsdec.resolution", int(yconf["resolution"]))
    w.add_uint32("gsdec.model_channels", int(yconf["model_channels"]))
    w.add_uint32("gsdec.latent_channels", int(yconf["latent_channels"]))
    w.add_uint32("gsdec.num_blocks", int(yconf["num_blocks"]))
    w.add_uint32("gsdec.num_heads", int(yconf["num_heads"]))
    w.add_float32("gsdec.mlp_ratio", float(yconf["mlp_ratio"]))
    w.add_string("gsdec.attn_mode", yconf.get("attn_mode", "swin"))
    w.add_uint32("gsdec.window_size", int(yconf.get("window_size", 8)))
    w.add_uint32("gsdec.qk_rms_norm", int(yconf.get("qk_rms_norm", False)))
    rep = yconf["representation_config"]
    w.add_uint32("gsdec.num_gaussians", int(rep["num_gaussians"]))
    w.add_float32("gsdec.voxel_size", float(rep["voxel_size"]))
    w.add_uint32("gsdec.perturb_offset", int(rep.get("perturb_offset", True)))
    w.add_float32("gsdec.scaling_bias", float(rep["scaling_bias"]))
    w.add_float32("gsdec.opacity_bias", float(rep["opacity_bias"]))
    w.add_string("gsdec.scaling_activation", rep["scaling_activation"])
    for k in ("_xyz", "_features_dc", "_opacity", "_scaling", "_rotation"):
        w.add_float32(f"gsdec.lr.{k}", float(rep["lr"][k]))
    w.add_float32("gsdec.filter_3d", float(rep["3d_filter_kernel_size"]))
    n = 0
    for key, tensor in sd.items():
        write_tensor(w, "gsdec." + key, tensor, dtype)
        n += 1
    finish_gguf(w)
    return out_path, n


def convert_slat_decoder_mesh(ckpt_dir, out_dir, dtype):
    name = "slat_decoder_mesh"
    yconf = yaml.safe_load(open(os.path.join(ckpt_dir, name + ".yaml")))
    sd = load_state_dict(os.path.join(ckpt_dir, name + ".ckpt"))
    out_path = os.path.join(out_dir, f"{name}-{dtype}.gguf")
    w = gguf.GGUFWriter(out_path, "sam3d.meshdec")
    add_common_kv(w, name, dtype)
    w.add_uint32("meshdec.resolution", int(yconf["resolution"]))
    w.add_uint32("meshdec.model_channels", int(yconf["model_channels"]))
    w.add_uint32("meshdec.latent_channels", int(yconf["latent_channels"]))
    w.add_uint32("meshdec.num_blocks", int(yconf["num_blocks"]))
    w.add_uint32("meshdec.num_heads", int(yconf["num_heads"]))
    w.add_float32("meshdec.mlp_ratio", float(yconf["mlp_ratio"]))
    w.add_string("meshdec.attn_mode", yconf.get("attn_mode", "swin"))
    w.add_uint32("meshdec.window_size", int(yconf.get("window_size", 8)))
    rep = yconf["representation_config"]
    w.add_uint32("meshdec.use_color", int(rep.get("use_color", False)))

    # spconv exposes checkpoint filters as (OC, KD, KH, KW, IC). Its regular
    # submanifold convolution consumes each 3x3 offset as (OC, IC). The CUDA
    # 1x1 fast path in the pinned spconv 2.3.8 build instead interprets the
    # same physical buffer as (IC, OC). Preserve those observed upstream
    # semantics explicitly; the frozen skip captures guard this quirk.
    def conv3d_weight(t):
        # mesh ResBlock convs store weights as (OC, KD, KH, KW, IC)
        oc, k, ic = t.shape[0], t.shape[1], t.shape[4]
        if k == 1:
            matrix = t.reshape(ic, oc).transpose(0, 1).contiguous()
            return matrix.reshape(1, oc * ic)
        return t.permute(1, 2, 3, 0, 4).reshape(k * k * k, oc * ic)

    n = 0
    for key, tensor in sd.items():
        t = tensor
        if key.endswith(".weight") and tensor.ndim == 5:
            t = np.ascontiguousarray(conv3d_weight(tensor.float()))
        write_tensor(w, "meshdec." + key, t, dtype)
        n += 1
    finish_gguf(w)
    return out_path, n


def convert_moge_vitl(checkpoint_path, out_dir, dtype):
    """Convert the exact MoGe checkpoint selected by ``pipeline.yaml``.

    MoGe is not a SAM 3D checkpoint, so it deliberately has a separate input
    path.  Keeping its source explicit prevents the native raw-image runtime
    from depending on a user's Hugging Face cache or Python environment.
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"MoGe checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("MoGe checkpoint must contain a 'model' state dictionary")
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("MoGe checkpoint must contain a 'model_config' dictionary")
    if config.get("encoder") != "dinov2_vitl14":
        raise ValueError(
            "native MoGe contract currently supports only the official "
            f"dinov2_vitl14 checkpoint, got {config.get('encoder')!r}")

    with open(checkpoint_path, "rb") as source_stream:
        source_hash = hashlib.file_digest(source_stream, "sha256").hexdigest()

    out_path = os.path.join(out_dir, f"moge_vitl-{dtype}.gguf")
    writer = gguf.GGUFWriter(out_path, "sam3d.moge")
    add_common_kv(writer, "moge_vitl", dtype)
    writer.add_string("moge.source", "Ruicheng/moge-vitl:model.pt")
    writer.add_string("moge.source_sha256", source_hash)
    writer.add_string("moge.encoder", str(config["encoder"]))
    writer.add_uint32("moge.image_patch_size", 14)
    writer.add_uint32("moge.hidden_size", 1024)
    writer.add_uint32("moge.num_heads", 16)
    writer.add_uint32("moge.num_blocks", 24)
    writer.add_uint32("moge.intermediate_layers", int(config["intermediate_layers"]))
    writer.add_string("moge.remap_output", str(config["remap_output"]))
    writer.add_float32("moge.mask_threshold", float(config.get("mask_threshold", 0.5)))
    writer.add_uint32("moge.dim_proj", int(config.get("dim_proj", 512)))
    writer.add_array("moge.dim_upsample", config["dim_upsample"])
    writer.add_uint32("moge.residual_blocks", int(config["num_res_blocks"]))
    writer.add_uint32("moge.last_conv_channels", int(config["last_conv_channels"]))
    writer.add_uint32("moge.last_conv_size", int(config["last_conv_size"]))

    tensors = checkpoint["model"]
    for key, tensor in tensors.items():
        write_tensor(writer, "moge." + key, tensor, dtype)
    finish_gguf(writer)
    return out_path, len(tensors)


def main():
    ap = argparse.ArgumentParser(description="SAM 3D Objects -> GGUF converter")
    ap.add_argument("--checkpoint-dir", default="checkpoints/hf")
    ap.add_argument("--output", default="cpp_ggml/models/gguf")
    ap.add_argument("--model", default="all",
                    choices=["ss_generator", "ss_decoder", "slat_generator",
                             "slat_decoder_gs", "slat_decoder_gs_4",
                             "slat_decoder_mesh", "moge_vitl", "all"])
    ap.add_argument("--moge-checkpoint", metavar="PATH",
                    help="official Ruicheng/moge-vitl model.pt; required with --model moge_vitl")
    ap.add_argument("--dtype", default="f16", choices=["f32", "f16", "q4_0", "q4_1", "q4_k", "q8_0"])
    ap.add_argument("--keep-f16-regex", action="append", default=[], metavar="REGEX",
                    help="store matching generator tensors as F16; repeatable and recorded in GGUF metadata")
    ap.add_argument("--quantize-matrix-only", action="store_true",
                    help="for generators, quantize only rank-2-or-greater .weight tensors; "
                        "store all other tensors as F16")
    ap.add_argument("--q4-calibration-config", metavar="JSON",
                    help="activation-aware Q4_0 scale-search config; records target tensor names "
                         "and rank-2 F32 SAMT activation samples")
    ap.add_argument("--weight-overrides", metavar="CHECKPOINT",
                    help="sparse torch checkpoint containing replacement generator weights; "
                         "validated against the source checkpoint before GGUF export")
    ap.add_argument("--q4-low-rank-residuals", metavar="JSON",
                    help="Q4_0-only JSON mapping GGUF .weight names to residual ranks; "
                    "stores exact-Q4 residual factors as F16 lora_a/lora_b tensors")
    ap.add_argument("--q4-low-rank-residual-overrides", metavar="CHECKPOINT",
                    help="Q4_0 SS-only trained lora_a/lora_b factor checkpoint from q4_ss_trajectory_qat.py")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    keep_f16_patterns = tuple(re.compile(pattern) for pattern in args.keep_f16_regex)
    models = (["ss_generator", "ss_decoder", "slat_generator", "slat_decoder_gs",
               "slat_decoder_gs_4", "slat_decoder_mesh"]
              if args.model == "all" else [args.model])
    if args.model == "moge_vitl" and not args.moge_checkpoint:
        ap.error("--model moge_vitl requires --moge-checkpoint PATH")
    if args.weight_overrides and any(model not in ("ss_generator", "slat_generator")
                                     for model in models):
        ap.error("--weight-overrides is supported only for ss_generator or slat_generator")
    if (args.q4_low_rank_residuals or args.q4_low_rank_residual_overrides) and \
            (args.dtype != "q4_0" or models != ["ss_generator"]):
        ap.error("--q4-low-rank-residuals is supported only for Q4_0 ss_generator export")
    if args.q4_low_rank_residuals and args.q4_low_rank_residual_overrides:
        ap.error("static residual ranks and trained residual overrides are mutually exclusive")
    low_rank_residuals = load_q4_low_rank_residuals(args.q4_low_rank_residuals)
    low_rank_residual_overrides = load_q4_low_rank_residual_overrides(args.q4_low_rank_residual_overrides)
    for model in models:
        if model in ("ss_generator", "slat_generator"):
            out, n = convert_generator(model, args.checkpoint_dir, args.output,
                                       args.dtype, is_ss=(model == "ss_generator"),
                                       keep_f16_patterns=keep_f16_patterns,
                                       quantize_matrix_only=args.quantize_matrix_only,
                                       q4_calibration_config=args.q4_calibration_config,
                                       weight_overrides=args.weight_overrides,
                                       q4_low_rank_residuals=low_rank_residuals,
                                       q4_low_rank_residual_overrides=low_rank_residual_overrides)
        elif model == "ss_decoder":
            out, n = convert_ss_decoder(args.checkpoint_dir, args.output, args.dtype)
        elif model in ("slat_decoder_gs", "slat_decoder_gs_4"):
            out, n = convert_slat_decoder_gs(args.checkpoint_dir, args.output,
                                             args.dtype, name=model)
        elif model == "slat_decoder_mesh":
            out, n = convert_slat_decoder_mesh(args.checkpoint_dir, args.output, args.dtype)
        else:
            out, n = convert_moge_vitl(args.moge_checkpoint, args.output, args.dtype)
        size_mb = os.path.getsize(out) / 1e6
        print(f"[convert] {model:16s} -> {out} ({n} tensors, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
