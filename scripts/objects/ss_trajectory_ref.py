#!/usr/bin/env python3
"""Replay native SS inputs through the pinned SAM 3D Objects generator.

Adapted from Asher-1/sam-3d-objects-ggml at
1c14b7c3c3e8d9109b943ddc83a0a39c73744246.  The model implementation and
weights remain Meta's SAM 3D Objects reference.  This script only provides a
numerical boundary oracle; it is not part of native inference.
"""

import argparse
import os
import struct

import numpy as np
import torch


SAMT_MAGIC = b"SAMT"
MODALITIES = [
    "6drotation_normalized",
    "scale",
    "shape",
    "translation",
    "translation_scale",
]


def load_samt(path):
    with open(path, "rb") as stream:
        if stream.read(4) != SAMT_MAGIC:
            raise ValueError(f"{path}: invalid SAMT header")
        (n_dims,) = struct.unpack("<i", stream.read(4))
        shape = struct.unpack(f"<{n_dims}q", stream.read(8 * n_dims))
        (value_type,) = struct.unpack("<i", stream.read(4))
        if value_type != 0:
            raise ValueError(f"{path}: expected an F32 tensor")
        data = np.frombuffer(stream.read(), dtype="<f4")
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path}: shape {shape} does not match {data.size} values")
    return shape, data


def write_samt_f32(path, shape, values):
    data = np.ascontiguousarray(values, dtype="<f4")
    with open(path, "wb") as stream:
        stream.write(SAMT_MAGIC)
        stream.write(struct.pack("<i", len(shape)))
        stream.write(struct.pack(f"<{len(shape)}q", *shape))
        stream.write(struct.pack("<i", 0))
        stream.write(data.tobytes())


def load_generator(checkpoint, config, device):
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    import sam3d_objects  # noqa: F401 - registers Hydra targets
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    generator = instantiate(OmegaConf.load(config)["module"]["generator"]["backbone"])
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = state_dict.get("state_dict", state_dict)
    prefix = "_base_models.generator."
    state_dict = {
        name[len(prefix) :]: value
        for name, value in state_dict.items()
        if name.startswith(prefix)
    }
    missing, unexpected = generator.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing={missing}, unexpected={unexpected}")
    return generator.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=25,
                        help="full Euler schedule length")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop after this many leading schedule steps")
    parser.add_argument("--rescale-t", type=float, default=3.0)
    parser.add_argument("--cfg-strength", type=float, default=7.0)
    parser.add_argument("--cfg-start", type=float, default=0.0)
    parser.add_argument("--cfg-end", type=float, default=500.0)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.steps <= 0 or args.max_steps is not None and not 1 <= args.max_steps <= args.steps:
        parser.error("step counts must satisfy 1 <= max-steps <= steps")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    generator = load_generator(args.checkpoint, args.config, args.device)
    wbb = generator.reverse_fn.backbone
    x = {}
    for modality in MODALITIES:
        shape, data = load_samt(os.path.join(args.e2e_dir, f"ss_x0_{modality}.samt"))
        if len(shape) != 2:
            raise ValueError(f"{modality}: expected native [channels,tokens] shape")
        # GGML stores ne[0] contiguously. PyTorch consumes [batch,tokens,channels].
        x[modality] = torch.from_numpy(data.reshape(shape[::-1]).copy()).unsqueeze(0).to(args.device)

    _, cond_data = load_samt(os.path.join(args.e2e_dir, "ss_cond_tokens.samt"))
    cond = torch.from_numpy(cond_data.reshape(1, -1, 1024).copy()).to(args.device)
    u = torch.linspace(0.0, 1.0, args.steps + 1, device=args.device)
    times = u / (1.0 + (args.rescale_t - 1.0) * (1.0 - u)) if args.rescale_t else u
    stop = args.max_steps or args.steps

    with torch.no_grad():
        for step, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
            if step >= stop:
                break
            t = (t0 * 1000.0).reshape(1)
            d = torch.zeros_like(t)
            vc = wbb(x, t, cond, d=d, cfg=False)
            cfg_active = args.cfg_start <= float(t) <= args.cfg_end
            vu = wbb(x, t, cond, d=d, cfg=True) if cfg_active else None
            for modality in MODALITIES:
                values = vc[modality][0].float().cpu().numpy()
                write_samt_f32(
                    os.path.join(args.out_dir, f"ss_torch_vc{step}_{modality}.samt"),
                    [values.shape[-1], values.shape[-2]], values)
                if vu is not None:
                    values_u = vu[modality][0].float().cpu().numpy()
                    write_samt_f32(
                        os.path.join(args.out_dir, f"ss_torch_vu{step}_{modality}.samt"),
                        [values_u.shape[-1], values_u.shape[-2]], values_u)
                velocity = vc[modality]
                if cfg_active and modality == "shape":
                    velocity = vc[modality] + args.cfg_strength * (vc[modality] - vu[modality])
                x[modality] = x[modality] + (t1 - t0) * velocity
                state = x[modality][0].float().cpu().numpy()
                write_samt_f32(
                    os.path.join(args.out_dir, f"ss_torch_state{step + 1}_{modality}.samt"),
                    [state.shape[-1], state.shape[-2]], state)
            print(f"step {step + 1:02d}/{args.steps}: t={float(t):.6f} "
                  f"dt={float(t1 - t0):.9f} cfg={int(cfg_active)}", flush=True)


if __name__ == "__main__":
    main()
