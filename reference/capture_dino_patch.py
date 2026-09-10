#!/usr/bin/env python3
"""Original pinned DINOv3 PatchEmbed with stored synthetic F32 weights.

This is operation-contract coverage, NOT trained Body/DINO model parity. The
final output calls the unmodified upstream class. Unfold and bias-free conv are
auxiliary PyTorch diagnostics for native graph boundaries, not upstream taps.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np
import torch
from safetensors.numpy import save_file

SOURCE_HASH = "c592c7262779c8e1789ed502aee81bdd55eba4607065f8d04ae95eea683d0ef1"
REVISION = "6876159a11b4df116f30f667f8c9888617df0751"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.upstream / "dinov3/layers/patch_embed.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError("DINO patch_embed.py does not match audited source")
    spec = importlib.util.spec_from_file_location("official_patch_embed", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(1953)
    args.output.mkdir(parents=True, exist_ok=True)
    # Includes non-square/non-divisible geometry, batch>1 and actual Body H+
    # patch width/output dimension. No learned checkpoint parameters are used.
    shapes = [(1,1,4,4,1,2),(2,3,19,23,7,4),(1,3,64,96,1280,16),
              (1,3,512,512,1280,16)]
    tensors, cases, rules = {}, [], []
    for index, (batch, channels, height, width, outputs, patch) in enumerate(shapes):
        prefix = f"case.{index:04d}"
        image = rng.normal(size=(batch,channels,height,width)).astype(np.float32)
        weight = (rng.normal(size=(outputs,channels,patch,patch)) /
                  np.sqrt(channels*patch*patch)).astype(np.float32)
        bias = rng.normal(scale=.1,size=outputs).astype(np.float32)
        net = module.PatchEmbed(img_size=(height,width), patch_size=patch,
                               in_chans=channels, embed_dim=outputs).float().eval()
        with torch.no_grad():
            net.proj.weight.copy_(torch.from_numpy(weight))
            net.proj.bias.copy_(torch.from_numpy(bias))
            x = torch.from_numpy(image)
            original = net(x)
            patches = torch.nn.functional.unfold(x, patch, stride=patch).transpose(1,2)
            projection = torch.nn.functional.conv2d(x, net.proj.weight, stride=patch).flatten(2).transpose(1,2)
        names = ["patches", "projection", "tokens"]
        for name, value in zip(names, [patches, projection, original]):
            key = prefix+"."+name
            tensors[key] = value.numpy().copy()
            # Limits declared in capture, before comparing native results.
            rules.append({"name":key,"mode":"exact"} if name == "patches" else
                         {"name":key,"mode":"float","max_abs":5e-5,
                          "relative_l2":1e-5,"zero_reference_floor":1e-12})
        filename = prefix+".input"
        with (args.output/filename).open("wb") as stream:
            stream.write(b"S3DPAT01"+struct.pack("<6I",batch,channels,height,width,outputs,patch))
            for value in [image,weight,bias]: stream.write(value.astype('<f4').tobytes())
        cases.append({"prefix":prefix,"input":filename,"shapes":{
            name:list(tensors[prefix+'.'+name].shape) for name in names}})
    save_file(tensors,args.output/"upstream.safetensors")
    boundary = "synthetic DINOv3 PatchEmbed F32 contract; final original module, auxiliary unfold/bias-free conv diagnostics; NOT trained-model parity"
    (args.output/"rules.json").write_text(json.dumps({"schema_version":1,"boundary":boundary,"tensors":rules},indent=2)+'\n')
    manifest = {"boundary":boundary,"revision":REVISION,"source_sha256":SOURCE_HASH,
                "torch":torch.__version__,"numpy":np.__version__,"device":"cpu",
                "dtype":"float32","threads":torch.get_num_threads(),
                "mkldnn":torch.backends.mkldnn.enabled,"cases":cases,"artifacts":{}}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["artifacts"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({"boundary":boundary,"cases":len(cases),"tensors":len(tensors)}))


if __name__ == "__main__": main()
