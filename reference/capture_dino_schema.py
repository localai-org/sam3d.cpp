#!/usr/bin/env python3
"""Capture original H+ factory state shapes on meta, without loading weights.

This is an architecture check, not evidence about checkpoint contents/parity.
Run only in the reviewed, offline reference container.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import torch
from capture_dino_block import HASHES as BLOCK_HASHES

HASHES = BLOCK_HASHES | {
    "layers/patch_embed.py":"c592c7262779c8e1789ed502aee81bdd55eba4607065f8d04ae95eea683d0ef1",
    "layers/rms_norm.py":"8509ffc53f0bbe9e353a8b9e57ca2396795640b4ecc85140c27b17127c971869",
    "models/vision_transformer.py":"1af05405e19bc42188f80bfcb204b87ed1ec72af998e0bbd4658a85d7fae93f2",
    "hub/backbones.py":"6c9a18aca0b5fcb1fce439213a94a528e072376a6023718e03dbd9fcfaa10deb",
    "hub/utils.py":"24d0e4bd083510bb00677ebd0082dd486a1f9c8b8467f98b7fe6109aefc4712d",
}


def load_factory(upstream):
    for name,digest in HASHES.items():
        if hashlib.sha256((upstream/'dinov3'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'upstream source mismatch: {name}')
    # Bypass unrelated optional FP8 package imports, but use original classes,
    # factory and initialization unchanged. Never call a pretrained factory.
    for name in ['dinov3','dinov3.layers','dinov3.models','dinov3.hub']:
        namespace=types.ModuleType(name)
        namespace.__path__=[str(upstream/Path(*name.split('.')))]
        sys.modules[name]=namespace
    for module,names in {
        'layer_scale':['LayerScale'], 'ffn_layers':['Mlp','SwiGLUFFN'],
        'patch_embed':['PatchEmbed'], 'rms_norm':['RMSNorm'],
        'rope_position_encoding':['RopePositionEmbedding'], 'block':['SelfAttentionBlock'],
    }.items():
        original=importlib.import_module('dinov3.layers.'+module)
        for name in names: setattr(sys.modules['dinov3.layers'],name,getattr(original,name))
    return importlib.import_module('dinov3.hub.backbones').dinov3_vith16plus


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    factory=load_factory(args.upstream)
    with torch.device('meta'):
        model=factory(pretrained=False).eval()
    state=model.state_dict()
    if any(t.device.type!='meta' for t in state.values()):
        raise ValueError('schema capture unexpectedly allocated real tensors')
    report={'schema_version':1,'boundary':'original H+ factory meta state; no weights or inference',
        'revision':'6876159a11b4df116f30f667f8c9888617df0751',
        'source_hashes':HASHES,'torch':torch.__version__,
        'parameter_count':sum(t.numel() for t in model.parameters()),
        'state_elements':sum(t.numel() for t in state.values()),
        'shapes':{k:list(v.shape) for k,v in sorted(state.items())},
        'dtypes':{k:str(v.dtype) for k,v in sorted(state.items())}}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'schema.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Original factory: {len(state)} tensors, {report["parameter_count"]} parameters; meta only')


if __name__=='__main__': main()
