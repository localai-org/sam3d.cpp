#!/usr/bin/env python3
"""Convert verified F32 safetensors to a strict SAM3D GGUF v3 component.

No PyTorch, pickle, torch.hub, remote code or model download. This initial
component is the complete Body DINOv3 H+ backbone, NOT the whole Body estimator.
Legacy extraction/upcasting belongs to the separately isolated trusted reference.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile

import numpy as np
from safetensors import safe_open
from gguf_schema import (ARCHITECTURE, BODY_REVISION, DINO_REVISION, CHECKPOINT_SHA256,
                         CONFIG_SHA256, body_dino_shapes)

ALIGNMENT = 32


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key,value in pairs:
        if key in result: raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_manifest(path):
    if path.stat().st_size > 65536: raise ValueError('manifest too large')
    def reject(value): raise ValueError(f'non-finite JSON constant: {value}')
    data=json.loads(path.read_text(),object_pairs_hook=unique_object,parse_constant=reject)
    required={'schema_version','architecture','body_revision','dinov3_revision',
              'checkpoint_sha256','config_sha256','safetensors_sha256','source_precision'}
    if not isinstance(data,dict) or set(data)!=required: raise ValueError('manifest field set mismatch')
    if type(data['schema_version']) is not int or data['schema_version']!=1:
        raise ValueError('unsupported manifest version')
    if data['architecture']!=ARCHITECTURE: raise ValueError('unsupported architecture')
    if data['body_revision']!=BODY_REVISION or data['dinov3_revision']!=DINO_REVISION:
        raise ValueError('unsupported source revision')
    for key in ['checkpoint_sha256','config_sha256','safetensors_sha256']:
        if not isinstance(data[key],str) or not re.fullmatch('[0-9a-f]{64}',data[key]):
            raise ValueError(f'invalid hash: {key}')
    if data['checkpoint_sha256']!=CHECKPOINT_SHA256 or data['config_sha256']!=CONFIG_SHA256:
        raise ValueError('unsupported checkpoint/config identity')
    precision=data['source_precision']
    if not isinstance(precision,list) or not precision or any(
            not isinstance(x,str) or x not in ['F32','F16','BF16'] for x in precision):
        raise ValueError('unsupported source precision')
    if len(set(precision))!=len(precision): raise ValueError('duplicate source precision')
    return data


def metadata(manifest):
    # Strings or U32 only; the native component loader rejects other encodings.
    return {'general.architecture':ARCHITECTURE,'general.alignment':ALIGNMENT,
        'sam3d.schema_version':1,'sam3d.component':'body.dinov3_backbone',
        'sam3d.source.body_repository':'https://github.com/facebookresearch/sam-3d-body',
        'sam3d.source.body_revision':manifest['body_revision'],
        'sam3d.source.dinov3_repository':'https://github.com/facebookresearch/dinov3',
        'sam3d.source.dinov3_revision':manifest['dinov3_revision'],
        'sam3d.source.checkpoint_sha256':manifest['checkpoint_sha256'],
        'sam3d.source.config_sha256':manifest['config_sha256'],
        'sam3d.source.safetensors_sha256':manifest['safetensors_sha256'],
        'sam3d.source.precision':','.join(sorted(manifest['source_precision'])),
        'sam3d.converted.precision':'F32',
        'sam3d.tensor_layout':'torch_contiguous; ggml_dimensions_reversed; no_data_transpose',
        'sam3d.preprocessing':'body_rgb_crop_v1;512x512;ImageNet_mean_std',
        'sam3d.required_companions':'none_for_backbone_features;not_a_complete_body_estimator',
        'sam3d.norm_epsilon':'1e-5','sam3d.rope_mode':'axial_separate_eval_no_rescale',
        'sam3d.output':'normalized_patch_features;decoder_and_MHR_not_included'}


def string(value):
    data=value.encode('utf-8')
    if not data or len(data)>65536 or b'\0' in data: raise ValueError('invalid GGUF string')
    return struct.pack('<Q',len(data))+data


def encode_header(meta, shapes, tensor_types=None):
    """Small standard GGUF writer, independently checked using GGML's reader."""
    if not 0<len(shapes)<=4096 or not 0<len(meta)<=128: raise ValueError('invalid GGUF counts')
    if tensor_types is not None and (set(tensor_types)!=set(shapes) or any(type(t) is not int or t not in [0,26] for t in tensor_types.values())):
        raise ValueError('unsupported GGUF tensor types')
    output=bytearray(struct.pack('<4sIQQ',b'GGUF',3,len(shapes),len(meta)))
    for key,value in sorted(meta.items()):
        output.extend(string(key))
        if type(value) is int and 0<=value<2**32:
            output.extend(struct.pack('<II',4,value)) # GGUF_TYPE_UINT32
        elif isinstance(value,str):
            output.extend(struct.pack('<I',8)+string(value)) # GGUF_TYPE_STRING
        else: raise ValueError('unsupported GGUF metadata type')
    offset=0
    for name,shape in sorted(shapes.items()):
        if not re.fullmatch('[A-Za-z0-9_.]+',name) or len(name)>=64: raise ValueError('invalid tensor name')
        if not 1<=len(shape)<=4 or any(type(x) is not int or x<=0 for x in shape):
            raise ValueError('invalid tensor shape')
        elements=math.prod(shape)
        if elements>2**31: raise ValueError('tensor too large')
        output.extend(string(name)+struct.pack('<I',len(shape)))
        output.extend(struct.pack('<'+'Q'*len(shape),*reversed(shape)))
        output.extend(struct.pack('<IQ',0 if tensor_types is None else tensor_types[name],offset)) # F32 or I32, both 4 bytes
        size=elements*4
        offset+=(size+ALIGNMENT-1)//ALIGNMENT*ALIGNMENT
    output.extend(b'\0'*(-len(output)%ALIGNMENT))
    if len(output)>4*1024*1024: raise ValueError('GGUF metadata too large')
    return bytes(output),offset


def validate_tensor(name,value,shape):
    if value.dtype!=np.float32 or value.shape!=tuple(shape): raise ValueError(f'tensor dtype/shape mismatch: {name}')
    if not np.isfinite(value).all(): raise ValueError(f'non-finite tensor: {name}')
    if name.endswith('bias_mask'):
        d=len(value)//3
        # Published Body state disables all QKV biases. The original factory
        # instead masks only K. Preserve both explicit patterns verbatim.
        if len(value)%3 or not (np.all(value==0) or
                (np.all(value[:d]==1) and np.all(value[d:2*d]==0) and np.all(value[2*d:]==1))):
            raise ValueError('invalid DINO key bias mask')
    if name=='rope_embed.periods' and not np.all(value>0): raise ValueError('invalid RoPE periods')


def convert(source, destination, meta, shapes, expected_sha256):
    """Stream one tensor at a time; publish only after all checks. Never overwrite."""
    source=Path(source); destination=Path(destination)
    if source.suffix!='.safetensors' or not source.is_file(): raise ValueError('input must be a safetensors file')
    if destination.exists(): raise FileExistsError('output already exists')
    if sha256(source)!=expected_sha256: raise ValueError('safetensors hash mismatch')
    # serde/map readers can collapse duplicate JSON tensor names. Check the
    # bounded original header before asking safetensors for any tensor views.
    with source.open('rb') as stream:
        prefix=stream.read(8)
        if len(prefix)!=8: raise ValueError('truncated safetensors header')
        length=struct.unpack('<Q',prefix)[0]
        if not 2<=length<=4*1024*1024: raise ValueError('safetensors header size limit')
        raw=stream.read(length)
        if len(raw)!=length: raise ValueError('truncated safetensors header')
        def reject_constant(value): raise ValueError(f'non-finite header constant: {value}')
        parsed=json.loads(raw,object_pairs_hook=unique_object,parse_constant=reject_constant)
        if not isinstance(parsed,dict): raise ValueError('invalid safetensors header')
    header,body_size=encode_header(meta,shapes)
    # NamedTemporaryFile is owned by this conversion. Keep failed conversions
    # from leaving an apparently usable/truncated model at the destination.
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with safe_open(source,framework='numpy') as tensors:
            if set(tensors.keys())!=set(shapes): raise ValueError('missing or unexpected safetensors tensors')
            for name,shape in shapes.items():
                view=tensors.get_slice(name)
                if view.get_dtype()!='F32' or tuple(view.get_shape())!=tuple(shape):
                    raise ValueError(f'tensor dtype/shape mismatch: {name}')
            with tempfile.NamedTemporaryFile(mode='wb',dir=destination.parent,prefix='.sam3d-',delete=False) as stream:
                temporary=Path(stream.name); stream.write(header)
                for name,shape in sorted(shapes.items()):
                    tensor=tensors.get_tensor(name)
                    validate_tensor(name,tensor,shape)
                    stream.write(np.ascontiguousarray(tensor,dtype='<f4').tobytes())
                    stream.write(b'\0'*(-tensor.nbytes%ALIGNMENT))
                if stream.tell()!=len(header)+body_size: raise ValueError('GGUF length mismatch')
                stream.flush(); os.fsync(stream.fileno())
        if sha256(source)!=expected_sha256: raise ValueError('input changed during conversion')
        # Atomic non-overwriting publication on the same filesystem. An existing
        # destination, including a dangling symlink, is never followed/replaced.
        os.link(temporary,destination)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)
    return {'architecture':meta['general.architecture'],'tensor_count':len(shapes),
            'bytes':destination.stat().st_size,'sha256':sha256(destination),
            'component_only':True,'model_parity_established':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=read_manifest(args.manifest)
    report=convert(args.input,args.output,metadata(manifest),body_dino_shapes(),manifest['safetensors_sha256'])
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
