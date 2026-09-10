#!/usr/bin/env python3
"""Convert verified safe Body branch state to GGUF; no PyTorch or pickle."""
import argparse,json,os,struct,tempfile
from pathlib import Path
import numpy as np
from safetensors import safe_open
from convert_gguf import encode_header,sha256,unique_object
from convert_mhr_gguf import read_json
from body_branch_schema import ARCHITECTURE,SAFE_SHA,BODY_REVISION,CHECKPOINT_SHA256,CONFIG_SHA256,INTEGER_NAMES,shapes,metadata

def validate_value(name,value,shape,integer):
    if value.shape!=tuple(shape) or value.dtype!=np.dtype('int64' if integer else 'float32'):raise ValueError('branch tensor shape/dtype mismatch: '+name)
    if integer:
        low,high=(0,18439) if name=='head_pose.faces' else (68,122)
        if np.any(value<low) or np.any(value>=high):raise ValueError('branch index out of range: '+name)
        if name!='head_pose.faces' and len(np.unique(value))!=27:raise ValueError('duplicate hand indices')
        return np.ascontiguousarray(value,dtype='<i4')
    if not np.isfinite(value).all():raise ValueError('nonfinite branch tensor: '+name)
    return np.ascontiguousarray(value,dtype='<f4')

def write_archive(source,destination,spec,meta,expected,integers):
    """Bounded streaming implementation; the public CLI uses the fixed schema."""
    source,destination=Path(source),Path(destination)
    if source.suffix!='.safetensors' or not source.is_file():raise ValueError('safetensors required')
    if os.path.lexists(destination):raise FileExistsError(destination)
    if sha256(source)!=expected:raise ValueError('safe state identity mismatch')
    with source.open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8:raise ValueError('truncated safe header')
        size,=struct.unpack('<Q',raw)
        if not 2<=size<=4*1024*1024:raise ValueError('safe header too large')
        raw=f.read(size)
        if len(raw)!=size:raise ValueError('truncated safe header')
        json.loads(raw,object_pairs_hook=unique_object)
    header,body_size=encode_header(meta,spec,{k:26 if k in integers else 0 for k in spec})
    destination.parent.mkdir(parents=True,exist_ok=True);temporary=None
    try:
        with safe_open(source,framework='numpy') as safe:
            if not set(spec).issubset(safe.keys()):raise ValueError('missing branch tensor')
            for name,shape in spec.items():
                view=safe.get_slice(name)
                if tuple(view.get_shape())!=tuple(shape) or view.get_dtype()!=('I64' if name in integers else 'F32'):raise ValueError('branch schema mismatch: '+name)
            if INTEGER_NAMES.issubset(spec):
                left=safe.get_tensor('head_pose.hand_joint_idxs_left');right=safe.get_tensor('head_pose.hand_joint_idxs_right')
                if set(left.tolist())!=set(range(95,122)) or set(right.tolist())!=set(range(68,95)):raise ValueError('wrong handedness/index partition')
            with tempfile.NamedTemporaryFile(mode='wb',dir=destination.parent,prefix='.body-branch-',delete=False) as f:
                temporary=Path(f.name);f.write(header)
                for name,shape in sorted(spec.items()):
                    value=validate_value(name,safe.get_tensor(name),shape,name in integers)
                    f.write(value.tobytes());f.write(b'\0'*(-value.nbytes%32))
                if f.tell()!=len(header)+body_size:raise ValueError('GGUF output length mismatch')
                f.flush();os.fsync(f.fileno())
            # Verify every payload byte independently of the write loop.
            with temporary.open('rb') as f:
                if f.read(len(header))!=header:raise ValueError('GGUF header readback mismatch')
                for name,shape in sorted(spec.items()):
                    value=validate_value(name,safe.get_tensor(name),shape,name in integers)
                    if f.read(value.nbytes)!=value.tobytes() or f.read(-value.nbytes%32)!=b'\0'*(-value.nbytes%32):raise ValueError('GGUF tensor readback mismatch: '+name)
                if f.read(1):raise ValueError('GGUF trailing bytes')
        if sha256(source)!=expected:raise ValueError('source changed during conversion')
        os.link(temporary,destination)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return dict(architecture=ARCHITECTURE,tensors=len(spec),bytes=destination.stat().st_size,sha256=sha256(destination))

def convert(source,manifest,destination):
    data=read_json(Path(manifest));artifact=data.get('artifacts',{}).get('other_state',{})
    identities={r.get('path'):r.get('sha256') for r in data.get('source',[])}
    if data.get('schema_version')!=1 or data.get('body_revision')!=BODY_REVISION or artifact.get('sha256')!=SAFE_SHA or artifact.get('file')!=Path(source).name or identities.get('model.ckpt')!=CHECKPOINT_SHA256 or identities.get('model_config.yaml')!=CONFIG_SHA256:raise ValueError('unsupported extraction identity')
    return write_archive(source,destination,shapes(),metadata(),SAFE_SHA,INTEGER_NAMES)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['input','manifest','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(convert(a.input,a.manifest,a.output),indent=2))
if __name__=='__main__':main()
