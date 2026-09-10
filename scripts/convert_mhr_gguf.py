#!/usr/bin/env python3
"""Convert isolated safe MHR extraction to F32/I32 GGUF; never loads TorchScript."""
import argparse,json,os,re,struct,tempfile
from pathlib import Path
import numpy as np
from safetensors import safe_open
from convert_gguf import encode_header,sha256,unique_object
from mhr_schema import ARCHITECTURE,ASSET_SHA,RELEASE_SHA,RELEASE_URL,TENSORS

def read_json(path):
    if path.stat().st_size>4*1024*1024:raise ValueError('JSON too large')
    def reject(x):raise ValueError(f'invalid JSON number: {x}')
    return json.loads(path.read_text(),object_pairs_hook=unique_object,parse_constant=reject)

def names(values,count):
    if not isinstance(values,list) or len(values)!=count or any(not isinstance(n,str) or not re.fullmatch('[A-Za-z0-9_.:-]{1,128}',n) for n in values) or len(set(values))!=count:
        raise ValueError('invalid/duplicate geometry names')
    return ','.join(values)

def validate_indices(name,a):
    def bounded(v,low,high):
        if np.any(v<low) or np.any(v>=high):raise ValueError(f'index out of bounds: {name}')
    if name=='pose.sparse.indices':
        bounded(a[0],0,3000);bounded(a[1],0,750)
        order=a[0].astype(np.int64)*750+a[1]
        if np.any(np.diff(order)<=0):raise ValueError('sparse COO indices must be unique and sorted')
    elif name=='skeleton.parents':
        bounded(a,-1,127)
        if a[0]!=-1 or np.any(a[1:]<0) or np.any(a[1:]>=np.arange(1,127)):raise ValueError('invalid parent topology')
    elif name in ['skeleton.prefix','skin.joints']:bounded(a,0,127)
    elif name in ['skin.vertices','mesh.faces']:bounded(a,0,18439)
    elif name=='mesh.texcoord_faces':bounded(a,0,19455)

def convert(source,manifest,geometry,destination):
    source,destination=Path(source),Path(destination)
    if source.suffix!='.safetensors' or not source.is_file():raise ValueError('safe tensor input required')
    if destination.exists():raise FileExistsError('output exists')
    m=read_json(Path(manifest));g=read_json(Path(geometry))
    if m.get('model_sha256')!=ASSET_SHA or m.get('model_bytes')!=696110248:raise ValueError('MHR asset identity mismatch')
    expected=m.get('artifacts',{}).get(source.name)
    if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected) or sha256(source)!=expected:raise ValueError('safe state hash mismatch')
    if m.get('artifacts',{}).get(Path(geometry).name)!=sha256(Path(geometry)):raise ValueError('geometry metadata hash mismatch')
    if set(g)!= {'joint_names','parameter_names','prefix_sizes'} or g['prefix_sizes']!=[65,56,62,83]:raise ValueError('geometry schema mismatch')
    meta={'general.architecture':ARCHITECTURE,'general.alignment':32,'sam3d.schema_version':1,'sam3d.component':'mhr.lod1',
        'sam3d.source.asset_sha256':ASSET_SHA,'sam3d.source.safetensors_sha256':expected,'sam3d.source.release':RELEASE_URL,'sam3d.source.release_sha256':RELEASE_SHA,
        'sam3d.converted.precision':'F32,I32','sam3d.tensor_layout':'torch_contiguous; ggml_dimensions_reversed; no_data_transpose',
        'sam3d.kinematics':'local_F32;prefix_F64;global_F32;XYZ;XYZW;log2_scale','sam3d.units':'centimeters','sam3d.prefix_sizes':'65,56,62,83',
        'sam3d.joint_names':names(g['joint_names'],127),'sam3d.parameter_names':names(g['parameter_names'],249),
        'sam3d.output':'vertices_cm;skeleton_tx_ty_tz_qx_qy_qz_qw_scale;not_image_estimator'}
    with source.open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8:raise ValueError('truncated safe header')
        n=struct.unpack('<Q',raw)[0]
        if not 2<=n<=4*1024*1024:raise ValueError('invalid safe header size')
        raw=f.read(n)
        if len(raw)!=n:raise ValueError('truncated safe header')
        json.loads(raw,object_pairs_hook=unique_object)
    header,body=encode_header(meta,{k:v[3] for k,v in TENSORS.items()},{k:v[2] for k,v in TENSORS.items()})
    destination.parent.mkdir(parents=True,exist_ok=True);temporary=None
    try:
        with safe_open(source,framework='numpy') as tensors:
            inventory=m.get('state_inventory',[])
            if len(inventory)!=len({v['name'] for v in inventory}) or {v['name'] for v in inventory}!=set(tensors.keys()):raise ValueError('state inventory mismatch')
            for name,(key,dtype,_,shape) in TENSORS.items():
                view=tensors.get_slice(key)
                if view.get_dtype()!=dtype or tuple(view.get_shape())!=shape:raise ValueError(f'dtype/shape mismatch: {name}')
            with tempfile.NamedTemporaryFile(mode='wb',dir=destination.parent,prefix='.mhr-',delete=False) as f:
                temporary=Path(f.name);f.write(header)
                for name,(key,_,dtype,_) in sorted(TENSORS.items()):
                    a=tensors.get_tensor(key)
                    if dtype==26:
                        if np.any(a<-(2**31)) or np.any(a>=2**31):raise ValueError('lossy integer conversion')
                        validate_indices(name,a);a=np.ascontiguousarray(a,dtype='<i4')
                    else:
                        if not np.isfinite(a).all():raise ValueError(f'nonfinite tensor: {name}')
                        if name=='skin.weights' and (np.any(a<0) or np.any(a>1)):raise ValueError('invalid skin weight')
                        a=np.ascontiguousarray(a,dtype='<f4')
                    f.write(a.tobytes());f.write(b'\0'*(-a.nbytes%32))
                if f.tell()!=len(header)+body:raise ValueError('output size mismatch')
                f.flush();os.fsync(f.fileno())
        if sha256(source)!=expected:raise ValueError('source changed during conversion')
        os.link(temporary,destination)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return {'architecture':ARCHITECTURE,'tensors':len(TENSORS),'bytes':destination.stat().st_size,'sha256':sha256(destination),'scope':'MHR geometry asset only; native inference parity not yet established'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['input','manifest','geometry','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(convert(a.input,a.manifest,a.geometry,a.output),indent=2))
if __name__=='__main__':main()
