#!/usr/bin/env python3
"""Small original LBS regression from selected actual MHR vertices, in isolation."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import load_file
from capture_mhr import MODEL_SHA,MODEL_BYTES,digest
from capture_mhr_geometry import Observe
from trace_mhr_repeat import error

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model','reference','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.model.stat().st_size!=MODEL_BYTES or digest(a.model)!=MODEL_SHA:raise ValueError('wrong model')
    manifest=json.loads((a.reference/'manifest.json').read_text());path=a.reference/'correctives_1/upstream.safetensors'
    if manifest['model_sha256']!=MODEL_SHA or digest(path)!=manifest['artifacts']['correctives_1/upstream.safetensors'] or manifest['device']!='cpu':raise ValueError('wrong geometry reference')
    r=load_file(path);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);model=torch.jit.load(str(a.model),map_location='cpu').eval();lbs=model.character_torch.linear_blend_skinning
    selected=np.linspace(0,18438,16,dtype=np.int32);mapping={int(v):i for i,v in enumerate(selected)};old_vertices=lbs.vert_indices_flattened.numpy();mask=np.isin(old_vertices,selected)
    data={'skeleton':r['11.skeleton'],'unposed':np.ascontiguousarray(r['26.unposed'][:,selected,:]),'inverse_bind':lbs.inverse_bind_pose.numpy().copy(),
          'skin_joints':lbs.skin_indices_flattened.numpy()[mask].astype(np.int32),'weights':lbs.skin_weights_flattened.numpy()[mask].copy(),
          'skin_vertices':np.array([mapping[int(v)] for v in old_vertices[mask]],dtype=np.int32)}
    # Only subset data/buffer dimensions change; execute the original released
    # skinning method, not replacement Python math. Validate against the same
    # vertices in the original complete model before using this as a fixture.
    lbs.num_vertices=16;lbs.skin_indices_flattened=torch.from_numpy(data['skin_joints']);lbs.skin_weights_flattened=torch.from_numpy(data['weights']);lbs.vert_indices_flattened=torch.from_numpy(data['skin_vertices'].astype(np.int64))
    observer=Observe()
    with torch.inference_mode(),torch.jit.optimized_execution(False),observer:output=lbs(torch.from_numpy(data['skeleton']),torch.from_numpy(data['unposed']))
    equivalence=error(r['90.vertices'][:,selected,:],output.numpy())
    if equivalence['max_abs']>1e-4 or equivalence['relative_l2']>2e-5:raise ValueError('subset changed original vertex mapping')
    n=len(data['weights']);result={}
    def pick(operation,shape):return [op for op in observer.ops if op['operation']==operation and op['shape']==list(shape)]
    def keep(name,op):result[name]=observer.values[op['name']]
    for name,op,shape in [('30.skin_joint_state','aten.cat.default',(2,127,8)),('31.skin_selected_state','aten.index_select.default',(2,n,8)),('32.skin_selected_points','aten.index_select.default',(2,n,3)),('90.vertices','aten.index_add.default',(2,16,3))]:
        found=pick(op,shape)
        if len(found)!=1:raise ValueError('missing original skin tap')
        keep(name,found[0])
    for op,label,width in [('aten.norm.ScalarOpt_dim','33.skin_norm',1),('aten.div.Tensor','33.skin_normalized',4),('aten.cross.default','34.skin_cross',3)]:
        found=pick(op,(2,n,width))
        if len(found)!=2:raise ValueError('missing normalization/rotation')
        for i,v in enumerate(found):keep(label+f'.{i}',v)
    adds=pick('aten.add.Tensor',(2,n,3));muls=pick('aten.mul.Tensor',(2,n,3))
    if len(adds)!=3 or not muls:raise ValueError('missing final transform')
    keep('35.skin_transformed',adds[-1]);keep('36.skin_weighted',muls[-1])
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'mhr-skin.txt').open('w') as f:
        f.write(f'S3D_MHR_SKIN_V1 2 16 {n}\n')
        for name in ['skeleton','unposed','inverse_bind','skin_joints','weights','skin_vertices']:
            v=data[name];f.write(f'{name} {v.size}\n');f.write(' '.join(format(x,'.9g') if v.dtype.kind=='f' else str(int(x)) for x in v.flat)+'\n')
        f.write(str(len(result))+'\n')
        for name,v in sorted(result.items()):f.write(f'{name} {v.size}\n'+' '.join(format(x,'.9g') for x in v.flat)+'\n')
    report={'scope':'isolated original skinning of 16 selected actual vertices; not whole geometry projection coverage',
        'model_sha256':MODEL_SHA,'full_reference_sha256':digest(path),'script_sha256':digest(Path(__file__)),'observer_sha256':digest(Path(__file__).with_name('capture_mhr_geometry.py')),
        'selected_original_vertices':selected.tolist(),'subset_vs_original_full_vertices':equivalence,'fixture_sha256':digest(a.output/'mhr-skin.txt'),'operations':observer.ops}
    (a.output/'mhr-skin.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'boundaries':len(result),'influences':n,'equivalence':equivalence}))
if __name__=='__main__':main()
