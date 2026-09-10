#!/usr/bin/env python3
"""Observe the released MHR skeleton methods; run only in the isolated container.

No Python replacement skeleton oracle. Disable the profiling executor while
observing individual operations, and compare against original whole-model output.
"""
import argparse,json,struct
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import load_file,save_file
from capture_mhr import MODEL_SHA,MODEL_BYTES,digest
from trace_mhr_repeat import Trace,error

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model','reference','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True);a=p.parse_args()
    if a.model.stat().st_size!=MODEL_BYTES or digest(a.model)!=MODEL_SHA:raise ValueError('not pinned MHR asset')
    manifest=json.loads((a.reference/'manifest.json').read_text());path=a.reference/'inputs.safetensors'
    if manifest['model_sha256']!=MODEL_SHA or digest(path)!=manifest['artifacts']['inputs.safetensors']:raise ValueError('wrong input identity')
    data=load_file(path)
    if set(data)!={'identity','parameters','face'} or any(data[k].shape!=(2,n) or data[k].dtype!=np.float32 or not np.isfinite(data[k]).all() for k,n in [('identity',45),('parameters',204),('face',72)]):raise ValueError('invalid inputs')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=torch.jit.load(str(a.model),map_location=a.device).eval();c=model.character_torch
    args=[torch.from_numpy(data[k]).to(a.device) for k in ['identity','parameters','face']]
    a.output.mkdir(parents=True,exist_ok=True);results={};mapping={};traces={}
    def keep(name,v):
        arr=v.detach().cpu().numpy().copy() if isinstance(v,torch.Tensor) else v.copy()
        if not np.isfinite(arr).all():raise ValueError('nonfinite boundary')
        results[name]=arr
    def select(trace,operation,shape):
        return [op for op in trace.operations if op['operation']==operation and op['shape']==list(shape)]
    def selected(name,trace,op):keep(name,trace.tensors[op['name']]);mapping[name]=op
    with torch.inference_mode():
        model(*args,False);baseline=model(*args,False)[1].cpu().numpy().copy()
        # This changes execution optimization, not the mathematical source. Its
        # output equivalence is measured below, not assumed.
        with torch.jit.optimized_execution(False):
            padded=torch.cat([args[1],torch.zeros_like(args[0])],1);keep('00.padded',padded)
            jp=c.model_parameters_to_joint_parameters(padded);keep('01.joint_parameters',jp)
            local_trace=Trace()
            with local_trace:local=c.joint_parameters_to_local_skeleton_state(jp)
            keep('10.local',local);traces['local']=local_trace
            trig=[op for op in local_trace.operations if op['operation'] in ['aten.cos.default','aten.sin.default']]
            if len(trig)!=6:raise ValueError('missing original Euler trig operations')
            for name,op in zip(['cy','sy','cp','sp','cr','sr'],trig):selected('02.'+name,local_trace,op)
            quats=select(local_trace,'aten.stack.default',(2,127,4))
            if len(quats)!=2:raise ValueError('missing original quaternion operations')
            selected('03.euler_quaternion',local_trace,quats[0]);selected('04.local_quaternion',local_trace,quats[1])
            translations=select(local_trace,'aten.add.Tensor',(2,127,3));scales=select(local_trace,'aten.exp.default',(2,127,1))
            if len(translations)!=1 or len(scales)!=1:raise ValueError('missing local translation or scale operation')
            selected('05.local_translation',local_trace,translations[0]);selected('06.local_scale',local_trace,scales[0])
            fk_trace=Trace()
            with fk_trace:global_state=c.local_skeleton_state_to_skeleton_state(local)
            traces['fk']=fk_trace;keep('90.skeleton',global_state)
            copies=select(fk_trace,'aten.index_copy_.default',(2,127,8))
            if len(copies)!=4:raise ValueError('wrong prefix operation count')
            for i,op in enumerate(copies):selected(f'20.prefix.{i}',fk_trace,op)
            # Capture normalization, rotation and multiply boundaries within
            # every original prefix group, preserving F64 (no cast to F32).
            previous=-1
            for i,copy in enumerate(copies):
                subset=[op for op in fk_trace.operations if previous<int(op['name'].split('.')[1])<int(copy['name'].split('.')[1])]
                for operation,label,expected in [('aten.norm.ScalarOpt_dim','norm',2),('aten.div.Tensor','normalized',2),('aten.cross.default','cross',2),('aten.cat.default','product',1)]:
                    found=[op for op in subset if op['operation']==operation]
                    if len(found)!=expected:raise ValueError(f'wrong prefix {label} count')
                    for j,op in enumerate(found):selected(f'19.prefix.{i}.{label}.{j}',fk_trace,op)
                previous=int(copy['name'].split('.')[1])
        equivalence=error(baseline,results['90.skeleton'])
        if equivalence['max_abs']>1e-4 or equivalence['relative_l2']>2e-5:raise ValueError(f'optimized/observed skeleton mismatch {equivalence}')
    save_file(results,a.output/'upstream.safetensors')
    save_file({'skeleton':baseline},a.output/'full-skeleton.safetensors')
    with (a.output/'input.bin').open('wb') as f:f.write(b'S3DMHS01'+struct.pack('<I',2)+data['parameters'].tobytes())
    # Small isolated local/FK regression: original geometry constants and joint
    # inputs, not the large parameter/learned matrices. Whole GGUF test consumes
    # only model parameters and must compute its own joint inputs.
    small={'offsets':c.skeleton.joint_translation_offsets.cpu().numpy(),'prerotations':c.skeleton.joint_prerotations.cpu().numpy(),
           'prefix':c.skeleton.pmi.cpu().numpy().astype(np.int32),'parents':c.skeleton.joint_parents.cpu().numpy().astype(np.int32),
           'joint_parameters':results['01.joint_parameters']}
    save_file(small,a.output/'local-inputs.safetensors')
    for label,trace in traces.items():save_file(trace.tensors,a.output/f'{label}.trace.safetensors')
    report={'schema_version':1,'boundary':'released MHR parameter transform + local skeleton + F64 prefix FK; native mesh not covered',
        'model_sha256':MODEL_SHA,'input_sha256':digest(path),'capture_script_sha256':digest(Path(__file__)),
        'trace_script_sha256':digest(Path(__file__).with_name('trace_mhr_repeat.py')),'device':a.device,'torch':torch.__version__,
        'optimized_full_vs_observed_skeleton':equivalence,'operation_mapping':mapping,
        'tensors':[{'name':k,'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in sorted(results.items())],
        'artifacts':{f.name:digest(f) for f in sorted(a.output.iterdir()) if f.is_file() and f.name!='manifest.json'}}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'boundaries':len(results),'equivalence':equivalence},indent=2))
if __name__=='__main__':main()
