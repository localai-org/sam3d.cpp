#!/usr/bin/env python3
"""Observe the actual complete released MHR forward; isolated container only."""
import argparse,json,struct
from pathlib import Path
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from safetensors.numpy import load_file,save_file
from capture_mhr import MODEL_SHA,MODEL_BYTES,digest
from trace_mhr_repeat import error

class Observe(TorchDispatchMode):
    def __init__(self):super().__init__();self.ops=[];self.values={};self.scratch=[];self.index=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        value=func(*args,**(kwargs or {}));index=self.index;self.index+=1
        def visit(v,path):
            if isinstance(v,torch.Tensor) and v.layout==torch.strided and v.dtype in [torch.float32,torch.float64] and 0<v.numel()<=1000000:
                name=f'op.{index:05d}.{path}';array=v.detach().cpu().numpy().copy();op={'name':name,'operation':str(func),'shape':list(v.shape),'dtype':str(v.dtype)}
                if not np.isfinite(array).all():
                    if str(func) not in ['aten.empty.memory_format','aten.fill_.Scalar','aten.select.int','aten.to.dtype_layout']:raise ValueError(f'nonfinite arithmetic {op}')
                    self.scratch.append(op);return
                self.ops.append(op);self.values[name]=array
            elif isinstance(v,(list,tuple)):
                for i,x in enumerate(v):visit(x,path+f'.{i}')
        visit(value,'output');return value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True);a=p.parse_args()
    if a.model.stat().st_size!=MODEL_BYTES or digest(a.model)!=MODEL_SHA:raise ValueError('wrong MHR identity')
    m=json.loads((a.reference/'manifest.json').read_text());path=a.reference/'inputs.safetensors'
    if m['model_sha256']!=MODEL_SHA or digest(path)!=m['artifacts']['inputs.safetensors']:raise ValueError('wrong reference inputs')
    data=load_file(path)
    if set(data)!={'identity','parameters','face'} or any(data[k].shape!=(2,n) or data[k].dtype!=np.float32 or not np.isfinite(data[k]).all() for k,n in [('identity',45),('parameters',204),('face',72)]):raise ValueError('invalid inputs')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=torch.jit.load(str(a.model),map_location=a.device).eval();args=[torch.from_numpy(data[k]).to(a.device) for k in ['identity','parameters','face']]
    a.output.mkdir(parents=True,exist_ok=True);reports=[]
    with torch.inference_mode():
        for enabled in [False,True]:
            model(*args,enabled);baseline=model(*args,enabled);baseline={k:v.cpu().numpy().copy() for k,v in zip(['vertices','skeleton'],baseline)}
            # A previously warmed CUDA graph retains cached fused plans even
            # inside optimized_execution(False). Observe a fresh load whose
            # first invocation is unoptimized; never substitute fused taps.
            observed_model=torch.jit.load(str(a.model),map_location=a.device).eval()
            observer=Observe()
            with torch.jit.optimized_execution(False),observer:observed=observed_model(*args,enabled)
            del observed_model
            diagnostic=a.output/f'operations_{int(enabled)}.json'
            diagnostic.write_text(json.dumps(observer.ops,indent=2)+'\n')
            equivalence={k:error(baseline[k],v.cpu().numpy()) for k,v in zip(['vertices','skeleton'],observed)}
            if any(e['max_abs']>1e-4 or e['relative_l2']>2e-5 for e in equivalence.values()):raise ValueError(f'observation changed model beyond fixed tolerance: {equivalence}')
            results={};mapping={}
            def select(op,shape):return [x for x in observer.ops if x['operation']==op and x['shape']==list(shape)]
            def keep(name,op,shape=None,transpose=False):
                v=observer.values[op['name']]
                if transpose:v=v.T
                if shape is not None:v=v.reshape(shape)
                results[name]=np.ascontiguousarray(v);mapping[name]=op|{'view_reshape':shape,'view_transpose':transpose}
            einsums=select('aten.einsum.default',(2,18439,3));adds=select('aten.add.Tensor',(2,18439,3))
            if len(einsums)!=2 or len(adds)!=(3 if enabled else 2):raise ValueError(f'unexpected blendshape operations: {einsums}, {adds}')
            keep('01.identity_projection',einsums[0]);keep('02.identity_rest',adds[0]);keep('03.expression',einsums[1]);keep('04.linear_unposed',adds[1])
            jp=select('aten.einsum.default',(2,889))
            if len(jp)!=1:raise ValueError('missing joint projection')
            keep('10.joint_parameters',jp[0]);results['11.skeleton']=observed[1].cpu().numpy().copy()
            if enabled:
                for name,op,shape in [('20.pose_cos','aten.cos.default',(2,125,3)),('21.pose_sin','aten.sin.default',(2,125,3)),
                                      ('22.pose_features','aten.flatten.using_ints',(2,750)),('24.pose_relu','aten.relu.default',(2,3000))]:
                    found=select(op,shape)
                    if len(found)!=1:raise ValueError(f'missing {name}: {found}')
                    keep(name,found[0])
                sparse=select('aten.matmul.default',(3000,2));dense=select('aten.linear.default',(2,55317))
                if len(sparse)!=1 or len(dense)!=1:raise ValueError('unexpected corrective projection operations')
                keep('23.pose_sparse_projection',sparse[0],transpose=True);keep('25.pose_correctives',dense[0],shape=(2,18439,3));keep('26.unposed',adds[2])
            else:results['26.unposed']=results['04.linear_unposed'].copy()
            state=select('aten.cat.default',(2,127,8))
            if len(state)!=2:raise ValueError('unexpected local/skin state concatenations')
            keep('30.skin_joint_state',state[-1])
            for name,op,shape in [('31.skin_selected_state','aten.index_select.default',(2,51337,8)),('32.skin_selected_points','aten.index_select.default',(2,51337,3))]:
                found=select(op,shape)
                if len(found)!=1:raise ValueError('unexpected skin gather')
                keep(name,found[0])
            for op,label in [('aten.norm.ScalarOpt_dim','norm'),('aten.div.Tensor','normalized')]:
                found=select(op,(2,51337,1 if label=='norm' else 4))
                if len(found)!=2:raise ValueError('unexpected point quaternion normalization')
                for i,x in enumerate(found):keep(f'33.skin_{label}.{i}',x)
            crosses=select('aten.cross.default',(2,51337,3))
            if len(crosses)!=2:raise ValueError('unexpected skin rotation')
            for i,x in enumerate(crosses):keep(f'34.skin_cross.{i}',x)
            translated=select('aten.add.Tensor',(2,51337,3));weighted=select('aten.mul.Tensor',(2,51337,3));verts=select('aten.index_add.default',(2,18439,3))
            if len(translated)!=3 or not weighted or len(verts)!=1:raise ValueError('unexpected final skin operations')
            keep('35.skin_transformed',translated[-1]);keep('36.skin_weighted',weighted[-1]);keep('90.vertices',verts[0])
            if not np.array_equal(results['90.vertices'],observed[0].cpu().numpy()):raise ValueError('final tap is not original returned mesh')
            root=a.output/f'correctives_{int(enabled)}';root.mkdir(exist_ok=True)
            save_file(results,root/'upstream.safetensors');save_file(baseline,root/'full.safetensors');save_file(observer.values,root/'trace.safetensors')
            with (root/'input.bin').open('wb') as f:f.write(b'S3DMHG01'+struct.pack('<II',2,int(enabled))+b''.join(data[k].tobytes() for k in ['identity','parameters','face']))
            report={'correctives':enabled,'equivalence':equivalence,'mapping':mapping,'operations':observer.ops,'excluded_scratch':observer.scratch,
                    'tensors':[{'name':k,'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in sorted(results.items())]}
            reports.append(report);print(json.dumps({'correctives':enabled,'boundaries':len(results),'equivalence':equivalence}),flush=True)
    manifest={'schema_version':1,'scope':'original complete MHR geometry on official demo inputs, not SAM neural image-to-body',
        'model_sha256':MODEL_SHA,'input_sha256':digest(path),'capture_script_sha256':digest(Path(__file__)),'device':a.device,'torch':torch.__version__,
        'reports':reports,'artifacts':{str(f.relative_to(a.output)):digest(f) for f in sorted(a.output.rglob('*')) if f.is_file() and f.name!='manifest.json'}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
