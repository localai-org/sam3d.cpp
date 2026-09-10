#!/usr/bin/env python3
"""Non-replacing ATen trace to localize exact official MHR repeatability noise."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from safetensors.numpy import load_file,save_file
from capture_mhr import MODEL_SHA,MODEL_BYTES,digest

class Trace(TorchDispatchMode):
    def __init__(self):super().__init__();self.tensors={};self.operations=[];self.number=0;self.scratch=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        value=func(*args,**(kwargs or {}));index=self.number;self.number+=1
        def visit(v,path):
            if isinstance(v,torch.Tensor) and v.layout==torch.strided and v.dtype in [torch.float32,torch.float64] and 0<v.numel()<=250000:
                name=f'op.{index:05d}.{path}';array=v.detach().cpu().numpy().copy()
                metadata={'name':name,'operation':str(func),'shape':list(v.shape),'dtype':str(v.dtype),
                    'tensor_inputs':[{'shape':list(x.shape),'dtype':str(x.dtype),'layout':str(x.layout)} for x in args if isinstance(x,torch.Tensor)]}
                if not np.isfinite(array).all():
                    # Deterministic-algorithms mode poisons torch.empty storage
                    # with NaNs. These allocation/view outputs are not initialized
                    # numerical boundaries; reject nonfinites in any arithmetic.
                    if str(func) not in ['aten.empty.memory_format','aten.fill_.Scalar','aten.select.int','aten.to.dtype_layout']:
                        raise ValueError(f'nonfinite arithmetic output: {metadata}')
                    self.scratch.append(metadata);return
                self.tensors[name]=array;self.operations.append(metadata)
            elif isinstance(v,(tuple,list)):
                for i,x in enumerate(v):visit(x,path+f'.{i}')
        visit(value,'output');return value

def error(a,b):
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite comparison')
    d=a.astype(np.float64)-b.astype(np.float64)
    return {'changed':int(np.count_nonzero(d)),'max_abs':float(np.max(np.abs(d))),'relative_l2':float(np.linalg.norm(d)/max(np.linalg.norm(a.astype(np.float64)),1e-12))}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True);a=p.parse_args()
    if a.model.stat().st_size!=MODEL_BYTES or digest(a.model)!=MODEL_SHA:raise ValueError('not the pinned asset')
    manifest=json.loads((a.reference/'manifest.json').read_text());input_path=a.reference/'inputs.safetensors'
    if manifest['model_sha256']!=MODEL_SHA or digest(input_path)!=manifest['artifacts']['inputs.safetensors']:raise ValueError('input identity mismatch')
    data=load_file(input_path)
    if set(data)!= {'identity','parameters','face'} or any(data[k].shape!=(2,n) or data[k].dtype!=np.float32 or not np.isfinite(data[k]).all() for k,n in [('identity',45),('parameters',204),('face',72)]):raise ValueError('invalid inputs')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=torch.jit.load(str(a.model),map_location=a.device).eval();args=[torch.from_numpy(data[k]).to(a.device) for k in ['identity','parameters','face']]
    a.output.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        # Let the original profiling executor settle; record actual baseline and
        # observed outputs, rather than assume observation is transparent.
        model(*args,True);baseline=model(*args,True);baseline=[v.cpu().numpy().copy() for v in baseline]
        traces=[];comparison=[]
        for i in range(3):
            trace=Trace()
            with trace:observed=model(*args,True)
            if len(trace.operations)<100:raise ValueError('operation observation is missing')
            diffs=[error(x,y.cpu().numpy()) for x,y in zip(baseline,observed)]
            comparison.append(diffs)
            # This is bounded equivalence, NOT an exact instrumentation claim.
            if any(d['max_abs']>1e-4 or d['relative_l2']>2e-5 for d in diffs):raise ValueError('observation is not within frozen geometry thresholds')
            save_file(trace.tensors,a.output/f'trace.{i}.safetensors');traces.append(trace)
        if not all(t.operations==traces[0].operations for t in traces):raise ValueError('different operation sequence')
        differences=[]
        for op in traces[0].operations:
            name=op['name'];metrics=[error(traces[0].tensors[name],t.tensors[name]) for t in traces[1:]]
            if any(m['changed'] for m in metrics):differences.append(op|{'repeat_metrics':metrics})
        report={'boundary':'original MHR TorchScript unchanged operations observed through dispatch; repeatability diagnosis, not native parity',
            'model_sha256':MODEL_SHA,'capture_script_sha256':digest(Path(__file__)),'device':a.device,'torch':torch.__version__,
            'operation_count':len(traces[0].operations),'observed_vs_unobserved':comparison,'first_varying_operation':differences[0] if differences else None,
            'all_varying_operations':differences,'operations':traces[0].operations,'excluded_uninitialized_scratch':traces[0].scratch}
        (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['all_varying_operations','operations','excluded_uninitialized_scratch']},indent=2,allow_nan=False))
if __name__=='__main__':main()
