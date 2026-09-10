#!/usr/bin/env python3
"""Isolated exact official MHR TorchScript reference and safe state extraction.

Run only in the reviewed reference container. This accepts one pinned official
asset, not arbitrary TorchScript/pickle, and does not import a third-party engine.
"""
import argparse,ast,hashlib,json,time,zipfile
from pathlib import Path
import torch
import numpy as np
from safetensors.numpy import save_file

MODEL_SHA='352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc'
MODEL_BYTES=696110248
DEMO_SHA='fb248b7dd5bf640e63f6d168593e8fd97e9e4db5537b5c14236d951b1a60f95c'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--export-state',action='store_true');args=p.parse_args()
    if args.model.stat().st_size!=MODEL_BYTES or digest(args.model)!=MODEL_SHA:raise ValueError('not the pinned official MHR asset')
    # Loading this reviewed, hash-verified TorchScript is restricted to isolation.
    torch.set_num_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=torch.jit.load(str(args.model),map_location=args.device).eval()
    args.output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.model) as z:code_hashes={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.endswith('.py')}
    state={};inventory=[]
    for name,v in model.state_dict().items():
        if v.layout!=torch.strided:raise ValueError(f'unsupported state layout: {name}')
        a=v.detach().cpu().contiguous().numpy()
        if a.dtype.kind=='f' and not np.isfinite(a).all():raise ValueError('nonfinite state')
        inventory.append({'name':name,'shape':list(a.shape),'dtype':str(a.dtype),'bytes':a.nbytes,'sha256':hashlib.sha256(a.tobytes()).hexdigest()})
        if args.export_state:state[name]=a.copy()
    if args.export_state:
        save_file(state,args.output/'state.safetensors');del state
        graph=model.forward.graph.copy();torch._C._jit_pass_inline(graph)
        (args.output/'forward.inlined.txt').write_text(str(graph))
        extra={'joint_names':model.get_joint_names(),'parameter_names':model.get_parameter_names(),
            'prefix_sizes':list(model.character_torch.skeleton._pmi_buffer_sizes)}
        (args.output/'geometry.json').write_text(json.dumps(extra,indent=2)+'\n')
    # Execute the original official demo's input generator, unchanged AST.
    demo=args.upstream/'demo.py'
    if digest(demo)!=DEMO_SHA:raise ValueError('upstream MHR demo changed')
    source=demo.read_text();tree=ast.parse(source,filename=str(demo));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_input_data']
    if len(nodes)!=1:raise ValueError('missing original input generator')
    ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(demo),'exec'),ns)
    identity,parameters,face=ns['_prepare_input_data'](2)
    inputs={'identity':identity.numpy(),'parameters':parameters.numpy(),'face':face.numpy()};save_file(inputs,args.output/'inputs.safetensors')
    arguments=[x.to(args.device) for x in [identity,parameters,face]];result={};timings={}
    with torch.inference_mode():
        for enabled in [False,True]:
            if args.device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();vertices,skeleton=model(*arguments,enabled)
            if args.device=='cuda':torch.cuda.synchronize()
            timings[str(enabled)]=time.perf_counter()-start
            for name,v in [('vertices',vertices),('skeleton',skeleton)]:
                a=v.detach().cpu().numpy().copy()
                if not np.isfinite(a).all():raise ValueError('nonfinite geometry')
                result[f'correctives_{int(enabled)}.{name}']=a
            print(f'original MHR correctives={enabled}: {list(vertices.shape)}, skeleton={list(skeleton.shape)}, {timings[str(enabled)]:.3f}s',flush=True)
    save_file(result,args.output/'upstream.safetensors')
    manifest={'schema_version':1,'boundary':'actual official MHR TorchScript forward on original MHR demo parameters; not SAM image-to-body',
        'model_sha256':MODEL_SHA,'model_bytes':MODEL_BYTES,'embedded_code_sha256':code_hashes,'capture_script_sha256':digest(Path(__file__)),
        'demo_sha256':digest(demo),'device':args.device,'threads':1,'torch':torch.__version__,'numpy':np.__version__,'tf32':False,
        'cuda_device':torch.cuda.get_device_name() if args.device=='cuda' else None,'state_inventory':inventory,
        'diagnostic_seconds_not_performance_acceptance':timings,'artifacts':{}}
    for f in sorted(args.output.iterdir()):
        if f.is_file() and f.name!='manifest.json':manifest['artifacts'][f.name]=digest(f)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
