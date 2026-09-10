#!/usr/bin/env python3
"""Original isolated BF16 SDPA controls on all 32 recorded trained Q/K/V inputs.

Not an end-to-end inference result: these inputs deliberately isolate attention.
The same original inputs go to CPU math and CUDA math/flash/efficient kernels.
No native output is read when fixing this operation's numerical limits.
"""
import argparse, hashlib, json, struct
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from safetensors import safe_open
from safetensors.numpy import save_file

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads((a.operations/'manifest.json').read_text())
    blocks=manifest['blocks']
    if [b['index'] for b in blocks]!=list(range(32)):raise ValueError('requires all 32 original blocks')
    if any(a.output.iterdir()):raise ValueError('output must be empty')
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    modes=[('cuda_math','cuda',SDPBackend.MATH),('cuda_flash','cuda',SDPBackend.FLASH_ATTENTION),
           ('cuda_efficient','cuda',SDPBackend.EFFICIENT_ATTENTION),('cpu_math','cpu',SDPBackend.MATH)]
    records=[];rules=[];results={}
    with torch.inference_mode():
        for block in blocks:
            path=a.operations/block['file']
            if path.name!=block['file'] or digest(path)!=block['sha256']:raise ValueError('original operation identity mismatch')
            with safe_open(path,framework='pt') as f:
                original=[f.get_tensor(k).contiguous() for k in ['07.q_rope','08.k_rope','06.v']]
                expected=f.get_tensor('11.attention').contiguous()
            if any(tuple(x.shape)!=(1,20,1029,64) or not torch.isfinite(x).all() or not torch.equal(x,x.bfloat16().float()) for x in original):
                raise ValueError('invalid original trained BF16 attention inputs')
            name=f'block.{block["index"]:02d}';input_path=a.output/(name+'.input')
            with input_path.open('xb') as stream:
                stream.write(b'S3DATT01'+struct.pack('<4I',*original[0].shape))
                for value in original:stream.write(value.numpy().astype('<f4').tobytes())
            outputs={}
            for key,device,backend in modes:
                q,k,v=[x.to(device=device,dtype=torch.bfloat16) for x in original]
                with sdpa_kernel([backend]):
                    def run():return torch.nn.functional.scaled_dot_product_attention(q,k,v).transpose(1,2).flatten(2).float().cpu().contiguous()
                    result=run()
                    if not torch.equal(result,run()):raise ValueError('nonrepeatable original '+key)
                outputs[key]=result.numpy()
                del q,k,v
            if not np.array_equal(outputs['cuda_math'],expected.numpy()):raise ValueError('isolated math result differs from actual original attention')
            base=outputs['cuda_math'];controls=[]
            for key in outputs.keys()-{'cuda_math'}:
                delta=outputs[key].astype(np.float64)-base
                controls.append(dict(mode=key,max_abs=float(abs(delta).max()),relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(base.astype(np.float64)),1e-12))))
            absolute=max(1e-4,2*max(c['max_abs'] for c in controls));relative=max(2e-5,2*max(c['relative_l2'] for c in controls))
            if relative>.05 or absolute>max(1e-4,.1*float(abs(base).max())):raise ValueError('unstable original attention controls')
            rules.append(dict(name=name,mode='float',max_abs=absolute,relative_l2=relative,zero_reference_floor=1e-12))
            for key,out in outputs.items():results[name+'.'+key]=out
            records.append(dict(name=name,input_sha256=digest(input_path),source_sha256=block['sha256'],shape=list(base.shape),controls=sorted(controls,key=lambda c:c['mode'])))
            print(name,controls,flush=True)
    save_file(results,a.output/'upstream.safetensors')
    report=dict(schema_version=1,boundary='isolated trained BF16 attention on original Q/K/V: not full own-intermediate inference',
        native_candidate_used=False,policy='two times independent original SDPA kernel variation, with 5% relative and 10%-of-peak absolute caps',
        torch=torch.__version__,cuda=torch.version.cuda,tf32=False,records=records,tensors=rules,
        source_manifest_sha256=digest(a.operations/'manifest.json'),script_sha256=digest(Path(__file__)),
        upstream_sha256=digest(a.output/'upstream.safetensors'))
    with (a.output/'policy.json').open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
if __name__=='__main__':main()
