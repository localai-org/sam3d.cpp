"""Isolate original BF16 Conv2d rounding; auxiliary controls are labelled."""
import argparse,json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['weights','image','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(6)
    with safe_open(a.weights,framework='pt') as f:
        w=f.get_tensor('patch_embed.proj.weight').cuda().bfloat16()
        b=f.get_tensor('patch_embed.proj.bias').cuda().bfloat16()
    with safe_open(a.image,framework='pt') as f:x=f.get_tensor('case.0000.prepare.normalized_rgb').cuda().bfloat16()
    with torch.no_grad():
        actual=torch.nn.functional.conv2d(x,w,b,stride=16)
        separate=torch.nn.functional.conv2d(x,w,None,stride=16)+b[None,:,None,None]
        fused_f32=(torch.nn.functional.conv2d(x.float(),w.float(),None,stride=16)+b.float()[None,:,None,None]).bfloat16()
    tensors={k:v.flatten(2).transpose(1,2).float().cpu().contiguous() for k,v in dict(actual=actual,separate_bias=separate,f32_fused_bias=fused_f32).items()}
    save_file(tensors,a.output)
    print(json.dumps({k:dict(exact=torch.equal(v,tensors['actual']),max_abs=float((v-tensors['actual']).abs().max())) for k,v in tensors.items()}))
if __name__=='__main__':main()
