"""Tiny BF16 original-module/math-SDPA fixture; not trained-model parity."""
import argparse,importlib,importlib.util,json,struct,types
from functools import partial
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from safetensors import safe_open
from capture_dino_schema import load_factory
from capture_dino_block import PARAMETERS
from capture_mhr import digest
from observe_trained_dino import TrainedDinoObserver

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['upstream','body-upstream','source','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    load_factory(a.upstream)
    utility=a.body_upstream/'sam_3d_body/models/optim/fp16_utils.py'
    if digest(utility)!='90e25559a99fa341666c9c5da5b7e91d679ffdfab635aed9e9cf21d8f43c107d':raise ValueError('unverified precision helper')
    spec=importlib.util.spec_from_file_location('original_precision',utility);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    Block=importlib.import_module('dinov3.layers.block').SelfAttentionBlock
    SwiGLU=importlib.import_module('dinov3.layers.ffn_layers').SwiGLUFFN
    Rope=importlib.import_module('dinov3.layers.rope_position_encoding').RopePositionEmbedding
    words=iter(a.source.read_text().split())
    if next(words)!='S3D_DINO_REGRESSION_V1':raise ValueError('invalid source fixture')
    b,h,w,d,heads,prefix,hidden=[int(next(words)) for _ in range(7)]
    def read(name):
        if next(words)!=name:raise ValueError('source order mismatch')
        n=int(next(words));return torch.tensor([float(next(words)) for _ in range(n)],dtype=torch.float32)
    x=read('input').reshape(b,h*w+prefix,d)
    weights={name:read(name) for name in PARAMETERS}
    net=Block(d,heads,ffn_ratio=6,qkv_bias=True,proj_bias=True,ffn_bias=True,init_values=1e-5,
        norm_layer=partial(torch.nn.LayerNorm,eps=1e-5),ffn_layer=SwiGLU,mask_k_bias=True).eval().to(a.device)
    rope=Rope(d,num_heads=heads,base=100,normalize_coords='separate',dtype=torch.float32,device=a.device).eval()
    with torch.no_grad():
        for name,target in net.state_dict().items():target.copy_(weights[name].reshape(target.shape).to(a.device))
        rope.periods.copy_(weights['periods'].to(a.device))
    net.apply(partial(module.convert_to_fp16_safe,dtype=torch.bfloat16));rope.to(torch.bfloat16)
    x=x.to(a.device,torch.bfloat16)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.output.mkdir(parents=True,exist_ok=True)
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
        angles=rope(H=h,W=w);baseline=net(x,angles).clone()
        observer=TrainedDinoObserver(types.SimpleNamespace(blocks=[net]),a.output/'operations')
        observed=net(x,angles);observer.close()
        if not torch.equal(baseline,observed):raise ValueError('BF16 observer changes original result')
    state=dict(net.state_dict());state['periods']=rope.periods
    with (a.output/'input.bin').open('xb') as f:
        f.write(b'S3DBLK01'+struct.pack('<7I',b,h,w,d,heads,prefix,hidden))
        f.write(x.float().cpu().numpy().astype('<f4').tobytes())
        for name in PARAMETERS:f.write(state[name].float().cpu().numpy().astype('<f4').tobytes())
    lines=['S3D_DINO_REGRESSION_V1',f'{b} {h} {w} {d} {heads} {prefix} {hidden}']
    def append(name,v):
        v=v.reshape(-1);lines.append(f'{name} {len(v)}')
        for start in range(0,len(v),8):lines.append(' '.join(format(float(t),'.9g') for t in v[start:start+8]))
    append('input',x.float().cpu().numpy())
    for name in PARAMETERS:append(name,state[name].float().cpu().numpy())
    with safe_open(a.output/'operations/block.0.safetensors',framework='np') as safe:
        for name in sorted(safe.keys()):append(name,safe.get_tensor(name))
    (a.output/'regression.txt').write_text('\n'.join(lines)+'\n')
    (a.output/'manifest.json').write_text(json.dumps(dict(scope=__doc__,precision='bfloat16',device=a.device,torch=torch.__version__,
        original_observed_equal=True,input_sha256=digest(a.output/'input.bin'),source_sha256=digest(a.source),
        regression_sha256=digest(a.output/'regression.txt'),script_sha256=digest(Path(__file__))),indent=2)+'\n')

if __name__=='__main__':main()
