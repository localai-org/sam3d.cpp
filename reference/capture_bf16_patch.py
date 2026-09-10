"""Tiny actual CUDA Conv2d BF16 fixture, using existing synthetic backbone state."""
import argparse,json
from pathlib import Path
import torch
from capture_dino_backbone import digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    words=iter(a.source.read_text().split())
    if next(words)!='S3D_BACKBONE_REGRESSION_V1':raise ValueError('bad source fixture')
    b,h,w,patch,d,heads,hidden,depth,storage=[int(next(words)) for _ in range(9)]
    def read(name):
        if next(words)!=name:raise ValueError('bad source tensor order')
        n=int(next(words));return torch.tensor([float(next(words)) for _ in range(n)],device='cuda').bfloat16()
    x=read('image').reshape(b,3,h,w);weight=read('patch_embed.proj.weight').reshape(d,3,patch,patch);bias=read('patch_embed.proj.bias')
    torch.backends.cudnn.allow_tf32=False
    with torch.no_grad():y=torch.nn.functional.conv2d(x,weight,bias,stride=patch).flatten(2).transpose(1,2)
    lines=['S3D_BF16_PATCH_V1',f'{b} 3 {h} {w} {d} {patch}']
    for name,value in [('image',x),('weight',weight),('bias',bias),('output',y)]:
        values=value.float().cpu().flatten().tolist();lines.append(f'{name} {len(values)}')
        for start in range(0,len(values),8):lines.append(' '.join(format(v,'.9g') for v in values[start:start+8]))
    with a.output.open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(scope=__doc__,device=torch.cuda.get_device_name(),torch=torch.__version__,dtype='bfloat16',
        source_sha256=digest(a.source),fixture_sha256=digest(a.output),script_sha256=digest(Path(__file__)))))
if __name__=='__main__':main()
