#!/usr/bin/env python3
"""Run GGUF-only Body inference from pixels; parity inputs contain no weights."""
import argparse,json,os,struct,subprocess
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from safetensors.numpy import save_file
from check_parity import sha256_file

def extract_image(source,target):
    # The original capture includes diagnostic parameters after the image.
    # Only pixels, dimensions, box and intrinsics are copied into runtime input.
    with source.open('rb') as f:
        if f.read(8) not in [b'S3DRGB02',b'S3DRGB03']:raise ValueError('requires trained original RGB input')
        f.seek(24*4+4+54*4,1)
        dims=f.read(12)
        if len(dims)!=12:raise ValueError('truncated RGB input')
        width,height,stride=struct.unpack('<3I',dims)
        if not 1<=width<=32766 or not 1<=height<=32766 or width*height>16000000 or stride!=width*3:raise ValueError('invalid RGB extent')
        data=f.read(32+stride*height)
        if len(data)!=32+stride*height:raise ValueError('truncated RGB payload')
    with target.open('xb') as f:f.write(b'S3DIMG01'+dims+data)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['runner','module','backbone','branch','mhr','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--backend',choices=['CPU','Vulkan'],required=True);p.add_argument('--threads',type=int,default=6)
    p.add_argument('--device',type=int,default=0);p.add_argument('--description',default='-')
    p.add_argument('--experimental-cm2',action='store_true',help='test F32 regression with the reviewed BF16/CM2 backend patch actually enabled')
    a=p.parse_args()
    if a.experimental_cm2 and a.backend!='Vulkan':p.error('--experimental-cm2 requires Vulkan')
    m=json.loads((a.reference/'manifest.json').read_text())
    if not m.get('learned_sam_checkpoint_loaded') or len(m['cases'])!=1 or m['device']!=('cpu' if a.backend=='CPU' else 'cuda'):raise ValueError('requires matched original trained reference')
    case=m['cases'][0]
    if Path(case['input']).name!=case['input'] or sha256_file(a.reference/case['input'])!=m['artifacts'][case['input']]:raise ValueError('original input identity mismatch')
    expected={a.backbone:'9228c12b5b34cdb3627fce731a3e3d5890c54dc78ca7eb357d66056893f5bf1f',
              a.branch:'eebd51ac66bab764b52c7c51a47671bfa6152980a5bc7da6ea7fe20f6ae61431',
              a.mhr:'d52ab772628fb6d851550381b428185a32da6398793f0ff70299bad1999ba8f8'}
    for path,identity in expected.items():
        if sha256_file(path)!=identity:raise ValueError('GGUF identity mismatch: '+str(path))
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise FileExistsError('output directory must be empty')
    input_path=a.output/'image.input';extract_image(a.reference/case['input'],input_path)
    env=os.environ.copy()
    if a.backend=='Vulkan':
        env['GGML_VK_DISABLE_F16']='1'
        if a.experimental_cm2:
            env['SAM3D_BF16_COOPMAT2']='1'
            for key in ['GGML_VK_DISABLE_COOPMAT','GGML_VK_DISABLE_COOPMAT2']:env.pop(key,None)
        else:
            env.pop('SAM3D_BF16_COOPMAT2',None)
            env.update(GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1')
    binary_hash=sha256_file(a.runner);module_hash=sha256_file(a.module)
    argv=[str(a.runner.resolve()),str(a.module.resolve()),a.backend,str(a.device),a.description,str(a.backbone.resolve()),str(a.branch.resolve()),str(a.mhr.resolve()),str(input_path.resolve()),str((a.output/'native.bin').resolve()),str(a.threads),str(int(m.get('decoder_operations',False)))]
    subprocess.run(argv,env=env,check=True)
    if sha256_file(a.runner)!=binary_hash or sha256_file(a.module)!=module_hash:raise ValueError('inference binary changed while running')
    flat=np.memmap(a.output/'native.bin',dtype='<f4',mode='r');tensors={};offset=0
    if case['order']!=sorted(set(case['order'])):raise ValueError('invalid original tap ordering')
    for name in case['order']:
        shape=case['shapes'][name];count=int(np.prod(shape));v=np.array(flat[offset:offset+count]).reshape(shape);offset+=count
        if not np.isfinite(v).all():raise ValueError('nonfinite native tensor '+name)
        tensors['case.0000.'+name]=v
    if offset!=flat.size:raise ValueError('native result layout/length mismatch')
    save_file(tensors,a.output/'native.safetensors')
    report=dict(scope='GGUF-only native trained body pose branch from RGB; no reference weights/intermediates supplied',
                reference_manifest_sha256=sha256_file(a.reference/'manifest.json'),gguf_sha256=expected[a.mhr],
                backbone_gguf_sha256=expected[a.backbone],branch_gguf_sha256=expected[a.branch],
                runner_sha256=binary_hash,module_sha256=module_hash,script_sha256=sha256_file(Path(__file__)),
                image_input_sha256=sha256_file(input_path),native_sha256=sha256_file(a.output/'native.safetensors'),
                argv=argv,returncode=0,backend=a.backend,threads=a.threads,
                experimental_cm2=a.experimental_cm2,batched_transfers=env.get('SAM3D_BATCHED_TRANSFERS'),simd_skinning=env.get('SAM3D_SIMD_SKINNING'),narrow_matmul=env.get('GGML_VK_F32_NARROW_MATMUL'),
                backend_environment={k:env.get(k) for k in ['GGML_VK_DISABLE_F16','GGML_VK_DISABLE_COOPMAT','GGML_VK_DISABLE_COOPMAT2','SAM3D_BF16_COOPMAT2','GGML_VK_FUSE_BF16_ROUND','GGML_VK_FUSE_BF16_BINARY','GGML_VK_BF16_BINARY_LINEAR','GGML_VK_BF16_MATMUL_TILE','GGML_VK_BF16_MATMUL_TRACE','SAM3D_BF16_FLASH_ATTENTION','SAM3D_BF16_PRECISE_PREFIX']})
    with (a.output/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(captured=len(tensors),runtime_inputs='RGB/box/intrinsics plus three GGUFs; no inline parameters',native_sha256=report['native_sha256'])),flush=True)
if __name__=='__main__':main()
