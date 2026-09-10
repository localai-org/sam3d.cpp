#!/usr/bin/env python3
"""Native complete MHR geometry versus original operations and final vertices."""
import argparse,json,os,struct,subprocess
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file,save_file
from run_mhr_skeleton import digest

def read_output(path):
    with path.open('rb') as f:
        def read(n):
            b=f.read(n)
            if len(b)!=n:raise ValueError('truncated geometry capture')
            return b
        if read(8)!=b'S3DMGO01':raise ValueError('invalid geometry capture')
        count,=struct.unpack('<I',read(4))
        if count not in [19,25]:raise ValueError('invalid geometry tap count')
        result={}
        for _ in range(count):
            size,=struct.unpack('<I',read(4))
            if not 0<size<100:raise ValueError('invalid tap name')
            name=read(size).decode('ascii');count,=struct.unpack('<Q',read(8))
            if name in result or not 0<count<=1000000:raise ValueError('invalid tap')
            result[name]=np.frombuffer(read(count*4),dtype='<f4').copy()
        if f.read(1):raise ValueError('trailing geometry capture')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['runner','module','gguf','reference','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--backend',choices=['CPU','Vulkan'],default='CPU');p.add_argument('--device',type=int,default=0);p.add_argument('--description',default='-');p.add_argument('--threads',type=int,default=1);a=p.parse_args()
    manifest=json.loads((a.reference/'manifest.json').read_text());a.output.mkdir(parents=True,exist_ok=True);checks=[];env=os.environ.copy()
    if a.backend=='Vulkan':env.update(GGML_VK_DISABLE_F16='1',GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1')
    for enabled in [False,True]:
        prefix=f'correctives_{int(enabled)}';root=a.reference/prefix;out=a.output/prefix;out.mkdir(exist_ok=True)
        for filename in ['input.bin','upstream.safetensors','full.safetensors']:
            if digest(root/filename)!=manifest['artifacts'][prefix+'/'+filename]:raise ValueError('reference identity mismatch')
        subprocess.run([str(a.runner.resolve()),str(a.module.resolve()),a.backend,str(a.device),a.description,str(a.gguf.resolve()),str(root/'input.bin'),str(out/'native.bin'),str(a.threads)],env=env,check=True)
        actual=read_output(out/'native.bin');reference=load_file(root/'upstream.safetensors');full=load_file(root/'full.safetensors');shaped={}
        if actual.keys()!=reference.keys():raise ValueError('tap set mismatch')
        def check(name,v,r):
            if v.shape!=r.shape or v.dtype!=np.float32 or r.dtype!=np.float32 or not np.isfinite(v).all() or not np.isfinite(r).all():raise ValueError('invalid shape/dtype/finite result')
            d=v.astype(np.float64)-r.astype(np.float64);maximum=float(np.abs(d).max());relative=float(np.linalg.norm(d)/max(np.linalg.norm(r.astype(np.float64)),1e-12))
            checks.append({'correctives':enabled,'name':name,'max_abs':maximum,'relative_l2':relative,'pass':maximum<=1e-4 and relative<=2e-5})
        for name,r in sorted(reference.items()):shaped[name]=actual[name].reshape(r.shape);check(name,shaped[name],r)
        check('91.full_model_vertices',shaped['90.vertices'],full['vertices']);check('92.full_model_skeleton',shaped['11.skeleton'],full['skeleton'])
        save_file(shaped,out/'native.safetensors')
    report={'scope':'complete released MHR geometry, own GGUF/native intermediates; not SAM neural image-to-body, output mapping or performance acceptance',
        'reference_manifest_sha256':digest(a.reference/'manifest.json'),'runner_sha256':digest(a.runner),'backend_module_sha256':digest(a.module),
        'gguf_sha256':digest(a.gguf),'script_sha256':digest(Path(__file__)),'backend':a.backend,'reference_device':manifest['device'],'threads':a.threads,
        'thresholds':{'max_abs':1e-4,'relative_l2':2e-5},'checks':checks,'passed':all(x['pass'] for x in checks)}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':report['passed'],'checks':len(checks),'failed':[x for x in checks if not x['pass']][:5]},indent=2))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
