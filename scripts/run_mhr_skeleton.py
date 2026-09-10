#!/usr/bin/env python3
"""Run native parameter-to-skeleton parity against isolated original MHR taps."""
import argparse,hashlib,json,os,struct,subprocess
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file,save_file

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read_output(path):
    with path.open('rb') as f:
        def read(n):
            data=f.read(n)
            if len(data)!=n:raise ValueError('truncated native capture')
            return data
        if read(8)!=b'S3DMHO01':raise ValueError('wrong native capture')
        count,=struct.unpack('<I',read(4))
        if count!=46:raise ValueError('wrong tap count')
        results={}
        for _ in range(count):
            size,=struct.unpack('<I',read(4))
            if not 0<size<100:raise ValueError('invalid tap name')
            name=read(size).decode('ascii');dtype,count=struct.unpack('<IQ',read(12))
            if name in results or dtype not in [4,8] or not 0<count<100000:raise ValueError('invalid tap shape/type')
            results[name]=np.frombuffer(read(count*dtype),dtype='<f'+str(dtype)).copy()
        if f.read(1):raise ValueError('trailing native capture')
    return results

def fixture(reference,output):
    data=load_file(reference/'local-inputs.safetensors');taps=load_file(reference/'upstream.safetensors')
    # This normal test is explicitly an isolated local/FK regression. The large
    # parameter projection is tested through GGUF separately, never injected in
    # that composed run. Only small Apache-licensed skeleton constants included.
    selected={k:v for k,v in taps.items() if k not in ['00.padded','01.joint_parameters']}
    with output.open('x') as f:
        f.write('S3D_MHR_LOCAL_V1 2\n')
        for name in ['offsets','prerotations','prefix','parents','joint_parameters']:
            v=data[name];f.write(f'{name} {v.size}\n');f.write(' '.join(format(x,'.9g') if v.dtype.kind=='f' else str(int(x)) for x in v.flat)+'\n')
        f.write(str(len(selected))+'\n')
        for name,v in sorted(selected.items()):
            f.write(f'{name} {v.dtype.itemsize} {v.size}\n');f.write(' '.join(format(x,'.9g' if v.dtype.itemsize==4 else '.17g') for x in v.flat)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--fixture',type=Path)
    p.add_argument('--runner',type=Path);p.add_argument('--module',type=Path);p.add_argument('--gguf',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--backend',choices=['CPU','Vulkan'],default='CPU');p.add_argument('--device',type=int,default=0);p.add_argument('--description',default='-');p.add_argument('--threads',type=int,default=1);a=p.parse_args()
    manifest=json.loads((a.reference/'manifest.json').read_text())
    for name in ['input.bin','upstream.safetensors','local-inputs.safetensors','full-skeleton.safetensors']:
        if digest(a.reference/name)!=manifest['artifacts'][name]:raise ValueError('reference hash mismatch')
    if a.fixture:fixture(a.reference,a.fixture);return
    if any(x is None for x in [a.runner,a.module,a.gguf,a.output]):raise ValueError('runner/module/GGUF/output required')
    a.output.mkdir(parents=True,exist_ok=True);env=os.environ.copy()
    if a.backend=='Vulkan':env.update(GGML_VK_DISABLE_F16='1',GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1')
    command=[str(a.runner.resolve()),str(a.module.resolve()),a.backend,str(a.device),a.description,str(a.gguf.resolve()),str(a.reference/'input.bin'),str(a.output/'native.bin'),str(a.threads)]
    subprocess.run(command,env=env,check=True)
    actual=read_output(a.output/'native.bin');reference=load_file(a.reference/'upstream.safetensors')
    if actual.keys()!=reference.keys():raise ValueError('native/reference tap mismatch')
    rows=[];failed=[];shaped={}
    for name,r in sorted(reference.items()):
        v=actual[name]
        if v.size!=r.size or v.dtype!=r.dtype or not np.isfinite(v).all():raise ValueError('native shape/dtype/finite failure')
        v=v.reshape(r.shape);shaped[name]=v;d=v.astype(np.float64)-r.astype(np.float64)
        maximum=float(np.abs(d).max());relative=float(np.linalg.norm(d)/max(np.linalg.norm(r.astype(np.float64)),1e-12))
        ok=maximum<=1e-4 and relative<=2e-5
        if name=='00.padded':ok=maximum==0
        row={'name':name,'max_abs':maximum,'relative_l2':relative,'pass':ok};rows.append(row)
        if not ok:failed.append(row)
    baseline=load_file(a.reference/'full-skeleton.safetensors')['skeleton']
    if baseline.shape!=shaped['90.skeleton'].shape or baseline.dtype!=np.float32 or not np.isfinite(baseline).all():raise ValueError('invalid full skeleton baseline')
    delta=shaped['90.skeleton'].astype(np.float64)-baseline.astype(np.float64)
    maximum=float(np.abs(delta).max());relative=float(np.linalg.norm(delta)/max(np.linalg.norm(baseline.astype(np.float64)),1e-12))
    row={'name':'91.uninstrumented_full_model_skeleton','max_abs':maximum,'relative_l2':relative,'pass':maximum<=1e-4 and relative<=2e-5};rows.append(row)
    if not row['pass']:failed.append(row)
    save_file(shaped,a.output/'native.safetensors')
    report={'scope':'own GGUF model-parameter projection through MHR skeleton; no identity, correctives, skinning or SAM neural inference',
        'reference_manifest_sha256':digest(a.reference/'manifest.json'),'gguf_sha256':digest(a.gguf),'runner_sha256':digest(a.runner),
        'backend_module_sha256':digest(a.module),'backend':a.backend,'reference_device':manifest['device'],'threads':a.threads,
        'thresholds':{'max_abs':1e-4,'relative_l2':2e-5},'checks':rows,'passed':not failed}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':not failed,'boundaries':len(rows),'max_abs':max(x['max_abs'] for x in rows),'failed':failed[:3]},indent=2))
    if failed:raise SystemExit(1)
if __name__=='__main__':main()
