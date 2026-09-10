#!/usr/bin/env python3
"""Compare streamed native PointPatch tensors against original own-input outputs."""
import argparse,json,os,subprocess
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
from check_parity import compare_array,sha256_file,read_rules

def check(name,reference,actual,allow_nonfinite,exact=False):
 if actual.shape!=reference.shape:raise ValueError('shape mismatch '+name)
 # Nonfinite values are meaningful ONLY in explicitly declared invalid-point
 # boundaries. Compare NaN/+Inf/-Inf categories exactly, never ignore them.
 if allow_nonfinite:
  masks=[np.isnan,np.isposinf,np.isneginf]
  if any(not np.array_equal(fn(reference),fn(actual)) for fn in masks):return {'name':name,'pass':False,'error':'invalid-point nonfinite category mismatch'}
  valid=np.isfinite(reference);reference=np.where(valid,reference,0);actual=np.where(valid,actual,0)
 rule={'name':name,'mode':'exact'} if exact else {'name':name,'mode':'float','max_abs':1e-4,'relative_l2':2e-5,'zero_reference_floor':1e-12}
 return compare_array(reference,actual,rule)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['reference','output','runner','module']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--backend',choices=['CPU','Vulkan'],default='CPU');p.add_argument('--device',type=int,default=0);p.add_argument('--description',default='-');p.add_argument('--threads',type=int,default=1);p.add_argument('--chunk',type=int,default=32);a=p.parse_args()
 manifest=read_rules(a.reference/'manifest.json');a.output.mkdir(parents=True,exist_ok=True);checks=[]
 if manifest['device']!=('cpu' if a.backend=='CPU' else 'cuda'):raise ValueError('wrong reference device')
 if manifest['max_abs_limit']!=1e-4 or manifest['relative_l2_limit']!=2e-5 or manifest['nonfinite_allowed']!=['00.resized','03.remapped','04.projected']:raise ValueError('unexpected original rules')
 env=os.environ.copy()
 if a.backend=='Vulkan':env.update(GGML_VK_DISABLE_F16='1',GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1')
 for case in manifest['cases']:
  prefix,name=case['prefix'],case['input']
  if Path(prefix).name!=prefix or Path(name).name!=name or sha256_file(a.reference/name)!=manifest['artifacts'][name]:raise ValueError('input identity mismatch')
  target=a.output/prefix;subprocess.run([str(a.runner.resolve()),str(a.module.resolve()),a.backend,str(a.device),a.description,str((a.reference/name).resolve()),str(target.resolve()),str(a.threads),str(a.chunk)],env=env,check=True)
  if set(p.name for p in target.iterdir())!={key+'.bin' for key in case['order']}|{'result.bin'}:raise ValueError('incomplete/unexpected native tap set')
  for key in case['order']:
   info=case['taps'][key];path=a.reference/prefix/info['file']
   if Path(info['file']).name!=info['file'] or sha256_file(path)!=info['sha256']:raise ValueError('original tap identity mismatch')
   r=load_file(path)['value'];v=np.memmap(target/(key+'.bin'),dtype='<f4',mode='r')
   if r.shape!=tuple(info['shape']) or v.size!=r.size:raise ValueError('tap shape mismatch')
   checks.append(check(prefix+'.'+key,r,v.reshape(r.shape),key in manifest['nonfinite_allowed'],key in ['00.resized','01.valid','02.safe','32.position']))
  path=a.reference/prefix/'full.safetensors'
  if sha256_file(path)!=case['full_sha256']:raise ValueError('full output hash mismatch')
  r=load_file(path)['output'];v=np.fromfile(target/'result.bin',dtype='<f4').reshape(r.shape);checks.append(check(prefix+'.full.output',r,v,False))
  print(json.dumps({'case':prefix,'checks':len(checks),'failed':sum(not x['pass'] for x in checks)}),flush=True)
 report={'scope':manifest['scope'],'backend':a.backend,'chunk_windows':a.chunk,'threads':a.threads,'reference_manifest_sha256':sha256_file(a.reference/'manifest.json'),
  'runner_sha256':sha256_file(a.runner),'module_sha256':sha256_file(a.module),'script_sha256':sha256_file(Path(__file__)),'pass':all(x['pass'] for x in checks),'checks':checks}
 (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'pass':report['pass'],'checks':len(checks),'failures':[x for x in checks if not x['pass']][:8]},indent=2))
 if not report['pass']:raise SystemExit(1)
if __name__=='__main__':main()
