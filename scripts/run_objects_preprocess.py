#!/usr/bin/env python3
"""Stream composed original/native preprocessing boundaries and actual finals."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
from check_parity import read_rules,sha256_file,compare_array
from run_objects_ssi import read_capture

def check(name,ref,v):
 if ref.shape!=v.shape:raise ValueError('preprocess tensor shape mismatch')
 point_apply=any(f'04.apply.{i}.' in name for i in [2,5,6])
 points='pointmap' in name or name.endswith(('.clean','.resized','.scale','.shift')) or point_apply
 moments=name.endswith(('scale','shift'))
 if moments and not np.all(np.isfinite(ref)):raise ValueError('nonfinite normalization parameters')
 if any(not np.array_equal(fn(ref),fn(v)) for fn in [np.isnan,np.isposinf,np.isneginf]):return {'name':name,'pass':False,'error':'nonfinite category mismatch'}
 if not points and not np.all(np.isfinite(ref)):raise ValueError('unexpected original nonfinite value')
 valid=np.isfinite(ref);ref=np.where(valid,ref,0);v=np.where(valid,v,0)
 # Shared image interpolation was separately frozen at 1e-6; pointmap stages
 # retain SSI's pre-existing 1e-4 / 2e-5 bounds. Masks/indices remain exact.
 if points:rule={'name':name,'mode':'float','max_abs':1e-4,'relative_l2':2e-5,'zero_reference_floor':1e-12}
 elif 'mask' in name or 'bbox' in name or any(f'04.apply.{i}.' in name for i in [1,4]):rule={'name':name,'mode':'exact'}
 else:rule={'name':name,'mode':'float','max_abs':1e-6,'relative_l2':1e-6,'zero_reference_floor':1e-12}
 return compare_array(ref,v,rule)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['reference','output','runner']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);meta=read_rules(a.reference/'manifest.json')
 if meta['device']!='cpu' or not meta['observer_exact'] or meta['max_abs']!=1e-4 or meta['relative_l2']!=2e-5:raise ValueError('unexpected composed reference policy')
 def verified(name):
  if Path(name).name!=name or sha256_file(a.reference/name)!=meta['artifacts'][name]:raise ValueError('invalid original artifact '+name)
  return a.reference/name
 checks=[];artifacts={}
 for case in meta['cases']:
  prefix=case['prefix']
  if Path(prefix).name!=prefix:raise ValueError('invalid case name')
  dest=a.output/prefix;dest.mkdir(exist_ok=True)
  subprocess.run([str(a.runner.resolve()),str(verified(case['input']).resolve()),str(dest.resolve())],check=True)
  for key,tap in sorted(case['taps'].items()):
   if Path(key).name!=key:raise ValueError('invalid tensor name')
   ref=load_file(verified(tap['file']))['value'];path=dest/(key+'.bin');values=read_capture(path,max_elements=4*4096*4096)
   if set(values)!={key} or values[key].size!=ref.size or list(ref.shape)!=tap['shape']:raise ValueError('invalid native boundary')
   v=values[key].reshape(ref.shape);checks.append(check(prefix+'.'+key,ref,v));artifacts[str(path.relative_to(a.output))]=sha256_file(path)
  full=load_file(verified(case['full']))
  for key,ref in sorted(full.items()):
   name='90.final.'+key;path=dest/(name+'.bin');values=read_capture(path,max_elements=4*4096*4096)
   if set(values)!={name} or values[name].size!=ref.size:raise ValueError('invalid actual returned tensor')
   checks.append(check(prefix+'.'+name,ref,values[name].reshape(ref.shape)));artifacts[str(path.relative_to(a.output))]=sha256_file(path)
  expected={k+'.bin' for k in case['taps']}|{'90.final.'+k+'.bin' for k in full}
  if {p.name for p in dest.iterdir()}!=expected:raise ValueError('missing/extra native field')
  print(json.dumps({'case':prefix,'checks':len(checks),'failed':sum(not c['pass'] for c in checks)}),flush=True)
 report={'scope':meta['scope'],'pass':all(c['pass'] for c in checks),'checks':checks,'artifacts':artifacts,
  'runner_sha256':sha256_file(a.runner),'reference_manifest_sha256':sha256_file(a.reference/'manifest.json'),'script_sha256':sha256_file(Path(__file__))}
 (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'pass':report['pass'],'checks':len(checks),'failures':[c for c in checks if not c['pass']][:10]},indent=2))
 if not report['pass']:raise SystemExit(1)
if __name__=='__main__':main()
