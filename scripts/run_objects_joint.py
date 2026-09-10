#!/usr/bin/env python3
"""Compare native raw RGBA/XYZ joint preparation against original transforms."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from check_parity import read_rules,sha256_file,compare_array
from run_objects_ssi import read_capture

POINTS={'04.aligned_pointmap','06.clean','07.resized','12.crop_pointmap','22.pointmap'}
def joint_check(name,ref,actual,point=False):
 if point:
  if ref.shape!=actual.shape:raise ValueError('pointmap shape mismatch')
  if any(not np.array_equal(fn(ref),fn(actual)) for fn in [np.isnan,np.isposinf,np.isneginf]):return {'name':name,'pass':False,'error':'nonfinite category mismatch'}
  valid=np.isfinite(ref);ref=np.where(valid,ref,0);actual=np.where(valid,actual,0)
 rule={'name':name,'mode':'float','max_abs':1e-5,'relative_l2':2e-5,'zero_reference_floor':1e-12} if point else {'name':name,'mode':'exact'}
 return compare_array(ref,actual,rule)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['reference','output','runner']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);meta=read_rules(a.reference/'manifest.json')
 if meta['device']!='cpu' or meta['max_abs']!=1e-5 or meta['relative_l2']!=2e-5 or not meta['observer_exact']:raise ValueError('unexpected original joint rules')
 for name in ['upstream.safetensors','full.safetensors']:
  if sha256_file(a.reference/name)!=meta['artifacts'][name]:raise ValueError('original joint artifact mismatch')
 def read_original(file,key):
  with safe_open(a.reference/file,framework='numpy') as f:return f.get_tensor(key)
 with safe_open(a.reference/'upstream.safetensors',framework='numpy') as f:original_keys=set(f.keys())
 with safe_open(a.reference/'full.safetensors',framework='numpy') as f:full_keys=set(f.keys())
 native={};native_final={};checks=[];seen=set()
 for case in meta['cases']:
  prefix,name=case['prefix'],case['input']
  if Path(prefix).name!=prefix or Path(name).name!=name or case['order']!=sorted(set(case['order'])):raise ValueError('invalid case identity')
  if sha256_file(a.reference/name)!=meta['artifacts'][name]:raise ValueError('raw joint input hash mismatch')
  path=a.output/(prefix+'.bin');subprocess.run([str(a.runner.resolve()),str((a.reference/name).resolve()),str(path.resolve())],check=True)
  taps=read_capture(path,max_elements=4*4096*4096)
  if set(taps)!=set(case['order'])|{'90.result_rgb','91.result_mask','92.result_pointmap'}:raise ValueError('missing/extra joint tap or returned field')
  for key in case['order']:
   r=read_original('upstream.safetensors',prefix+'.'+key);v=taps[key]
   if list(r.shape)!=case['shapes'][key] or v.size!=r.size:raise ValueError('joint extent mismatch')
   v=v.reshape(r.shape);native[prefix+'.'+key]=v;checks.append(joint_check(prefix+'.'+key,r,v,key in POINTS))
  for key,tap in [('rgb','90.result_rgb'),('mask','91.result_mask'),('pointmap','92.result_pointmap')]:
   name=prefix+'.'+key;seen.add(name);r=read_original('full.safetensors',name);v=taps[tap]
   if v.size!=r.size:raise ValueError('returned buffer extent mismatch')
   v=v.reshape(r.shape);native_final[name]=v;checks.append(joint_check(prefix+'.full.'+key,r,v,key=='pointmap'))
  print(json.dumps({'case':prefix,'checks':len(checks),'failed':sum(not c['pass'] for c in checks)}),flush=True)
 if original_keys!=set(native) or full_keys!=seen:raise ValueError('untested original output')
 save_file(native,a.output/'native.safetensors');save_file(native_final,a.output/'native-final.safetensors');report={'scope':meta['scope'],'checks':checks,'pass':all(c['pass'] for c in checks),
  'runner_sha256':sha256_file(a.runner),'reference_manifest_sha256':sha256_file(a.reference/'manifest.json'),'script_sha256':sha256_file(Path(__file__)),'native_sha256':sha256_file(a.output/'native.safetensors'),'native_final_sha256':sha256_file(a.output/'native-final.safetensors')}
 (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'pass':report['pass'],'checks':len(checks),'failures':[c for c in checks if not c['pass']][:10]},indent=2))
 if not report['pass']:raise SystemExit(1)
if __name__=='__main__':main()
