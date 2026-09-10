#!/usr/bin/env python3
"""Original/native post-encoder fusion boundary and independent-output checks."""
import argparse,json,os,subprocess
from pathlib import Path
from safetensors.numpy import load_file,save_file
from check_parity import read_rules,sha256_file,compare_array
from run_objects_ssi import read_capture

def check(name,r,v):
 rule={'name':name,'mode':'float','max_abs':1e-4,'relative_l2':2e-5,'zero_reference_floor':1e-12}
 # Input echoes are exact; all intermediate/final values must be finite.
 if name.endswith('.input') and '.projection.' not in name and '.compression.' not in name:rule={'name':name,'mode':'exact'}
 return compare_array(r,v,rule)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['reference','output','runner','module']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--backend',choices=['CPU','Vulkan'],default='CPU');p.add_argument('--device',type=int,default=0);p.add_argument('--description',default='-');a=p.parse_args()
 meta=read_rules(a.reference/'manifest.json');a.output.mkdir(parents=True,exist_ok=True)
 if meta['device']!=('cpu' if a.backend=='CPU' else 'cuda') or not meta['observer_exact'] or meta['max_abs']!=1e-4 or meta['relative_l2']!=2e-5:raise ValueError('wrong fuser reference policy')
 env=os.environ.copy()
 if a.backend=='Vulkan':env.update(GGML_VK_DISABLE_F16='1',GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1')
 checks=[];artifacts={}
 for case in meta['cases']:
  prefix=case['prefix'];name=case['input']
  if Path(prefix).name!=prefix or Path(name).name!=name or sha256_file(a.reference/name)!=meta['artifacts'][name]:raise ValueError('invalid original fusion input')
  path=a.output/(prefix+'.bin');subprocess.run([str(a.runner.resolve()),str(a.module.resolve()),a.backend,str(a.device),a.description,str((a.reference/name).resolve()),str(path.resolve()),'1'],env=env,check=True)
  taps=read_capture(path,max_tensors=256)
  if set(taps)!=set(case['shapes'])|{'91.independent_output'}:raise ValueError('missing/extra native fusion boundary')
  source=a.reference/prefix/'upstream.safetensors'
  if sha256_file(source)!=case['taps_sha256']:raise ValueError('original fusion taps changed')
  original=load_file(source)
  if set(original)!=set(case['shapes']):raise ValueError('missing original field')
  candidate={}
  for key,r in original.items():
   if list(r.shape)!=case['shapes'][key] or taps[key].size!=r.size:raise ValueError('fusion tensor extent mismatch')
   v=taps[key].reshape(r.shape);checks.append(check(prefix+'.'+key,r,v));candidate[key]=v
  source=a.reference/prefix/'full.safetensors'
  if sha256_file(source)!=case['full_sha256']:raise ValueError('original independent output changed')
  r=load_file(source)['output'];v=taps['91.independent_output']
  if v.size!=r.size:raise ValueError('fusion output extent mismatch')
  checks.append(check(prefix+'.full.output',r,v.reshape(r.shape)));candidate['91.independent_output']=v.reshape(r.shape)
  save_file(candidate,a.output/(prefix+'.safetensors'));artifacts[prefix]=sha256_file(a.output/(prefix+'.safetensors'))
  print(json.dumps({'case':prefix,'checks':len(checks),'failed':sum(not c['pass'] for c in checks)}),flush=True)
 report={'scope':meta['scope'],'backend':a.backend,'checks':checks,'pass':all(c['pass'] for c in checks),'artifacts':artifacts,
  'reference_manifest_sha256':sha256_file(a.reference/'manifest.json'),'runner_sha256':sha256_file(a.runner),'module_sha256':sha256_file(a.module),'script_sha256':sha256_file(Path(__file__))}
 (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'pass':report['pass'],'checks':len(checks),'failures':[c for c in checks if not c['pass']][:8]},indent=2))
 if not report['pass']:raise SystemExit(1)
if __name__=='__main__':main()
