#!/usr/bin/env python3
"""Compare native raw RGBA preprocessing against original boundaries/full outputs."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file,save_file
from check_parity import sha256_file,read_rules,compare_array,compare_files

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['reference','output','runner']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 manifest=read_rules(a.reference/'manifest.json');rules=read_rules(a.reference/'rules.json')
 for name in ['upstream.safetensors','full.safetensors','rules.json']:
  if sha256_file(a.reference/name)!=manifest['artifacts'][name]:raise ValueError('reference artifact hash mismatch')
 tensors={};commands=[];full=load_file(a.reference/'full.safetensors');full_checks=[];expected_full=set()
 for case in manifest['cases']:
  name,prefix=case['input'],case['prefix']
  if Path(name).name!=name or Path(prefix).name!=prefix or case['order']!=sorted(set(case['order'])):raise ValueError('invalid case identity/order')
  if sha256_file(a.reference/name)!=manifest['artifacts'][name]:raise ValueError('input hash mismatch')
  target=a.output/(prefix+'.bin');command=[str(a.runner.resolve()),str((a.reference/name).resolve()),str(target.resolve())]
  r=subprocess.run(command,capture_output=True,text=True);commands.append({'argv':command,'returncode':r.returncode,'stderr':r.stderr})
  (a.output/'run.json').write_text(json.dumps({'commands':commands},indent=2)+'\n');r.check_returncode()
  flat=np.fromfile(target,dtype='<f4');offset=0
  for key in case['order']:
   shape=case['shapes'][key];count=int(np.prod(shape));tensors[prefix+'.'+key]=flat[offset:offset+count].reshape(shape).copy();offset+=count
  if offset!=flat.size:raise ValueError('trailing native capture')
  for key in ['image','mask','rgb_image','rgb_image_mask']:
   name=prefix+'.'+key;expected_full.add(name)
   rule=next(r for r in rules['tensors'] if r['name']==prefix+'.08.'+key).copy();rule['name']=prefix+'.full.'+key
   full_checks.append(compare_array(full[name],tensors[prefix+'.08.'+key],rule))
  print(json.dumps({'case':prefix,'native_completed':True}),flush=True)
 if set(full)!=expected_full:raise ValueError('missing or untested original full outputs')
 save_file(tensors,a.output/'native.safetensors');report=compare_files(a.reference/'upstream.safetensors',a.output/'native.safetensors',rules)
 report['full_outputs']=full_checks;report['pass']=report['pass'] and all(x['pass'] for x in full_checks)
 report['full_output_count']=len(full_checks);report['full_output_sha256']=sha256_file(a.reference/'full.safetensors')
 report['reference_manifest_sha256']=sha256_file(a.reference/'manifest.json');report['runner_sha256']=sha256_file(a.runner)
 report['rules_sha256']=sha256_file(a.reference/'rules.json');report['script_sha256']=sha256_file(Path(__file__))
 (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'pass':report['pass'],'checks':report['tensor_count']+len(full_checks),'failures':[x for x in report['tensors']+full_checks if not x['pass']][:8]},indent=2))
 if not report['pass']:raise SystemExit(1)
if __name__=='__main__':main()
