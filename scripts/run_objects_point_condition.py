#!/usr/bin/env python3
"""Native raw-input point-conditioning composition, verified upstream references."""
import argparse,json,os,struct,subprocess
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
from check_parity import read_rules,sha256_file,compare_array
from run_objects_preprocess import check as preprocess_check
def check(name,reference,actual):
 if name.startswith('00.prepared.'):return preprocess_check(name,reference,actual)
 return compare_array(reference,actual,{'name':name,'mode':'float','max_abs':1e-4,'relative_l2':2e-5,'zero_reference_floor':1e-12})
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['reference','preprocessing','output','runner','module']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--backend',choices=['CPU','Vulkan'],default='CPU');p.add_argument('--device',type=int,default=0);p.add_argument('--description',default='-');p.add_argument('--chunk',type=int,default=32);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 meta=read_rules(a.reference/'manifest.json');pre=read_rules(a.preprocessing/'manifest.json')
 if sha256_file(a.preprocessing/'manifest.json')!=meta['preprocessing_manifest_sha256'] or not meta['observer_exact'] or meta['neural_device']!=('cpu' if a.backend=='CPU' else 'cuda'):raise ValueError('incompatible original composition reference')
 env=os.environ.copy()
 if a.backend=='Vulkan':env.update(GGML_VK_DISABLE_F16='1',GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1')
 checks=[];artifacts={};rejections=[]
 for case in meta['cases']:
  prefix=case['prefix'];name=case['preprocess_input'];weights=case['weights']
  if any(Path(k).name!=k for k in [prefix,name,weights]):raise ValueError('invalid case path')
  if sha256_file(a.preprocessing/name)!=pre['artifacts'][name] or sha256_file(a.reference/weights)!=meta['artifacts'][weights]:raise ValueError('changed raw input/state')
  dest=a.output/prefix;dest.mkdir(exist_ok=True)
  subprocess.run([str(a.runner.resolve()),str(a.module.resolve()),a.backend,str(a.device),a.description,str((a.preprocessing/name).resolve()),str((a.reference/weights).resolve()),str(dest.resolve()),'1',str(a.chunk)],env=env,check=True)
  expected={key+'.bin' for key in case['taps']}|{'90.result.bin','91.embedding.0.bin','91.embedding.1.bin'}
  if {p.name for p in dest.iterdir()}!=expected:raise ValueError('missing/extra composed native boundary')
  for key,info in sorted(case['taps'].items()):
   file=info['file']
   if Path(key).name!=key or Path(file).name!=file or sha256_file(a.reference/file)!=info['sha256']:raise ValueError('changed original composition tap')
   r=load_file(a.reference/file)['value'];path=dest/(key+'.bin');v=np.fromfile(path,dtype='<f4')
   if list(r.shape)!=info['shape'] or v.size!=r.size:raise ValueError('composition tensor extent mismatch')
   checks.append(check(key,r,v.reshape(r.shape)));checks[-1]['name']=prefix+'.'+key;artifacts[str(path.relative_to(a.output))]=sha256_file(path)
  if Path(case['full']).name!=case['full']:raise ValueError('invalid original result path')
  path=a.reference/case['full']
  if sha256_file(path)!=meta['artifacts'][path.name]:raise ValueError('changed original complete output')
  r=load_file(path)['conditioning'];v=np.fromfile(dest/'90.result.bin',dtype='<f4');checks.append(check('90.result',r,v.reshape(r.shape)));checks[-1]['name']=prefix+'.90.result';artifacts[prefix+'/90.result.bin']=sha256_file(dest/'90.result.bin')
  for i in range(2):
   info=case['taps'][f'10.encoder.{i}.90.output'];r=load_file(a.reference/info['file'])['value'];v=np.fromfile(dest/f'91.embedding.{i}.bin',dtype='<f4');checks.append(check(f'91.embedding.{i}',r,v.reshape(r.shape)));checks[-1]['name']=prefix+f'.91.embedding.{i}'
  for invalid in meta['invalid_configurations']:
   if meta['cases'][invalid['case']]['prefix']!=prefix:continue
   bad=bytearray((a.reference/weights).read_bytes());bad[24:28]=struct.pack('<I',2);bad_path=a.output/(prefix+'.invalid.weights');bad_path.write_bytes(bad)
   proc=subprocess.run([str(a.runner.resolve()),str(a.module.resolve()),a.backend,str(a.device),a.description,str((a.preprocessing/name).resolve()),str(bad_path.resolve()),str((a.output/(prefix+'.invalid')).resolve()),'1',str(a.chunk)],env=env,capture_output=True,text=True)
   ok=proc.returncode==1 and invalid['native_expected'] in proc.stderr
   rejections.append({'case':prefix,'pass':ok,'returncode':proc.returncode,'stderr':proc.stderr,'upstream':invalid})
  print(json.dumps({'case':prefix,'checks':len(checks),'failed':sum(not c['pass'] for c in checks)}),flush=True)
 report={'scope':meta['scope'],'backend':a.backend,'checks':checks,'rejections':rejections,'pass':all(c['pass'] for c in checks+rejections),'artifacts':artifacts,'chunk_windows':a.chunk,
  'reference_manifest_sha256':sha256_file(a.reference/'manifest.json'),'script_sha256':sha256_file(Path(__file__)),'runner_sha256':sha256_file(a.runner),'module_sha256':sha256_file(a.module)}
 (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'pass':report['pass'],'checks':len(checks),'failures':[c for c in checks if not c['pass']][:8]},indent=2))
 if not report['pass']:raise SystemExit(1)
if __name__=='__main__':main()
