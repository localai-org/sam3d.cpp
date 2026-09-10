#!/usr/bin/env python3
"""Compare native CPU SSI preprocessing against original CPU or CUDA operations."""
import argparse,json,struct,subprocess
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file,save_file
from check_parity import read_rules,sha256_file
from run_objects_pointpatch import check

NONFINITE={'00.remapped','02.centered','03.statistic','11.homogeneous','12.transformed','13.unclipped',
           '20.normalized','31.homogeneous','32.transformed','33.metric_space','40.denormalized'}
def read_capture(path,max_elements=4*2048*2048,max_tensors=32):
 if not 1<=max_elements<=4*4096*4096:raise ValueError('invalid capture allocation bound')
 tensors={}
 with path.open('rb') as f:
  def read(n):
   b=f.read(n)
   if len(b)!=n:raise ValueError('truncated native SSI capture')
   return b
  if read(8)!=b'S3DST001':raise ValueError('invalid SSI capture tag')
  count=struct.unpack('<I',read(4))[0]
  if not 1<=max_tensors<=256 or not 1<=count<=max_tensors:raise ValueError('invalid capture tap count')
  for _ in range(count):
   length=struct.unpack('<I',read(4))[0]
   if not 1<=length<=64:raise ValueError('invalid SSI key length')
   key=read(length).decode('ascii');n=struct.unpack('<Q',read(8))[0]
   if key in tensors or not 1<=n<=max_elements:raise ValueError('invalid SSI tensor extent/key')
   tensors[key]=np.frombuffer(read(n*4),dtype='<f4').copy()
  if f.read(1):raise ValueError('trailing native SSI capture')
 return tensors

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['reference','output','runner']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);meta=read_rules(a.reference/'manifest.json')
 if meta['max_abs']!=1e-4 or meta['relative_l2']!=2e-5 or not meta['observer_exact'] or meta['repeats']!=3:raise ValueError('unexpected original SSI rules')
 for name in ['upstream.safetensors','full.safetensors']:
  if sha256_file(a.reference/name)!=meta['artifacts'][name]:raise ValueError('original artifact identity mismatch')
 original=load_file(a.reference/'upstream.safetensors');full=load_file(a.reference/'full.safetensors');native={};checks=[];seen_full=set()
 for case in meta['cases']:
  name,prefix=case['input'],case['prefix']
  if Path(name).name!=name or Path(prefix).name!=prefix or case['order']!=sorted(set(case['order'])):raise ValueError('invalid case identity/order')
  if sha256_file(a.reference/name)!=meta['artifacts'][name]:raise ValueError('input identity mismatch')
  path=a.output/(prefix+'.bin');subprocess.run([str(a.runner.resolve()),str((a.reference/name).resolve()),str(path.resolve())],check=True)
  tensors=read_capture(path)
  if set(tensors)!=set(case['order']):raise ValueError('native tap set differs from original')
  for key in case['order']:
   r=original[prefix+'.'+key];v=tensors[key]
   if list(r.shape)!=case['shapes'][key] or v.size!=r.size:raise ValueError('tap extent mismatch')
   v=v.reshape(r.shape);native[prefix+'.'+key]=v
   checks.append(check(prefix+'.'+key,r,v,key in NONFINITE,key=='01.mask'))
  for key,tap in [('pointmap','20.normalized'),('scale','21.scale'),('shift','22.shift'),('denormalized','40.denormalized')]:
   name=prefix+'.'+key;seen_full.add(name);checks.append(check(prefix+'.full.'+key,full[name],native[prefix+'.'+tap],key in ['pointmap','denormalized']))
  print(json.dumps({'case':prefix,'checks':len(checks),'failed':sum(not x['pass'] for x in checks)}),flush=True)
 if set(original)!=set(native) or set(full)!=seen_full:raise ValueError('missing/untested original tensors')
 save_file(native,a.output/'native.safetensors')
 report={'scope':meta['scope'],'native_backend':'CPU preprocessing','reference_device':meta['device'],'checks':checks,'pass':all(x['pass'] for x in checks),
  'reference_manifest_sha256':sha256_file(a.reference/'manifest.json'),'runner_sha256':sha256_file(a.runner),'script_sha256':sha256_file(Path(__file__)),'native_sha256':sha256_file(a.output/'native.safetensors')}
 (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'pass':report['pass'],'checks':len(checks),'failures':[x for x in checks if not x['pass']][:12]},indent=2))
 if not report['pass']:raise SystemExit(1)
if __name__=='__main__':main()
