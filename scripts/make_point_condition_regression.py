#!/usr/bin/env python3
"""Package small verified ORIGINAL references as download-free CTest fixtures."""
import argparse,json,shutil
from pathlib import Path
from safetensors.numpy import load_file
from check_parity import sha256_file
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['reference','preprocessing','output']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();m=json.loads((a.reference/'manifest.json').read_text());pm=json.loads((a.preprocessing/'manifest.json').read_text())
 if m['full_size'] or m['neural_device']!='cpu' or m['preprocessing_manifest_sha256']!=sha256_file(a.preprocessing/'manifest.json'):raise ValueError('wrong small original reference')
 a.output.mkdir(parents=True,exist_ok=True);artifacts={}
 for index,c in enumerate(m['cases']):
  stem=f'point-condition-{index}';raw=a.preprocessing/c['preprocess_input'];weights=a.reference/c['weights'];full=a.reference/c['full'];pc=pm['cases'][c['preprocess_case']];prep=a.preprocessing/pc['full']
  for file,sha in [(raw,pm['artifacts'][raw.name]),(weights,m['artifacts'][weights.name]),(full,m['artifacts'][full.name]),(prep,pm['artifacts'][prep.name])]:
   if sha256_file(file)!=sha:raise ValueError('original artifact changed')
  tensors={'00.prepared.'+k:v for k,v in load_file(prep).items()};tensors['90.result']=load_file(full)['conditioning']
  for i in range(2):
   t=c['taps'][f'10.encoder.{i}.90.output'];file=a.reference/t['file']
   if sha256_file(file)!=t['sha256']:raise ValueError('original encoder result changed')
   tensors[f'91.embedding.{i}']=load_file(file)['value']
  lines=['S3D_POINT_CONDITION_FINAL_V1',str(len(tensors))]
  for key,v in sorted(tensors.items()):
   v=v.reshape(-1);lines.append(f'{key} {len(v)}')
   for start in range(0,len(v),8):lines.append(' '.join(format(float(x),'.9g') for x in v[start:start+8]))
  (a.output/(stem+'.txt')).write_text('\n'.join(lines)+'\n');shutil.copyfile(raw,a.output/(stem+'.input'));shutil.copyfile(weights,a.output/(stem+'.weights'))
  for suffix in ['txt','input','weights']:artifacts[stem+'.'+suffix]=sha256_file(a.output/(stem+'.'+suffix))
 meta={'scope':m['scope'],'reference_manifest':m,'artifacts':artifacts,'packager_sha256':sha256_file(Path(__file__))}
 (a.output/'point-condition.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
