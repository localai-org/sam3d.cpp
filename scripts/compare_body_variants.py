#!/usr/bin/env python3
"""Compare every public output with a native baseline; differences are not ground-truth error."""
import argparse
import json
from pathlib import Path
import numpy as np
from check_body_api import read_output


def summary(values):
    a=np.asarray(values,dtype=np.float64)
    return {'mean':float(a.mean()),'p95':float(np.percentile(a,95)),'max':float(a.max())}


def compare(base,candidate):
    if set(base)!=set(candidate):raise ValueError('result fields differ')
    out={'exact':True,'fields':{}}
    for key in base:
        a,b=base[key],candidate[key]
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError(f'{key}: shape/type changed')
        if not np.isfinite(b).all():raise ValueError(f'{key}: nonfinite result')
        exact=a.tobytes()==b.tobytes()
        out['exact'] &= exact
        out['fields'][key]={'exact':exact,'max_abs':float(np.max(np.abs(a.astype(np.float64)-b))),
                            'mean_abs':float(np.mean(np.abs(a.astype(np.float64)-b)))}
    if not out['fields']['faces']['exact']:raise ValueError('mesh topology changed')
    for key in ('vertices','joints','keypoints'):
        a,b=base[key].reshape(-1,3).astype(np.float64),candidate[key].reshape(-1,3).astype(np.float64)
        if key=='joints':a,b=a[1:],b[1:] # Exclude artificial body_world.
        out[key+'_distance_mm']=summary(np.linalg.norm(a-b,axis=-1)*1000)
    a,b=base['joints'].reshape(-1,3).astype(np.float64)[1:],candidate['joints'].reshape(-1,3).astype(np.float64)[1:]
    out['pelvis_aligned_joints_mm']=summary(np.linalg.norm((a-a[:1])-(b-b[:1]),axis=-1)*1000)
    camera=base['camera_translation'].astype(np.float64)-candidate['camera_translation']
    out['camera_translation_distance_mm']=float(np.linalg.norm(camera)*1000)
    out['camera_relative_joints_mm']=summary(np.linalg.norm(a-b+camera,axis=-1)*1000)
    out['keypoint_projection_pixels']=summary(np.linalg.norm(base['keypoints_pixels'].astype(np.float64)-candidate['keypoints_pixels'],axis=-1))
    qa=base['joint_transforms'].reshape(-1,8)[1:,3:7].astype(np.float64)
    qb=candidate['joint_transforms'].reshape(-1,8)[1:,3:7].astype(np.float64)
    qa/=np.linalg.norm(qa,axis=-1,keepdims=True);qb/=np.linalg.norm(qb,axis=-1,keepdims=True)
    angle=2*np.arccos(np.clip(np.abs(np.sum(qa*qb,axis=-1)),0,1))*180/np.pi
    if np.array_equal(base['joint_transforms'],candidate['joint_transforms']):angle[:]=0
    out['joint_rotation_degrees']=summary(angle)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True,help='original benchmark directory')
    p.add_argument('--candidate',type=Path,required=True,help='candidate benchmark directory')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-exact',action='append',default=[])
    a=p.parse_args()
    old=json.loads((a.baseline/'report.json').read_text());new=json.loads((a.candidate/'report.json').read_text())
    for key in ('inputs','precision','backend','models','threads','environment'):
        if old[key]!=new[key]:raise ValueError(f'benchmark configuration differs: {key}')
    baseline=old['variants']['baseline'];out={'scope':__doc__,'baseline':str(a.baseline),'candidate':str(a.candidate),'variants':{}}
    for name,v in new['variants'].items():
        samples=[compare(read_output(a.baseline/'baseline'/f'input-{i}.bin'),read_output(a.candidate/name/f'input-{i}.bin')) for i in range(len(old['inputs']))]
        exact=all(x['exact'] for x in samples)
        out['variants'][name]={'median_ms':v['median_ms'],'speedup':baseline['median_ms']/v['median_ms'],'repeat_exact':v['repeat_exact'],'exact':exact,'samples':samples}
        print(f'{name:15s} {v["median_ms"]:7.2f} ms  {baseline["median_ms"]/v["median_ms"]:.2f}x  joints mean/max mm: '+', '.join(f'{s["joints_distance_mm"]["mean"]:.2f}/{s["joints_distance_mm"]["max"]:.2f}' for s in samples))
    for name in a.require_exact:
        if name not in out['variants'] or not out['variants'][name]['exact']:raise ValueError(f'{name}: expected exact outputs')
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
