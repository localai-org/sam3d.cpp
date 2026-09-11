#!/usr/bin/env python3
"""Verify every public Body result against an accepted own-intermediate run."""
import argparse,json,struct
from pathlib import Path
import numpy as np
from safetensors import safe_open
from check_parity import sha256_file

# Public logical shapes and independently named native acceptance boundaries.
FIELDS={
 'vertices':('layer.5.pose.map.90.vertices',(1,18439,3)),
 'joints':('layer.5.pose.map.91.joints',(1,127,3)),
 'joint_rotations':('layer.5.pose.map.03.joint_rotations',(1,127,3,3)),
 'keypoints':('layer.5.pose.map.92.keypoints',(1,70,3)),
 'keypoints_pixels':('layer.5.camera.20.pixels',(1,70,2)),
 'vertices_pixels':('layer.5.03.vertex_pixels',(1,18439,2)),
 'camera_translation':('layer.5.camera.15.translation',(1,3)),
 'camera_parameters':('layer.5.camera.10.pred_cam',(1,3)),
 'pose_raw':('layer.5.pose.pred_pose_raw',(1,266)),
 'global_rotation':('layer.5.pose.global_rot',(1,3)),
 'body_pose':('layer.5.pose.body_pose',(1,133)),
 'shape':('layer.5.pose.pose.24.shape',(1,45)),
 'scale':('layer.5.pose.pose.25.scale',(1,28)),
 'hand':('layer.5.pose.pose.26.hand',(1,108)),
 'face':('layer.5.pose.pose.27.face',(1,72)),
 'mhr_model_parameters':('layer.5.pose.pose.90.model_params',(1,204)),
 'hand_boxes':('branch.hand_box',(1,2,4)),
 'hand_logits':('branch.hand_logits',(1,2,2)),
 'joint_transforms':('layer.5.pose.mhr.skeleton',(1,127,8)),
 'faces':('head_pose.faces',(36874,3))}

def read_output(path):
    result={}
    with path.open('rb') as f:
        def read(n):
            data=f.read(n)
            if len(data)!=n:raise ValueError('truncated API output')
            return data
        def unpack(fmt):return struct.unpack(fmt,read(struct.calcsize(fmt)))
        if read(8)!=b'S3DOUT01' or unpack('<I')[0]!=len(FIELDS):raise ValueError('invalid API header/count')
        for _ in range(len(FIELDS)):
            length=unpack('<I')[0]
            if not 1<=length<=64:raise ValueError('invalid tensor name length')
            name=read(length).decode('ascii')
            if name not in FIELDS or name in result:raise ValueError('unexpected/duplicate result')
            dtype,rank,elements=unpack('<IIQ');shape=FIELDS[name][1]
            if rank!=len(shape) or elements!=int(np.prod(shape)) or dtype!=(2 if name=='faces' else 1):raise ValueError('invalid API descriptor')
            if unpack('<'+'Q'*rank)!=shape:raise ValueError('invalid API shape')
            value=np.frombuffer(read(elements*4),dtype='<i4' if dtype==2 else '<f4').reshape(shape)
            if not np.isfinite(value).all():raise ValueError('nonfinite API output')
            result[name]=value
        if f.read(1):raise ValueError('trailing API data')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['input','accepted','safe-state','report']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();acceptance=json.loads((a.accepted/'parity-trained-v1.json').read_text())
    if not acceptance['pass'] or len(acceptance['checks'])!=646 or not all(v['pass'] for v in acceptance['checks']):raise ValueError('requires complete accepted 646-check trained run')
    candidate=a.accepted/'native.safetensors'
    if sha256_file(candidate)!=acceptance['native_safetensors_sha256']:raise ValueError('accepted output changed')
    if sha256_file(a.safe_state)!='4c6b3f63ce8a050f6587cf833a036bad3f68377d86cfe69d591502c1373ba0a3':raise ValueError('unverified topology source')
    actual=read_output(a.input);checks=[]
    with safe_open(candidate,framework='numpy') as native,safe_open(a.safe_state,framework='numpy') as state:
        for name,(source,shape) in FIELDS.items():
            expected=state.get_tensor(source).astype('<i4') if name=='faces' else native.get_tensor('case.0000.'+source)
            passed=actual[name].shape==expected.shape and actual[name].tobytes()==expected.tobytes()
            checks.append(dict(name=name,pass_exact=passed,shape=list(shape)))
    report=dict(scope=__doc__,checks=checks,passed=all(v['pass_exact'] for v in checks),
                input_sha256=sha256_file(a.input),accepted_report_sha256=sha256_file(a.accepted/'parity-trained-v1.json'),script_sha256=sha256_file(Path(__file__)))
    with a.report.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
