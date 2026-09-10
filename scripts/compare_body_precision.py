#!/usr/bin/env python3
"""Report native/public versus original final fields; not layer-parity acceptance."""
import argparse,json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from check_body_api import read_output
from check_parity import sha256_file

FIELDS={'vertices':'pred_vertices','joints':'pred_joint_coords','joint_rotations':'joint_global_rots',
    'keypoints':'pred_keypoints_3d','keypoints_pixels':'pred_keypoints_2d','vertices_pixels':'pred_keypoints_2d_verts',
    'camera_translation':'pred_cam_t','camera_parameters':'pred_cam','pose_raw':'pred_pose_raw',
    'global_rotation':'global_rot','body_pose':'body_pose','shape':'shape','scale':'scale','hand':'hand',
    'face':'face','mhr_model_parameters':'mhr_model_params','hand_boxes':'hand_box','hand_logits':'hand_logits'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['native','reference','report']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();native=read_output(a.native)
    if set(native)!=set(FIELDS)|{'faces'}:raise ValueError('native public result field coverage changed')
    checks=[]
    with safe_open(a.reference,framework='np') as f:
        for name,key in FIELDS.items():
            r=f.get_tensor(key);v=native[name]
            if r.shape!=v.shape or not np.isfinite(r).all() or not np.isfinite(v).all():raise ValueError('shape/finiteness mismatch: '+name)
            r=r.astype(np.float64);d=v.astype(np.float64)-r
            row=dict(name=name,max_abs=float(np.abs(d).max()),relative_l2=float(np.linalg.norm(d)/max(np.linalg.norm(r),1e-12)))
            if name in ['vertices','joints','keypoints']:row['mean_euclidean_metres']=float(np.linalg.norm(d,axis=-1).mean())
            checks.append(row)
    report=dict(scope=__doc__,native_sha256=sha256_file(a.native),reference_sha256=sha256_file(a.reference),
        script_sha256=sha256_file(Path(__file__)),fields=checks,acceptance='diagnostic only; no thresholds or model parity claim')
    with a.report.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
