#!/usr/bin/env python3
"""Prepare a small, labeled ORIGINAL Body-branch comparison for the demo.

No inference and no candidate tensors: only verified original captures. The PNG
preserves the official JPEG's captured decoded RGB exactly, avoiding independent
JPEG decoders' rounding differences. It is an optional example, not runtime state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np
from safetensors import safe_open
from check_parity import sha256_file
from run_body_model import extract_image
from check_body_api import FIELDS

ORIGINAL = {
 'vertices':'pred_vertices', 'joints':'pred_joint_coords',
 'joint_rotations':'joint_global_rots', 'keypoints':'pred_keypoints_3d',
 'keypoints_pixels':'pred_keypoints_2d', 'vertices_pixels':'pred_keypoints_2d_verts',
 'camera_translation':'pred_cam_t', 'camera_parameters':'pred_cam',
 'pose_raw':'pred_pose_raw', 'global_rotation':'global_rot',
 'body_pose':'body_pose', 'shape':'shape', 'scale':'scale', 'hand':'hand',
 'face':'face', 'mhr_model_parameters':'mhr_model_params',
 'hand_boxes':'hand_box', 'hand_logits':'hand_logits',
}
def png_rgb(w,h,rgb):
    if len(rgb)!=w*h*3: raise ValueError('RGB size mismatch')
    def chunk(name,data):
        return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data))
    scan=b''.join(b'\0'+rgb[y*w*3:(y+1)*w*3] for y in range(h))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(scan))+chunk(b'IEND',b'')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['reference','safe-state','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();m=json.loads((a.reference/'manifest.json').read_text())
    if not m.get('learned_sam_checkpoint_loaded') or m.get('device')!='cuda' or len(m['cases'])!=1:raise ValueError('requires real original CUDA Body image branch')
    case=m['cases'][0];image=case['image_pipeline']
    if image['photo_sha256']!='0112b0a32ea5860db6a6fc700804751528341a02bf07fed0329a0c2610f905d3':raise ValueError('unexpected official image')
    for name in ['case.0000.input','full.safetensors']:
        if sha256_file(a.reference/name)!=m['artifacts'][name]:raise ValueError('original artifact hash mismatch')
    if sha256_file(a.safe_state)!='4c6b3f63ce8a050f6587cf833a036bad3f68377d86cfe69d591502c1373ba0a3':raise ValueError('unverified topology')
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise ValueError('output must be empty')
    extract_image(a.reference/'case.0000.input',a.output/'image.input')
    packed=(a.output/'image.input').read_bytes();w,h,stride=struct.unpack_from('<3I',packed,8)
    box=struct.unpack_from('<4f',packed,20);camera=struct.unpack_from('<4f',packed,36)
    if stride!=w*3 or list(box)!=image['bbox_xyxy'] or list(camera)!=image['intrinsics_fx_fy_cx_cy']:raise ValueError('original image settings mismatch')
    (a.output/'input.png').write_bytes(png_rgb(w,h,packed[52:]))
    result=dict(schema='sam3d.body.pose_branch.v1',coordinates='body metres; x right, y down, z forward; camera translation not applied',tensors={})
    with safe_open(a.reference/'full.safetensors',framework='numpy') as original:
        for name,source in ORIGINAL.items():
            v=original.get_tensor('case.0000.layer.5.'+source)
            if tuple(v.shape)!=FIELDS[name][1] or v.dtype!=np.float32 or not np.isfinite(v).all():raise ValueError('unexpected original '+name)
            result['tensors'][name]=v.reshape(-1).tolist()
    with safe_open(a.safe_state,framework='numpy') as state:
        faces=state.get_tensor('head_pose.faces')
        if faces.shape!=(36874,3) or np.any(faces<0) or np.any(faces>=18439):raise ValueError('invalid topology')
        result['faces']=faces.reshape(-1).tolist()
    (a.output/'result.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    manifest=dict(scope='ORIGINAL PyTorch F32 body pose branch only; not full hand-refined estimator or published gallery screenshot',
        original_source='facebookresearch/sam-3d-body',revision='b5c765a0d89d789985e186d396315e7590887b94',
        original_manifest_sha256=sha256_file(a.reference/'manifest.json'),original_full_sha256=sha256_file(a.reference/'full.safetensors'),
        original_photo_sha256=image['photo_sha256'],native_input_sha256=hashlib.sha256(packed).hexdigest(),
        settings=dict(box=box,camera=camera),width=w,height=h,
        image_note='Lossless PNG of original captured decoded RGB; same pixels, no resizing or geometric edits. JPEG decoder differences are excluded.',
        artifacts={name:sha256_file(a.output/name) for name in ['input.png','result.json','image.input']},script_sha256=sha256_file(Path(__file__)))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
