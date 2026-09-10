#!/usr/bin/env python3
"""Recheck raw-RGB trained Body captures; reuse only an identical backbone policy.

No tolerance calibration occurs here. Pixel/geometry/head limits are unchanged.
The separately frozen backbone limits require byte-identical original tensors.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from safetensors import safe_open
from check_parity import compare_array,sha256_file,validate_rules
from calibrate_trained_dino import reference as backbone_reference
from rotation_parity import compare_euler

POLICY_SHA='cdc4bd14042edb87e43c9f0655e469e572874856c29a5776cec3d6c9f4364299'
FULL_NAMES={'pred_pose_raw':'pose.pred_pose_raw','global_rot':'pose.global_rot','body_pose':'pose.body_pose',
 'shape':'pose.pose.24.shape','scale':'pose.pose.25.scale','hand':'pose.pose.26.hand','face':'pose.pose.27.face',
 'pred_keypoints_3d':'pose.map.92.keypoints','pred_vertices':'pose.map.90.vertices','pred_joint_coords':'pose.map.91.joints',
 'joint_global_rots':'pose.map.03.joint_rotations','mhr_model_params':'pose.pose.90.model_params',
 'pred_cam':'camera.10.pred_cam','pred_cam_t':'camera.15.translation','focal_length':'camera.13.focal',
 'pred_keypoints_2d':'camera.20.pixels','pred_keypoints_2d_depth':'camera.17.depth','pred_keypoints_2d_verts':'03.vertex_pixels','pred_keypoints_2d_cropped':'04.crop_points'}
DECODER_NAMES=['00.token_pe','01.context_pe','02.ln1','20.self_residual','21.ln2_1','22.ln2_2','40.cross_residual',
               '41.ln3','42.ffn_linear1','43.ffn_gelu','44.ffn_linear2','45.ffn_residual','90.tokens','91.context']
DECODER_NAMES += [stage+'.'+key for stage in ['10.self','30.cross'] for key in
                 ['00.q_input','01.k_input','02.v_input','03.q','04.k','05.v','06.logits','07.probs','08.attended','09.output']]

def validate_operation_coverage(case,enabled):
    expected={f'layer.{i}.decoder.{name}' for i in range(6) for name in DECODER_NAMES} if enabled else set()
    actual={name for name in case['order'] if '.decoder.' in name}
    if actual!=expected or len(case['order'])!=(529 if enabled else 325):raise ValueError('incomplete trained operation coverage')

def same_bytes(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def boundary_rule(name,backbone_rules):
    short=name.removeprefix('case.0000.backbone.')
    if name.startswith('case.0000.backbone.') and short in backbone_rules:
        return dict(backbone_rules[short],name=name)
    pixel=any(name.endswith(x) for x in ['03.vertex_pixels','12.scaled_box','19.intrinsic_projection','20.pixels','pred_keypoints_2d','pred_keypoints_2d_verts'])
    return dict(name=name,mode='float',max_abs=1e-3 if pixel else 1e-4,relative_l2=2e-5,zero_reference_floor=1e-12)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','candidate','backbone-reference','report']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();manifest=json.loads((a.reference/'manifest.json').read_text())
    if manifest.get('learned_sam_checkpoint_loaded') is not True or manifest.get('sdpa_backend')!='MATH' or manifest.get('tf32') is not False:raise ValueError('requires trained F32 math reference')
    state=manifest['trained_state']
    if state['state_sha256']!='4c6b3f63ce8a050f6587cf833a036bad3f68377d86cfe69d591502c1373ba0a3' or state['config_sha256']!='1012fc3f39cb5e90e3f8fbadf7bded31604bfafdce0321d17a7c1a2d3f08b88d' or state['missing_initialized_tensors']:raise ValueError('unverified learned state/config')
    if len(manifest['cases'])!=1:raise ValueError('requires one complete trained case')
    case=manifest['cases'][0]
    validate_operation_coverage(case,manifest.get('decoder_operations',False))
    if case['prefix']!='case.0000' or case['depth']!=6 or any(e['max_abs']!=0 for e in case['unobserved_vs_observed'].values()):raise ValueError('incomplete or nonneutral original capture')
    for file in ['upstream.safetensors','full.safetensors',case['input']]:
        if Path(file).name!=file or sha256_file(a.reference/file)!=manifest['artifacts'][file]:raise ValueError('reference artifact mismatch')
    policy_path=Path(__file__).resolve().parents[1]/'reference/trained-dino-backbone-policy-v1.json'
    if sha256_file(policy_path)!=POLICY_SHA:raise ValueError('backbone policy changed')
    policy=json.loads(policy_path.read_text());rules={r['name'].removeprefix('case.0000.'):r for r in validate_rules(policy)}
    control,control_path=backbone_reference(a.backbone_reference,manifest['device'])
    if sha256_file(a.backbone_reference/'manifest.json')!=policy[manifest['device']+'_manifest_sha256']:raise ValueError('not the frozen original backbone cohort')
    run=json.loads((a.candidate/'report.json').read_text())
    if run['reference_manifest_sha256']!=sha256_file(a.reference/'manifest.json') or run['gguf_sha256']!='d52ab772628fb6d851550381b428185a32da6398793f0ff70299bad1999ba8f8' or run['backbone_gguf_sha256']!='9228c12b5b34cdb3627fce731a3e3d5890c54dc78ca7eb357d66056893f5bf1f':raise ValueError('native capture identity mismatch')
    checks=[]
    with safe_open(a.reference/'upstream.safetensors',framework='numpy') as ref,safe_open(a.candidate/'native.safetensors',framework='numpy') as native,safe_open(control_path,framework='numpy') as standalone,safe_open(a.reference/'full.safetensors',framework='numpy') as full:
        names=['case.0000.'+key for key in case['order']]
        if set(names)!=set(ref.keys()) or set(names)!=set(native.keys()):raise ValueError('incomplete original/native boundary set')
        for key in rules:
            if not same_bytes(ref.get_tensor('case.0000.backbone.'+key),standalone.get_tensor('case.0000.'+key)):raise ValueError('cannot reuse policy: original backbone boundary changed '+key)
        def check(name,reference,candidate):
            if name.endswith(('.pose.global_rot','.full.global_rot')):
                checks.append(dict(name=name,**compare_euler(reference,candidate)))
            else:checks.append(compare_array(reference,candidate,boundary_rule(name,rules)))
        for name in names:check(name,ref.get_tensor(name),native.get_tensor(name))
        tested=set()
        def final(key,native_key):
            tested.add(key);prefix,field=key.rsplit('.',1);check(prefix+'.full.'+field,full.get_tensor(key),native.get_tensor(native_key))
        final('case.0000.tokens','case.0000.90.output_tokens')
        for i in range(6):
            for key,native_key in FULL_NAMES.items():final(f'case.0000.layer.{i}.'+key,f'case.0000.layer.{i}.'+native_key)
        for key in ['hand_box','hand_logits']:final('case.0000.layer.5.'+key,'case.0000.branch.'+key)
        if tested!=set(full.keys()):raise ValueError('untested original final fields')
    report=dict(scope=manifest['scope'],pass_=all(c['pass'] for c in checks),checks=checks,
                backbone_policy_sha256=POLICY_SHA,original_backbone_policy_reuse='all 36 original tensors byte-identical to frozen cohort',
                reference_manifest_sha256=sha256_file(a.reference/'manifest.json'),native_capture_report_sha256=sha256_file(a.candidate/'report.json'),
                native_safetensors_sha256=sha256_file(a.candidate/'native.safetensors'),script_sha256=sha256_file(Path(__file__)))
    report['pass']=report.pop('pass_')
    with a.report.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=report['pass'],checks=len(checks),failures=[c for c in checks if not c['pass']][:8]),indent=2))
    if not report['pass']:raise SystemExit(1)

if __name__=='__main__':main()
