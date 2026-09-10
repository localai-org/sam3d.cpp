#!/usr/bin/env python3
"""Unchanged original left-hand output conversion; isolated component, not merge."""
import argparse, ast, hashlib, json, struct, types
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file
from capture_mhr import digest
from capture_body_condition import BODY_SHA256
from trained_body_state import STATE_SHA


def pattern(count):
    return (((np.arange(count, dtype=np.int64) * 7) % 41 - 20) / 16).astype(np.float32)


def fnv(values):
    result = 14695981039346656037
    for byte in values.astype('<f4').tobytes(): result = ((result ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['upstream', 'extraction', 'left-reference', 'output']: p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    a = p.parse_args()
    source = a.upstream / 'sam_3d_body/models/meta_arch/sam3d_body.py'
    if digest(source) != BODY_SHA256: raise ValueError('unverified original source')
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'SAM3DBody')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run_inference')
    def assignment(node, name):
        return isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name
    start = next(i for i,n in enumerate(method.body) if assignment(n, 'scale_r_hands_mean'))
    end = next(i for i,n in enumerate(method.body) if assignment(n, 'batch_rhand'))
    statements = method.body[start:end]
    if len(statements) != 9: raise ValueError('unexpected original unmirror statements')
    code = compile(ast.Module(body=statements, type_ignores=[]), str(source), 'exec')
    weights = a.extraction / 'body-other-state.safetensors'
    if digest(weights) != STATE_SHA: raise ValueError('unverified shared body scale state')
    with safe_open(weights, framework='numpy') as safe:
        mean = safe.get_tensor('head_pose.scale_mean').copy()
        components = safe.get_tensor('head_pose.scale_comps').copy()
    manifest = json.loads((a.left_reference / 'manifest.json').read_text())
    if not manifest.get('hand_branch') or manifest['cases'][0]['image_pipeline']['hand_side'] != 'left': raise ValueError('requires original left hand image capture')
    for name in ['full.safetensors', 'upstream.safetensors']:
        if digest(a.left_reference / name) != manifest['artifacts'][name]: raise ValueError('original input capture changed')
    with safe_open(a.left_reference / 'full.safetensors', framework='numpy') as safe:
        learned = {name: safe.get_tensor('case.0000.layer.5.' + name).copy() for name in ['scale', 'joint_global_rots', 'hand']}
    with safe_open(a.left_reference / 'upstream.safetensors', framework='numpy') as safe:
        learned['bbox_center'] = safe.get_tensor('case.0000.prepare.box_center').reshape(1, 1, 2).copy()
        width = int(safe.get_tensor('case.0000.prepare.image_size').reshape(-1)[0])
    a.output.mkdir(parents=True, exist_ok=True)
    if any(a.output.iterdir()): raise FileExistsError('output must be empty')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    descriptors, tensors, rules = [], {}, []
    regression = ['S3D_HAND_UNMIRROR_V1 2']
    for i, (b, w) in enumerate([(1, 997), (2, 1920), (1, width)]):
        if i < 2:
            m = np.zeros(68, np.float32); m[8] = .125; m[9] = -.375
            c = np.zeros((28, 68), np.float32); c[8, 8] = .25; c[9, 9] = .5
            data = dict(scale=pattern(b*28).reshape(b,28), joint_global_rots=pattern(b*127*9).reshape(b,127,3,3),
                        hand=pattern(b*108).reshape(b,108), bbox_center=pattern(b*2).reshape(b,1,2))
        else: m, c, data = mean, components, learned
        holder = types.SimpleNamespace(head_pose=types.SimpleNamespace(scale_mean=torch.from_numpy(m), scale_comps=torch.from_numpy(c)))
        def run():
            values = {key: torch.from_numpy(value.copy()).to(a.device) for key,value in data.items()}
            local = dict(self=holder, width=w, lhand_output={'mhr_hand':{key:values[key] for key in ['scale','joint_global_rots','hand']}}, batch_lhand={'bbox_center':values['bbox_center']})
            with torch.no_grad(): exec(code, {}, local)
            return {key:value.cpu().numpy().copy() for key,value in values.items()}
        first, second = run(), run()
        if any(first[key].tobytes() != second[key].tobytes() for key in first): raise ValueError('nonrepeatable original unmirror')
        prefix = f'case.{i:04d}'; filename = prefix + '.input'
        with (a.output / filename).open('xb') as out:
            out.write(b'S3DHUF01' + struct.pack('<2I', b, w))
            for value in [m, c, data['scale'], data['joint_global_rots'], data['hand'], data['bbox_center']]: out.write(value.astype('<f4').tobytes())
        for key,value in first.items():
            tensors[prefix + '.' + key] = value
            rules.append(dict(name=prefix + '.' + key, mode='exact'))
        descriptors.append(dict(prefix=prefix, input=filename, order=sorted(first), shapes={key:list(value.shape) for key,value in first.items()}, origin='synthetic inputs/state' if i<2 else 'fixed original learned left-hand output and real shared body scale state'))
        if i < 2:
            regression.append(f'{b} {w}')
            regression.extend(f'{key} {fnv(first[key])}' for key in sorted(first))
    save_file(tensors, a.output / 'upstream.safetensors')
    (a.output / 'rules.json').write_text(json.dumps(dict(schema_version=1, boundary=__doc__, tensors=rules), indent=2) + '\n')
    (a.output / 'hand-unmirror.txt').write_text('\n'.join(regression) + '\n')
    record = dict(scope=__doc__, source_sha256=BODY_SHA256, script_sha256=digest(Path(__file__)), state_sha256=STATE_SHA,
                  left_reference_manifest_sha256=digest(a.left_reference / 'manifest.json'), device=a.device, torch=torch.__version__,
                  oracle='nine unchanged original run_inference statements; original arithmetic and indexing, not a rewritten oracle',
                  repeated='all output bytes exact', cases=descriptors,
                  artifacts={path.name:digest(path) for path in a.output.iterdir() if path.is_file()})
    (a.output / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(cases=len(descriptors), exact_tensors=len(tensors))), flush=True)


if __name__ == '__main__': main()
