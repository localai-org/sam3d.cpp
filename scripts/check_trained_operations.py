#!/usr/bin/env python3
"""Bounded-memory comparison of all trained DINO operations in one backbone.

Optional policy generation consumes original CPU/CUDA traces only. Probability
budgets have their own absolute ceiling because softmax can amplify logit
rounding; geometric/image/other model tolerances are never affected.
"""
import argparse
import json
import math
import os
from pathlib import Path

# These are small chunked diagnostic reductions, not model inference. Avoid
# spawning a full BLAS thread team for every reduction unless explicitly asked.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from safetensors import safe_open
from check_parity import compare_array, sha256_file
from calibrate_trained_dino import limits, reference

NAMES = ['00.rope_sin','01.rope_cos','02.norm1','03.qkv','04.q','05.k','06.v',
         '07.q_rope','08.k_rope','09.logits','10.probs','11.attention','12.attn_proj',
         '13.ls1','14.residual1','15.norm2','16.w1','17.w2','18.hidden','19.w3','20.ls2','21.output']


def original(directory):
    parent = json.loads((directory/'manifest.json').read_text())
    parent, _ = reference(directory, parent.get('device'))
    if parent['device'] not in ['cpu','cuda']:
        raise ValueError('not an original CPU/CUDA reference')
    manifest_path = directory/'operations/manifest.json'
    if sha256_file(manifest_path) != parent['operation_manifest_sha256']:
        raise ValueError('operation manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('logits_probabilities') != 'actual math SDPA dispatch, not auxiliary formulas':
        raise ValueError('requires actual math SDPA operation capture')
    records = manifest['blocks']
    if len(records) != 32 or [r['index'] for r in records] != list(range(32)):
        raise ValueError('incomplete or unordered blocks')
    for record in records:
        if record['order'] != NAMES or set(record['shapes']) != set(NAMES):
            raise ValueError('incomplete operation set')
        if record['file'] != f"block.{record['index']}.safetensors":
            raise ValueError('invalid operation path')
        if sha256_file(directory/'operations'/record['file']) != record['sha256']:
            raise ValueError('operation bytes hash mismatch')
    return parent, records


def native_views(path, record):
    shapes = [record['shapes'][name] for name in NAMES]
    if any(not shape or len(shape)>4 or any(type(n) is not int or n<=0 for n in shape) or
           math.prod(shape)>32*1024**2 for shape in shapes):
        raise ValueError('invalid capture shape')
    counts = [math.prod(shape) for shape in shapes]
    if path.stat().st_size != sum(counts)*4:
        raise ValueError('native operation file length mismatch')
    mapped = np.memmap(path, mode='r', dtype='<f4')
    cursor, result = 0, {}
    for name, shape, count in zip(NAMES, shapes, counts, strict=True):
        result[name] = mapped[cursor:cursor+count].reshape(shape)
        cursor += count
    return result


def calibrated_rule(control, peak, name):
    if 'max_abs' not in control or 'relative_l2' not in control:
        raise ValueError('invalid original operation control')
    rule = dict(name=name, mode='float', **limits(control | {'pass':True}))
    # Softmax probabilities are unitless bounded values with a different
    # conditioning profile from generic activations. Still require both errors,
    # keep the 1e-3 relative cap, and reject budgets exceeding 1 percentage point.
    cap = .01 if name.endswith('.10.probs') else max(1e-4, peak*1e-4)
    if control['max_abs'] > cap:
        raise ValueError('original absolute discrepancy exceeds operation ceiling')
    rule['max_abs'] = min(rule['max_abs'], cap)
    return rule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['reference','candidate','report']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--candidate-reference', action='store_true')
    parser.add_argument('--rules', type=Path)
    parser.add_argument('--policy-output', type=Path)
    args = parser.parse_args()
    if args.policy_output and (not args.candidate_reference or args.rules):
        raise ValueError('calibrate only from original CPU/CUDA without candidate-specific rules')
    parent, records = original(args.reference)
    peer_records = None
    if args.candidate_reference:
        peer, peer_records = original(args.candidate)
        if parent['device'] == peer['device'] or parent['cases'] != peer['cases'] or parent['source_hashes'] != peer['source_hashes']:
            raise ValueError('original CPU/CUDA control mismatch')
        if parent['artifacts']['case.0000.input'] != peer['artifacts']['case.0000.input']:
            raise ValueError('different original image/state input')
    else:
        run = json.loads((args.candidate/'run.json').read_text())
        if run.get('gguf_sha256') != '9228c12b5b34cdb3627fce731a3e3d5890c54dc78ca7eb357d66056893f5bf1f' or \
                len(run['commands']) != 1 or run['commands'][0]['returncode'] != 0:
            raise ValueError('requires completed native trained GGUF run')
    expected_names = [f'block.{r["index"]}.{name}' for r in records for name in NAMES]
    rules = None
    if args.rules:
        policy = json.loads(args.rules.read_text())
        if policy.get('scope') != 'all 32 trained DINO blocks; actual math SDPA; original-control policy v1' or \
                policy.get('native_candidate_used') is not False:
            raise ValueError('wrong operation policy')
        if [r['name'] for r in policy['tensors']] != expected_names:
            raise ValueError('policy coverage mismatch')
        if sha256_file(args.reference/'manifest.json') not in [policy['reference_manifest_sha256'],policy['candidate_manifest_sha256']]:
            raise ValueError('policy was not frozen for this original reference cohort')
        rules = {r['name']:r for r in policy['tensors']}
    results, generated, errors, blocks = [], [], [], []
    for record in records:
        index = record['index']
        with safe_open(args.reference/'operations'/record['file'], framework='numpy') as upstream:
            if set(upstream.keys()) != set(NAMES):
                raise ValueError('operation tensor set mismatch')
            native = None
            candidate_context = None
            try:
                if peer_records is not None:
                    other = peer_records[index]
                    if record['shapes'] != other['shapes'] or record['order'] != other['order']:
                        raise ValueError('original operation layout mismatch')
                    candidate_context = safe_open(args.candidate/'operations'/other['file'], framework='numpy')
                    candidate_context.__enter__()
                else:
                    native = native_views(args.candidate/'case.0000.blocks'/f'block.{index}.bin', record)
                start = len(results)
                for key in NAMES:
                    name = f'block.{index}.{key}'
                    ref = upstream.get_tensor(key)
                    candidate = candidate_context.get_tensor(key) if candidate_context is not None else native[key]
                    if list(ref.shape) != record['shapes'][key]:
                        raise ValueError('reference layout mismatch')
                    rule = rules[name] if rules else dict(name=name, mode='float', max_abs=1e-4, relative_l2=2e-5, zero_reference_floor=1e-12)
                    result = compare_array(ref, candidate, rule)
                    results.append(result)
                    if args.policy_output:
                        try:
                            generated.append(calibrated_rule(result, float(np.abs(ref).max()), name))
                        except ValueError as exc:
                            errors.append(dict(name=name, error=str(exc)))
                print(json.dumps(dict(block=index, checks=len(results)-start,
                                      failures=sum(not r['pass'] for r in results[start:]))), flush=True)
                blocks.append(dict(index=index, reference_sha256=record['sha256'],
                    candidate_sha256=peer_records[index]['sha256'] if peer_records is not None else
                    sha256_file(args.candidate/'case.0000.blocks'/f'block.{index}.bin')))
            finally:
                if candidate_context is not None:
                    candidate_context.__exit__(None, None, None)
                del native
    policy_data = dict(schema_version=1, scope='all 32 trained DINO blocks; actual math SDPA; original-control policy v1',
        native_candidate_used=False, reference_manifest_sha256=sha256_file(args.reference/'manifest.json'),
        candidate_manifest_sha256=sha256_file(args.candidate/'manifest.json') if args.candidate_reference else None,
        formula='max(legacy floor, 4 * original CPU/CUDA error), headroom clipped at fixed absolute ceiling; reject original errors above ceiling; rel cap 1e-3; abs cap peak*1e-4 except probabilities capped at .01',
        tensors=generated, controls=results, artifacts=blocks, script_sha256=sha256_file(Path(__file__)))
    if args.policy_output and not errors:
        with args.policy_output.open('x') as stream:
            json.dump(policy_data, stream, indent=2); stream.write('\n')
    report = dict(scope='full trained backbone own-intermediate operation comparison',
                  pass_=all(r['pass'] for r in results), checks=results, artifacts=blocks,
                  policy_errors=errors, rules_sha256=sha256_file(args.rules) if args.rules else None,
                  reference_manifest_sha256=sha256_file(args.reference/'manifest.json'),
                  candidate_run_sha256=sha256_file(args.candidate/('manifest.json' if args.candidate_reference else 'run.json')),
                  script_sha256=sha256_file(Path(__file__)))
    report['pass'] = report.pop('pass_')
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(checks=len(results), passed=report['pass'], calibration_errors=errors)), flush=True)
    return 0 if (not errors if args.policy_output else report['pass']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
