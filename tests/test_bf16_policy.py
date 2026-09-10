import hashlib,json,sys,unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_bf16_backbone import limits
from check_parity import compare_array
from calibrate_bf16_body import field_limits,HAND_FIELDS
from compare_body_precision import FIELDS
from check_bf16_body import check
from safetensors.numpy import save_file

class BF16PolicyTests(unittest.TestCase):
    def test_frozen_rider_backbone_bounded_policy(self):
        p=Path(__file__).resolve().parents[1]/'reference/bf16-backbone-rider-policy-v2.json'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'33f485de2361ff8521ca7105d8bc8edc86299b20599cea21e34300993cd887f0')
        policy=json.loads(p.read_text());self.assertFalse(policy['native_candidate_used'])
        self.assertEqual(len(policy['tensors']),36)
        self.assertIn('4df2c680c6aafcc3fd510382473961f4e0faceb3c5d6a15fe54835bfa44d6253',policy['boundary'])
        for rule,record in zip(policy['tensors'],policy['controls'],strict=True):
            self.assertEqual(rule['name'],record['name'])
            for k,v in limits(record['controls'],record['reference_peak'],clip_headroom=True).items():self.assertEqual(rule[k],v)
            self.assertLessEqual(rule['max_abs'],max(1e-4,.1*record['reference_peak']))
            self.assertLessEqual(rule['relative_l2'],.05)
            for control in record['controls']:
                self.assertLessEqual(control['max_abs'],rule['max_abs'])
                self.assertLessEqual(control['relative_l2'],rule['relative_l2'])
            reference=np.linspace(-record['reference_peak'],record['reference_peak'],4096,dtype=np.float32)
            self.assertTrue(compare_array(reference,reference.copy(),rule)['pass'])
            for bad in [np.full_like(reference,np.nan),reference[::-1].copy(),np.zeros_like(reference),reference+2*rule['max_abs']]:
                self.assertFalse(compare_array(reference,bad,rule)['pass'])

    def test_bounded_headroom_cannot_raise_caps_or_exclude_originals(self):
        c=dict(reference_dtype='float32',max_abs=.06,relative_l2=.03)
        c['pass']=True
        # The legacy policy is unchanged: a 2x reservation over its cap fails.
        with self.assertRaises(ValueError):limits([c],1)
        bounded=limits([c],1,clip_headroom=True)
        self.assertEqual(bounded,dict(max_abs=.1,relative_l2=.05,zero_reference_floor=1e-12))
        self.assertLessEqual(c['max_abs'],bounded['max_abs'])
        self.assertLessEqual(c['relative_l2'],bounded['relative_l2'])
        for bad in [dict(c,max_abs=.100001),dict(c,relative_l2=.050001),dict(c,max_abs=float('nan')),dict(c,relative_l2=-1)]:
            with self.assertRaises(ValueError):limits([bad],1,clip_headroom=True)
        stable=dict(c,max_abs=.001,relative_l2=.0001)
        self.assertEqual(limits([stable],1),limits([stable],1,clip_headroom=True))

    def test_frozen_rider_original_only_policy(self):
        p=Path(__file__).resolve().parents[1]/'reference/bf16-body-rider-policy-v1.json'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'4dba2bc8fcde92da6a60251e62c15b0f01ae69e9f6502fb6721e1dadf715c1e8')
        policy=json.loads(p.read_text());self.assertFalse(policy['native_candidate_used'])
        self.assertEqual(policy['source']['photo_sha256'],'4df2c680c6aafcc3fd510382473961f4e0faceb3c5d6a15fe54835bfa44d6253')
        self.assertEqual({r['name'] for r in policy['tensors']},set(FIELDS))
        for rule,record in zip(policy['tensors'],policy['controls'],strict=True):
            self.assertEqual(rule['name'],record['name'])
            for k,v in field_limits(rule['name'],record['controls'],record['reference_peak']).items():self.assertEqual(rule[k],v)
            reference=np.linspace(-record['reference_peak'],record['reference_peak'],4096,dtype=np.float32)
            self.assertTrue(compare_array(reference,reference.copy(),rule)['pass'])
            for bad in [np.full_like(reference,np.nan),reference+max(.1*record['reference_peak'],.01)]:
                self.assertFalse(compare_array(reference,bad,rule)['pass'])

    def test_frozen_isolated_attention_and_negative_controls(self):
        from check_bf16_attention import POLICY_SHA
        p=Path(__file__).resolve().parents[1]/'reference/bf16-attention-policy-v1.json'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),POLICY_SHA)
        policy=json.loads(p.read_text());self.assertFalse(policy['native_candidate_used'])
        self.assertEqual([r['name'] for r in policy['tensors']],[f'block.{i:02d}' for i in range(32)])
        for rule,record in zip(policy['tensors'],policy['records'],strict=True):
            self.assertEqual(rule['name'],record['name'])
            self.assertEqual({c['mode'] for c in record['controls']},{'cpu_math','cuda_flash','cuda_efficient'})
            self.assertEqual(rule['max_abs'],max(1e-4,2*max(c['max_abs'] for c in record['controls'])))
            self.assertEqual(rule['relative_l2'],max(2e-5,2*max(c['relative_l2'] for c in record['controls'])))
            reference=np.linspace(-1,1,4096,dtype=np.float32)
            self.assertTrue(compare_array(reference,reference.copy(),rule)['pass'])
            for bad in [reference*1.1,reference[::-1].copy(),reference+.2,np.zeros_like(reference),np.full_like(reference,np.nan)]:
                self.assertFalse(compare_array(reference,bad,rule)['pass'])

    def test_frozen_reference_only_policy_and_negative_controls(self):
        p=Path(__file__).resolve().parents[1]/'reference/bf16-backbone-policy-v1.json'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'b9f28b6cabd2e0e5da50a6937b89b3ca9e5d503aa5aff39c021797db8e4cc176')
        policy=json.loads(p.read_text());self.assertFalse(policy['native_candidate_used']);self.assertEqual(len(policy['tensors']),36)
        for rule,record in zip(policy['tensors'],policy['controls'],strict=True):
            self.assertEqual(rule['name'],record['name'])
            for k,v in limits(record['controls'],record['reference_peak']).items():self.assertEqual(rule[k],v)
            reference=np.linspace(-record['reference_peak'],record['reference_peak'],4096,dtype=np.float32)
            self.assertTrue(compare_array(reference,reference.copy(),rule)['pass'])
            for bad in [reference*1.1,reference[::-1].copy(),np.zeros_like(reference),reference+.1*record['reference_peak'],np.full_like(reference,np.nan)]:
                self.assertFalse(compare_array(reference,bad,rule)['pass'])
            spike=reference.copy();spike[0]+=2*rule['max_abs']
            self.assertFalse(compare_array(reference,spike,rule)['pass'])
    def test_caps_and_invalid_calibration(self):
        c=dict(pass_=True,reference_dtype='float32',max_abs=0,relative_l2=0);c['pass']=True
        self.assertEqual(limits([c],1),dict(max_abs=1e-4,relative_l2=2e-5,zero_reference_floor=1e-12))
        for bad in [dict(c,relative_l2=.1),dict(c,max_abs=.2),dict(c,max_abs=float('nan')),dict(c,relative_l2=-1)]:
            with self.assertRaises(ValueError):limits([bad],1)
        with self.assertRaises(ValueError):limits([],1)
        with self.assertRaises(ValueError):limits([c],0)

    def test_final_body_frozen_limits_and_negative_controls(self):
        p=Path(__file__).resolve().parents[1]/'reference/bf16-body-policy-v1.json'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'7b57d6083c51ce0ff82d5d13157da241c8d1d40c4610c4ff8a66d4a1a9902249')
        policy=json.loads(p.read_text());self.assertFalse(policy['native_candidate_used'])
        self.assertEqual({r['name'] for r in policy['tensors']},set(FIELDS))
        for rule,record in zip(policy['tensors'],policy['controls'],strict=True):
            name=rule['name'];self.assertEqual(name,record['name']);self.assertEqual(rule['blocking'],name not in HAND_FIELDS)
            for k,v in field_limits(name,record['controls'],record['reference_peak']).items():self.assertEqual(rule[k],v)
            reference=np.linspace(-record['reference_peak'],record['reference_peak'],4096,dtype=np.float32)
            self.assertTrue(compare_array(reference,reference.copy(),rule)['pass'])
            for bad in [np.full_like(reference,np.nan),reference+max(.1*record['reference_peak'],.01)]:
                self.assertFalse(compare_array(reference,bad,rule)['pass'])
            if record['reference_peak']:
                for bad in [reference*1.1,reference[::-1].copy(),np.zeros_like(reference)]:self.assertFalse(compare_array(reference,bad,rule)['pass'])

    def test_final_body_checker_does_not_hide_invalid_hands_or_geometry(self):
        policy=json.loads((Path(__file__).resolve().parents[1]/'reference/bf16-body-policy-v1.json').read_text())
        native={name:np.zeros((1,3,3) if name=='vertices' else (1,2),dtype=np.float32) for name in FIELDS}
        native['faces']=np.array([[0,1,2]],dtype=np.int32)
        with tempfile.TemporaryDirectory() as d:
            reference=Path(d)/'original.safetensors';save_file({key:native[name] for name,key in FIELDS.items()},reference)
            policy['reference_sha256']=hashlib.sha256(reference.read_bytes()).hexdigest()
            with patch('check_bf16_body.read_output',return_value=native):
                self.assertTrue(check(Path(d)/'native.bin',reference,policy)['passed'])
                native['hand'][0,0]=100
                result=check(Path(d)/'native.bin',reference,policy)
                self.assertTrue(result['passed']);self.assertEqual(result['nonblocking_hand_failures'],['hand'])
                native['hand'][0,0]=np.nan
                self.assertEqual(check(Path(d)/'native.bin',reference,policy)['blocking_failures'],['hand'])
                native['hand'][0,0]=0;native['vertices'][0,0,0]=.1
                self.assertEqual(check(Path(d)/'native.bin',reference,policy)['blocking_failures'],['vertices'])
                native['faces'][0,0]=3
                with self.assertRaises(ValueError):check(Path(d)/'native.bin',reference,policy)
