import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
from safetensors.numpy import save_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from body_branch_schema import shapes,metadata,INTEGER_NAMES,SAFE_SHA,BODY_REVISION,CHECKPOINT_SHA256,CONFIG_SHA256
from convert_body_branch import write_archive,validate_value,convert
from convert_gguf import sha256

class BodyBranchConversionTests(unittest.TestCase):
    def test_schema(self):
        s=shapes();self.assertEqual(len(s),316);self.assertEqual(len(metadata()),17)
        self.assertEqual(s['decoder.layers.5.cross_attn.k_proj.weight'],(512,1280))
        self.assertEqual(s['head_pose.proj.layers.0.0.weight'],(1024,1024))
        self.assertEqual(s['decoder.layers.0.ffn.layers.0.0.weight'],(1024,1024))
        self.assertFalse(any('hand.' in k or 'cross_attn_2' in k for k in s))
        self.assertTrue(INTEGER_NAMES.issubset(s));self.assertTrue(all(len(k)<64 for k in s))
    def test_safe_stream_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'safe.safetensors';target=Path(directory)/'branch.gguf'
            values={'weight':np.array([[-0.,2.,-1.]],np.float32),'head_pose.faces':np.array([[0,1,2]],np.int64)}
            save_file(values,source);spec={k:v.shape for k,v in values.items()}
            result=write_archive(source,target,spec,metadata(),sha256(source),{'head_pose.faces'})
            self.assertEqual(result['tensors'],2);self.assertEqual(target.read_bytes()[:4],b'GGUF')
            with self.assertRaises(FileExistsError):write_archive(source,target,spec,metadata(),sha256(source),{'head_pose.faces'})
            before=target.read_bytes();target.unlink();target.symlink_to('missing')
            with self.assertRaises(FileExistsError):write_archive(source,target,spec,metadata(),sha256(source),{'head_pose.faces'})
            target.unlink()
            with self.assertRaises(ValueError):write_archive(source,target,spec,metadata(),'0'*64,{'head_pose.faces'})
            self.assertFalse(target.exists());self.assertFalse(list(Path(directory).glob('.body-branch-*')))
    def test_reject_lossy_or_invalid_values(self):
        for values in [np.array([[-1,1,2]],np.int64),np.array([[0,18439,2]],np.int64),np.array([[0,2**40,2]],np.int64)]:
            with self.assertRaises(ValueError):validate_value('head_pose.faces',values,(1,3),True)
        with self.assertRaises(ValueError):validate_value('weight',np.array([np.nan],np.float32),(1,),False)
        with self.assertRaises(ValueError):validate_value('weight',np.ones(1,np.float64),(1,),False)
        with self.assertRaises(ValueError):validate_value('head_pose.hand_joint_idxs_left',np.full(27,95,np.int64),(27,),True)
    def test_extraction_identity_required(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest=Path(directory)/'manifest.json';manifest.write_text('{}')
            with self.assertRaises(ValueError):convert(Path(directory)/'safe.safetensors',manifest,Path(directory)/'out.gguf')
    def test_revision_and_source_identity(self):
        good=dict(schema_version=1,body_revision=BODY_REVISION,
                  artifacts={'other_state':dict(file='safe.safetensors',sha256=SAFE_SHA)},
                  source=[dict(path='model.ckpt',sha256=CHECKPOINT_SHA256),dict(path='model_config.yaml',sha256=CONFIG_SHA256)])
        with tempfile.TemporaryDirectory() as directory:
            manifest=Path(directory)/'manifest.json'
            for revision in [None,'wrong',BODY_REVISION]:
                data=dict(good,body_revision=revision);manifest.write_text(json.dumps(data))
                with patch('convert_body_branch.write_archive',return_value={'verified':True}) as writer:
                    if revision==BODY_REVISION:
                        self.assertEqual(convert(Path('safe.safetensors'),manifest,Path(directory)/'out.gguf'),{'verified':True})
                        writer.assert_called_once()
                    else:
                        with self.assertRaises(ValueError):convert(Path('safe.safetensors'),manifest,Path(directory)/'out.gguf')
                        writer.assert_not_called()
