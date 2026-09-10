"""Small deterministic format/rejection tests; no model downloads or GGUF fuzzing."""
import hashlib,json,struct,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from safetensors.numpy import save_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import convert_mhr_gguf as mhr
from convert_gguf import encode_header

class MHRConversionTests(unittest.TestCase):
    def test_schema_matches_original_asset_inventory(self):
        manifest=json.loads((Path(__file__).parent/'fixtures/mhr-schema.json').read_text())
        self.assertEqual(manifest['model_sha256'],mhr.ASSET_SHA)
        source={v['name']:v for v in manifest['state_inventory']}
        dtype={'F32':'float32','I32':'int32','I64':'int64'}
        self.assertEqual(len(mhr.TENSORS),18)
        for name,(key,kind,output,shape) in mhr.TENSORS.items():
            self.assertEqual(source[key]['shape'],list(shape),name);self.assertEqual(source[key]['dtype'],dtype[kind],name)
            self.assertEqual(output,0 if kind=='F32' else 26,name)

    def fixture(self,root,indices=None,weights=None):
        source=root/'state.safetensors';geometry=root/'geometry.json';manifest=root/'manifest.json'
        save_file({'indices':np.array([0,1,126] if indices is None else indices,dtype=np.int64),
                   'weights':np.array([.2,.3,.5] if weights is None else weights,dtype=np.float32)},source)
        geometry.write_text(json.dumps({'joint_names':[f'j{i}' for i in range(127)],'parameter_names':[f'p{i}' for i in range(249)],'prefix_sizes':[65,56,62,83]}))
        manifest.write_text(json.dumps({'model_sha256':mhr.ASSET_SHA,'model_bytes':696110248,
            'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,geometry]},'state_inventory':[{'name':'indices'},{'name':'weights'}]}))
        return source,manifest,geometry

    def convert(self,root,*args):
        schema={'skin.joints':('indices','I64',26,(3,)),'skin.weights':('weights','F32',0,(3,))}
        with patch.dict(mhr.TENSORS,schema,clear=True):return mhr.convert(*args,root/'out.gguf')

    def test_round_trip_and_nonoverwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root);r=self.convert(root,*args);self.assertEqual(r['tensors'],2)
            raw=(root/'out.gguf').read_bytes();self.assertEqual(raw[:4],b'GGUF')
            # Canonical two 32-byte-aligned payloads: no lossy float index detour.
            self.assertEqual(struct.unpack('<3i',raw[-64:-52]),(0,1,126))
            np.testing.assert_array_equal(np.frombuffer(raw[-32:-20],dtype='<f4'),np.array([.2,.3,.5],dtype=np.float32))
            with self.assertRaises(FileExistsError):self.convert(root,*args)
            self.assertEqual(raw,(root/'out.gguf').read_bytes())

    def test_reject_bad_indices_and_floats(self):
        for idx,w in [([-1,1,2],None),([0,127,1],None),([0,2**40,1],None),(None,[float('nan'),0,1]),(None,[-.1,.5,.6])]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);args=self.fixture(root,idx,w)
                with self.assertRaises(ValueError):self.convert(root,*args)
                self.assertFalse((root/'out.gguf').exists());self.assertFalse(list(root.glob('.mhr-*')))

    def test_manifest_and_geometry_rejections(self):
        for field,value in [('model_sha256','0'*64),('model_bytes',1)]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);args=self.fixture(root);data=json.loads(args[1].read_text());data[field]=value;args[1].write_text(json.dumps(data))
                with self.assertRaises(ValueError):self.convert(root,*args)
        for v in [['a','a'],['a,b'],['../bad'],[''],['x']]:
            with self.assertRaises(ValueError):mhr.names(v,2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root);args[2].write_text('{}')
            with self.assertRaises(ValueError):self.convert(root,*args)

    def test_index_contracts_and_header_types(self):
        for name,values in [('skeleton.parents',[-1,1]),('skeleton.prefix',[[0],[127]]),('pose.sparse.indices',[[0,0],[1,1]]),('mesh.faces',[[0,18439,2]])]:
            with self.assertRaises(ValueError):mhr.validate_indices(name,np.asarray(values,dtype=np.int64))
        for types in [{'x':1},{'x':True},{'other':26}]:
            with self.assertRaises(ValueError):encode_header({'a':'b'},{'x':(1,)},types)
        header,_=encode_header({'a':'b'},{'x':(1,)},{'x':26});self.assertTrue(header.startswith(b'GGUF'))
if __name__=='__main__':unittest.main()
