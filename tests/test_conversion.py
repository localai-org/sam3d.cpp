import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import convert_gguf as convert
import gguf_schema as schema


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.source=self.root/'input.safetensors'
        self.output=self.root/'output.gguf'
        self.values={'linear.weight':np.arange(6,dtype=np.float32).reshape(2,3),
                     'linear.bias':np.array([.25,-.5],dtype=np.float32)}
        self.shapes={key:value.shape for key,value in self.values.items()}
        save_file(self.values,self.source)
        self.manifest={'schema_version':1,'architecture':schema.ARCHITECTURE,
            'body_revision':schema.BODY_REVISION,'dinov3_revision':schema.DINO_REVISION,
            'checkpoint_sha256':schema.CHECKPOINT_SHA256,'config_sha256':schema.CONFIG_SHA256,
            'safetensors_sha256':convert.sha256(self.source),'source_precision':['F32']}

    def run_conversion(self,values=None):
        if values is not None:
            save_file(values,self.source)
            self.manifest['safetensors_sha256']=convert.sha256(self.source)
        return convert.convert(self.source,self.output,convert.metadata(self.manifest),
                               self.shapes,self.manifest['safetensors_sha256'])

    def test_layout_and_no_overwrite(self):
        report=self.run_conversion()
        data=self.output.read_bytes()
        header,body_size=convert.encode_header(convert.metadata(self.manifest),self.shapes)
        self.assertEqual(data[:len(header)],header)
        self.assertEqual(len(data),len(header)+body_size)
        # Sorted bias first, padded to 32; source [out,in] bytes are unchanged.
        np.testing.assert_array_equal(np.frombuffer(data[len(header):len(header)+8],dtype='<f4'),self.values['linear.bias'])
        np.testing.assert_array_equal(np.frombuffer(data[len(header)+32:len(header)+56],dtype='<f4'),self.values['linear.weight'].ravel())
        self.assertFalse(report['model_parity_established'])
        with self.assertRaises(FileExistsError): self.run_conversion()
        self.assertEqual(data,self.output.read_bytes())

    def test_hash_missing_extra_dtype_shape_nonfinite(self):
        self.manifest['safetensors_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'hash mismatch'): self.run_conversion()
        variants=[{'linear.bias':self.values['linear.bias']},
            self.values|{'extra':np.ones(1,dtype=np.float32)},
            self.values|{'linear.weight':self.values['linear.weight'].astype(np.float16)},
            self.values|{'linear.weight':self.values['linear.weight'].reshape(3,2)},
            self.values|{'linear.bias':np.array([np.nan,0],dtype=np.float32)}]
        for values in variants:
            with self.assertRaises(ValueError): self.run_conversion(values)
            self.assertFalse(self.output.exists())
            self.assertEqual(list(self.root.glob('.sam3d-*')),[])

    def test_no_pickle_or_dangling_symlink_output(self):
        legacy=self.root/'model.ckpt'; legacy.write_bytes(b'not a checkpoint')
        with self.assertRaisesRegex(ValueError,'safetensors'):
            convert.convert(legacy,self.output,convert.metadata(self.manifest),self.shapes,'0'*64)
        self.output.symlink_to(self.root/'absent')
        with self.assertRaises(FileExistsError): self.run_conversion()
        self.assertTrue(self.output.is_symlink())

    def test_duplicate_safetensors_header(self):
        header=b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
        self.source.write_bytes(struct.pack('<Q',len(header))+header+bytes(4))
        self.manifest['safetensors_sha256']=convert.sha256(self.source)
        with self.assertRaisesRegex(ValueError,'duplicate JSON'): self.run_conversion()
        self.assertFalse(self.output.exists())

    def test_manifest_contract(self):
        path=self.root/'manifest.json'
        path.write_text(json.dumps(self.manifest))
        self.assertEqual(convert.read_manifest(path),self.manifest)
        for change in [{'architecture':'sam3d.body.other'}, {'schema_version':True},
                       {'source_precision':['Q4']},{'source_precision':['F32','F32']},
                       {'checkpoint_sha256':'bad'},{'checkpoint_sha256':'0'*64},
                       {'config_sha256':'0'*64},{'extra':0}]:
            path.write_text(json.dumps(self.manifest|change))
            with self.assertRaises(ValueError): convert.read_manifest(path)
        path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ValueError,'duplicate'): convert.read_manifest(path)

    def test_architecture_contract(self):
        shapes=schema.body_dino_shapes()
        self.assertEqual(len(shapes),552)
        original=json.loads((Path(__file__).parent/'fixtures/dino-schema.json').read_text())
        self.assertEqual(original['revision'],schema.DINO_REVISION)
        self.assertEqual({k:list(v) for k,v in shapes.items()},original['shapes'])
        self.assertEqual(shapes['blocks.31.mlp.w3.weight'],(1280,5120))
        self.assertEqual(shapes['storage_tokens'],(1,4,1280))
        with self.assertRaises(ValueError): convert.validate_tensor('rope_embed.periods',np.array([0],dtype=np.float32),(1,))
        with self.assertRaises(ValueError): convert.validate_tensor('attn.qkv.bias_mask',np.ones(6,dtype=np.float32),(6,))
        convert.validate_tensor('attn.qkv.bias_mask',np.zeros(6,dtype=np.float32),(6,))
        convert.validate_tensor('attn.qkv.bias_mask',np.array([1,1,0,0,1,1],dtype=np.float32),(6,))
        with self.assertRaises(ValueError):
            convert.validate_tensor('attn.qkv.bias_mask',np.array([.5,1,0,0,1,1],dtype=np.float32),(6,))


if __name__=='__main__': unittest.main()
