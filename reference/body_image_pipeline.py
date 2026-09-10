"""Reviewed original image/pose-branch composition with bounded reference memory.

Only parameter residency and the original hardcoded CUDA ray-grid placement are
adapted. Original transforms, backbone layers/wrapper and pose branch are called
unchanged. This is a correctness reference, NOT an optimized timing baseline.
"""
import ast,gc,importlib,importlib.util,json,struct,sys,types
from pathlib import Path
from typing import Dict,Optional
import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
from yacs.config import CfgNode
from safetensors import safe_open
from capture_body_camera import HASHES as PREP_HASHES
from capture_dino_schema import load_factory,HASHES as DINO_HASHES
from capture_dino_backbone import BODY_PATH,BODY_HASH,digest
from capture_dino_block import PARAMETERS
from body_image_cases import CASES

PHOTO_SHA='0112b0a32ea5860db6a6fc700804751528341a02bf07fed0329a0c2610f905d3'

class OriginalImagePipeline:
    def __init__(self,a,holder):
        self.holder=holder;self.device=a.device;self.taps={};self.observe=False;self.decoder_result=None
        root=a.upstream/'sam_3d_body'
        for name,sha in PREP_HASHES.items():
            if digest(root/name)!=sha:raise ValueError('unverified image source '+name)
        for name in ['sam_3d_body.data','sam_3d_body.data.utils','sam_3d_body.models.meta_arch','sam_3d_body.models.optim']:
            if name not in sys.modules:
                module=types.ModuleType(name);module.__path__=[str(a.upstream/Path(*name.split('.')))];sys.modules[name]=module
        transforms=importlib.import_module('sam_3d_body.data.transforms.common')
        self.prepare=importlib.import_module('sam_3d_body.data.utils.prepare_batch').prepare_batch
        Base=importlib.import_module('sam_3d_body.models.meta_arch.base_model').BaseModel
        holder.data_preprocess=types.MethodType(Base.data_preprocess,holder)
        holder.image_mean=torch.tensor([.485,.456,.406],device=a.device).view(3,1,1)
        holder.image_std=torch.tensor([.229,.224,.225],device=a.device).view(3,1,1)
        self.trained=getattr(a,'trained_extraction',None) is not None
        self.hand=getattr(a,'hand_branch',False)
        holder.cfg.MODEL.DECODER.CONDITION_TYPE='cliff'
        if not self.trained:holder.cfg.MODEL.PROMPT_ENCODER=CfgNode({})
        holder.backbone_dtype=torch.float32
        if self.hand:holder.body_batch_idx=[];holder.hand_batch_idx=torch.arange(1,device=a.device)
        else:holder.hand_batch_idx=[]
        self.transform=transforms.Compose([transforms.GetBBoxCenterScale(padding=.9 if self.hand else 1.25),transforms.TopdownAffine(input_size=(512,512)),transforms.VisionTransformWrapper(ToTensor())])
        source=root/'models/meta_arch/sam3d_body.py';tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SAM3DBody')
        names=['get_ray_condition','_get_decoder_condition','forward_pose_branch'];nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
        if self.trained:
            names.append('_get_mask_prompt');nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
        if len(nodes)!=len(names) or any(n.decorator_list for n in nodes):raise ValueError('unexpected original pose branch')
        namespace={'torch':torch,'Dict':Dict,'Optional':Optional};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
        for name in ['_get_decoder_condition','forward_pose_branch']:setattr(holder,name,types.MethodType(namespace[name],holder))
        if self.trained:holder._get_mask_prompt=types.MethodType(namespace['_get_mask_prompt'],holder)
        def rays(batch):
            cuda_batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            return namespace['get_ray_condition'](holder,cuda_batch).to(a.device)
        holder.get_ray_condition=rays
        decoder_name='forward_decoder_hand' if self.hand else 'forward_decoder'
        original_decoder=getattr(holder,decoder_name)
        def decoder(*args,**kwargs):
            self.decoder_result=original_decoder(*args,**kwargs);return self.decoder_result
        setattr(holder,decoder_name,decoder) # Non-replacing observation of the returned tuple.
        case_name=getattr(a,'image_case','dancer')
        photo_path,photo_sha,box=CASES[case_name]
        photo=a.upstream/photo_path
        if digest(photo)!=photo_sha:raise ValueError('unverified official sample image')
        self.rgb=cv2.cvtColor(cv2.imread(str(photo)),cv2.COLOR_BGR2RGB)
        self.height,self.width=self.rgb.shape[:2]
        self.box=np.array(box,dtype=np.float32) # Explicit manually selected box, not detector output.
        hand_metadata={}
        if self.hand:
            crop_manifest=json.loads((a.hand_crop_reference/'manifest.json').read_text())
            crop_path=a.hand_crop_reference/'upstream.safetensors'
            if crop_manifest['photo_sha256']!=PHOTO_SHA or digest(crop_path)!=crop_manifest['artifacts']['upstream.safetensors']:
                raise ValueError('unverified hand crop reference')
            with safe_open(crop_path,framework='np') as crop:
                self.box=crop.get_tensor('case.0000.'+a.hand_side+'.input_xyxy').reshape(4).copy()
            if self.box.dtype!=np.float32 or not np.isfinite(self.box).all() or np.any(self.box[2:]<=self.box[:2]):raise ValueError('invalid hand ROI')
            # Same full-image mirror as original run_inference. The supplied ROI
            # is already mirrored by the original crop fixture, not mirrored twice.
            if a.hand_side=='left':self.rgb=self.rgb[:,::-1,:].copy()
            hand_metadata={'hand_side':a.hand_side,'hand_crop_reference_sha256':digest(a.hand_crop_reference/'manifest.json'),
                'hand_crop_tensors_sha256':digest(crop_path),'roi_scope':'fixed original-derived ROI, not own-intermediate body detection',
                'input_mirrored':a.hand_side=='left','padding':.9}
        f=np.float32((self.height**2+self.width**2)**.5)
        self.intrinsics=np.array([f,f,self.width/2,self.height/2],dtype=np.float32)
        self.camera=torch.tensor([[[f,0,self.intrinsics[2]],[0,f,self.intrinsics[3]],[0,0,1]]],dtype=torch.float32)
        self.stream=None;self.safe=None
        if self.trained:
            from trained_body_state import BACKBONE_SHA
            self.weight_file=a.trained_extraction/'body-dinov3-f32.safetensors'
            if digest(self.weight_file)!=BACKBONE_SHA:raise ValueError('unverified trained backbone')
            self.safe=safe_open(self.weight_file,framework='pt');self.safe.__enter__()
        else:
            self.backbone_manifest=json.loads((a.backbone_reference/'manifest.json').read_text());self.weight_file=a.backbone_reference/'case.0000.input'
            if digest(self.weight_file)!=self.backbone_manifest['artifacts']['case.0000.input']:raise ValueError('unverified original synthetic backbone weights')
        factory=load_factory(a.dino_upstream)
        path=a.upstream/BODY_PATH
        if digest(path)!=BODY_HASH:raise ValueError('unverified Body backbone wrapper')
        spec=importlib.util.spec_from_file_location('original_body_wrapper_pipeline',path);body=importlib.util.module_from_spec(spec);spec.loader.exec_module(body)
        with torch.device('meta'):net=factory(pretrained=False).float().eval()
        self.net=net;self.shape=(1,512,512,16,1280,20,5120,32,4)
        if not self.trained:
            self.stream=self.weight_file.open('rb')
            if self.stream.read(8)!=b'S3DBBN01':raise ValueError('invalid backbone capture magic')
            if struct.unpack('<9I',self.stream.read(36))!=self.shape:raise ValueError('unsupported backbone capture shape')
            self.stream.seek(1*3*512*512*4,1) # Old fixture IMAGE is never used in this new RGB run.
        state=net.state_dict();order=['patch_embed.proj.weight','patch_embed.proj.bias','cls_token','mask_token','storage_tokens','rope_embed.periods']
        for i in range(32):order.extend(f'blocks.{i}.{key}' for key in PARAMETERS if key!='periods')
        order+=['norm.weight','norm.bias']
        if set(order)!=set(state):raise ValueError('incomplete original backbone schema')
        if self.trained and set(self.safe.keys())!=set(state):raise ValueError('trained backbone tensor set mismatch')
        self.entries={};offset=self.stream.tell() if self.stream else 0
        for name in order:
            value=state[name]
            if value.dtype!=torch.float32:raise ValueError('non-F32 original backbone state')
            self.entries[name]=(offset,tuple(value.shape),value.numel());offset+=value.numel()*4
        if not self.trained and offset!=self.weight_file.stat().st_size:raise ValueError('backbone file extent mismatch')
        def load(names):
            for name in names:
                offset,shape,count=self.entries[name]
                if self.trained:
                    tensor=self.safe.get_tensor(name)
                    if tuple(tensor.shape)!=shape or tensor.dtype!=torch.float32:raise ValueError('trained backbone layout mismatch')
                    tensor=tensor.to(a.device)
                else:
                    self.stream.seek(offset);raw=self.stream.read(count*4)
                    if len(raw)!=count*4:raise ValueError('truncated backbone tensor')
                    values=np.frombuffer(raw,dtype='<f4').copy()
                    if not np.isfinite(values).all():raise ValueError('non-finite backbone weights')
                    tensor=torch.from_numpy(values.reshape(shape)).to(a.device)
                parent,_,key=name.rpartition('.');module=net.get_submodule(parent) if parent else net
                if key in module._parameters:module._parameters[key]=torch.nn.Parameter(tensor,requires_grad=False)
                elif key in module._buffers:module._buffers[key]=tensor
                else:raise ValueError('unknown original state target')
        load([name for name in order if not name.startswith('blocks.')])
        self.resident_benchmark=getattr(a,'benchmark_precision',None) is not None
        self.observer_hooks=[]
        for i,block in enumerate(net.blocks):
            names=[name for name in order if name.startswith(f'blocks.{i}.')]
            def before(_m,_a,names=names):
                if torch.is_grad_enabled():raise ValueError('streamed reference requires no-grad')
                load(names)
            def after(module,_a,value,i=i):
                self.keep(f'backbone.02.block.{i:02d}',value)
                module.to_empty(device='meta')
                print(f'original backbone block {i+1}/32',flush=True)
            if self.resident_benchmark:load(names)
            else:block.register_forward_pre_hook(before);block.register_forward_hook(after)
        self.observer_hooks.append(net.patch_embed.register_forward_hook(lambda _m,_a,v:self.keep('backbone.00.patch',v.flatten(1,2))))
        self.observer_hooks.append(net.norm.register_forward_hook(lambda _m,_a,v:self.keep('backbone.03.norm',v)))
        original_prepare=net.prepare_tokens_with_masks
        def prepare_tokens(*args,**kwargs):
            value=original_prepare(*args,**kwargs);self.keep('backbone.01.tokens',value[0]);return value
        net.prepare_tokens_with_masks=prepare_tokens
        wrapper=torch.nn.Module();wrapper.encoder=net;wrapper.forward=types.MethodType(body.Dinov3Backbone.forward,wrapper)
        self.observer_hooks.append(wrapper.register_forward_pre_hook(lambda _m,args:self.keep('prepare.normalized_rgb',args[0])))
        self.observer_hooks.append(wrapper.register_forward_hook(lambda _m,_a,value:self.keep('backbone.04.features',value)))
        holder.backbone=wrapper
        if self.trained:
            for i,layer in enumerate(holder.bbox_embed.layers):
                self.observer_hooks.append(layer.register_forward_hook(lambda _m,_a,value,i=i:self.keep(f'branch.box.{i}.linear',value)))
        if self.resident_benchmark:
            for hook in self.observer_hooks:hook.remove()
            net.prepare_tokens_with_masks=original_prepare
            if a.benchmark_precision=='bf16':
                # Call upstream's explicit DINO conversion; do not blanket-cast
                # the F32 decoder, MHR or camera branches.
                Base._set_fp16(holder,wrapper,torch.bfloat16)
                holder.backbone_dtype=torch.bfloat16
        self.metadata={'photo_sha256':photo_sha,'bbox_xyxy':self.box.tolist(),'bbox_origin':'manually selected explicit box, not upstream detector output',
            'intrinsics_fx_fy_cx_cy':self.intrinsics.tolist(),
            'backbone_source_manifest_sha256':digest((a.trained_extraction/'extraction.json') if self.trained else (a.backbone_reference/'manifest.json')),
            ('backbone_safetensors_sha256' if self.trained else 'backbone_input_sha256'):digest(self.weight_file),'source_hashes':PREP_HASHES,'dino_hashes':DINO_HASHES,'body_wrapper_sha256':BODY_HASH,
            'script_sha256':digest(Path(__file__)),'reference_weight_residency':('all original layers resident; no loading/observer hooks' if self.resident_benchmark else 'original layers with pre/post-hook weight loading/release; no neural operation replacement'),
            'ray_device':'original hardcoded CUDA calculation, then transfer to selected neural device; not a CPU-only performance baseline'}
        if self.hand:self.metadata.update(hand_metadata,bbox_origin='explicit original hand-crop fixture ROI')
        if case_name!='dancer':self.metadata.update(image_case=case_name,photo_path=photo_path,published_bbox_overlay=True)

    def close(self):
        if self.stream:self.stream.close()
        if self.safe:self.safe.__exit__(None,None,None)

    def keep(self,name,value):
        if self.observe:self.taps[name]=value.detach().clone()

    def run(self):
        batch=self.prepare(self.rgb.copy(),self.transform,self.box[None],cam_int=self.camera)
        batch={k:v.to(self.device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        branch=self.holder.forward_pose_branch(batch)
        output=branch['mhr_hand' if self.hand else 'mhr']
        if self.decoder_result is None or output is not self.decoder_result[1][-1]:raise ValueError('original branch/decoder output identity changed')
        self.keep('backbone.features',branch['image_embeddings'])
        if self.trained:
            if torch.any(batch['mask_score']>0):raise ValueError('this fixture requires absent mask')
            self.keep('branch.hand_box',output['hand_box']);self.keep('branch.hand_logits',output['hand_logits'])
        for name,key in [('rays','ray_cond_hand' if self.hand else 'ray_cond'),('cliff','condition_info')]:self.keep('prepare.'+name,batch[key] if key in batch else branch[key])
        for name,key in [('box_center','bbox_center'),('image_size','ori_img_size'),('crop_size','img_size'),('affine','affine_trans'),('intrinsics','cam_int')]:self.keep('prepare.'+name,batch[key])
        self.keep('prepare.box_size',batch['bbox_scale'][...,:1])
        return self.decoder_result

    def write_input(self,stream):
        stream.write(struct.pack('<3I',self.width,self.height,self.width*3))
        stream.write(self.box.astype('<f4').tobytes()+self.intrinsics.astype('<f4').tobytes()+self.rgb.tobytes())

    def normalized_image(self):
        """Original preprocessing boundary for separate layer captures, untimed."""
        if self.holder.cfg.MODEL.BACKBONE.TYPE!='dinov3_vith16plus' or self.hand:raise ValueError('normalized capture requires the body DINO branch')
        batch=self.prepare(self.rgb.copy(),self.transform,self.box[None],cam_int=self.camera)
        image=self.holder._flatten_person(batch['img'].to(self.device))
        result=self.holder.data_preprocess(image,crop_width=False)
        if tuple(result.shape)!=(1,3,512,512) or result.dtype!=torch.float32 or not torch.isfinite(result).all():raise ValueError('invalid original normalized image')
        return result.detach().cpu().contiguous()
