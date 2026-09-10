#!/usr/bin/env python3
"""Unchanged Body forward_decoder and full promptable feedback loop, real MHR.

Synthetic SAM neural/PCA/keypoint state and input features: NOT trained image
inference. Original heads and callbacks produce every intermediate themselves.
Run only in the reviewed isolated container; the verified MHR is TorchScript.
"""
import argparse,ast,importlib,json,os,struct,sys,types
from pathlib import Path
from typing import Dict,Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from torch.utils._python_dispatch import TorchDispatchMode
from safetensors.numpy import save_file
from yacs.config import CfgNode
from capture_mhr import MODEL_SHA,MODEL_BYTES,digest
from capture_body_pose import ROMA_SHA,MHR_SHA
from capture_body_condition import BODY_SHA256,DECODER_SHA256,PROMPT_SHA256
from capture_camera_head import HEAD_SHA256,GEOMETRY_SHA256
from capture_camera_encoder import HASHES
from trace_mhr_repeat import error

class Observe(TorchDispatchMode):
    def __init__(self,keep,active):super().__init__();self.keep=keep;self.active=active;self.mappings=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        value=func(*args,**(kwargs or {}))
        if str(func) in ['aten.mm.default','aten.matmul.default'] and isinstance(value,torch.Tensor) and tuple(value.shape)==(308,self.active['batch']*3):
            self.mappings+=1;self.keep('pose.map.05.mapping_input',args[1]);self.keep('pose.map.06.keypoints_linear',value)
        if self.active['projection']==70 and func==torch.ops.aten.div.Tensor and isinstance(value,torch.Tensor) and value.ndim==3:
            self.keep('camera.18.normalized',value)
        return value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model','upstream','roma-wheel','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--full-width',action='store_true',help='one complete six-layer 512px/1280-context/1024-token case, instead of small regression cases')
    p.add_argument('--image-pipeline',action='store_true',help='run original image preparation/backbone/pose branch on official dancing image')
    p.add_argument('--dino-upstream',type=Path);p.add_argument('--backbone-reference',type=Path)
    p.add_argument('--trained-extraction',type=Path);p.add_argument('--trained-config',type=Path)
    p.add_argument('--decoder-operations',action='store_true')
    p.add_argument('--hand-branch',action='store_true',help='actual trained hand decoder; synthetic features unless --image-pipeline is supplied')
    p.add_argument('--hand-crop-reference',type=Path,help='verified original hand crop fixture supplying an explicit hand ROI, not a body-to-hand inference claim')
    p.add_argument('--hand-side',choices=['left','right'])
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--benchmark-precision',choices=['f32','bf16'])
    p.add_argument('--benchmark-compile',choices=['none','backbone'],default='none')
    p.add_argument('--benchmark-attention',choices=['math','auto'],default='auto')
    p.add_argument('--benchmark-warmup',type=int,default=5)
    p.add_argument('--benchmark-repeats',type=int,default=20)
    from body_image_cases import CASES
    p.add_argument('--image-case',choices=list(CASES),default='dancer')
    a=p.parse_args();a.upstream=a.upstream.resolve();root=a.upstream/'sam_3d_body'
    helpers=['body_image_pipeline.py','trained_body_state.py','observe_body_decoder.py','inspect_math_sdpa.py']
    helper_hashes={name:digest(Path(__file__).with_name(name)) for name in helpers}
    if not 1<=a.threads<=24:raise ValueError('invalid thread count')
    if a.image_case!='dancer' and (not a.image_pipeline or a.hand_branch):raise ValueError('alternate photograph requires body image pipeline, not hand fixture')
    if a.benchmark_precision and (not a.trained_extraction or not a.image_pipeline or a.hand_branch):raise ValueError('benchmark requires trained body image pipeline')
    if a.hand_branch and (not a.full_width or not a.trained_extraction or not a.decoder_operations):raise ValueError('hand capture requires full width, real trained state and decoder operations')
    if a.hand_branch and a.image_pipeline and (not a.hand_crop_reference or not a.hand_side):raise ValueError('hand image capture requires explicit original crop fixture and side')
    if (a.hand_crop_reference or a.hand_side) and not (a.hand_branch and a.image_pipeline):raise ValueError('hand crop options require the hand image pipeline')
    if a.trained_extraction and (not (a.image_pipeline or a.hand_branch) or not a.trained_config):raise ValueError('trained capture requires RGB/hand pipeline and verified config')
    if a.decoder_operations and not a.trained_extraction:raise ValueError('decoder operation tracing requires trained branch')
    if a.image_pipeline and (not a.full_width or not a.dino_upstream or not (a.backbone_reference or a.trained_extraction)):raise ValueError('image pipeline requires full width and original backbone source/state')
    trained_cfg=None;trained_metadata=None
    if a.trained_extraction:
        from trained_body_state import config,load as load_trained
        trained_cfg=config(a.trained_config)
    if a.model.stat().st_size!=MODEL_BYTES or digest(a.model)!=MODEL_SHA or digest(a.roma_wheel)!=ROMA_SHA:raise ValueError('unexpected MHR/RoMa asset')
    hashes={f'models/modules/{k}':v for k,v in HASHES.items()};hashes|={
        'models/heads/mhr_head.py':'03793d51ef82484c5d9906358c6314981c0bff9dbcc6f6e6b175bd977a251b35',
        'models/heads/camera_head.py':HEAD_SHA256,'models/modules/mhr_utils.py':MHR_SHA,'models/modules/geometry_utils.py':GEOMETRY_SHA256,
        'models/modules/__init__.py':'ddad96a6a9bf09d36156ba82ebaa9ee6798045fe1fc715174cec4c0b4bd2ead1',
        'models/modules/misc.py':'f942d181ee8578c67a9ab4c0828d030aefdadd8891450ab1319b33a8237891eb',
        'models/meta_arch/sam3d_body.py':BODY_SHA256,'models/meta_arch/base_model.py':'baf4c93ab865e6e9f4f498056a673698e59bafe89b17f969833884a3b352e8bc',
        'models/decoders/prompt_encoder.py':PROMPT_SHA256,'models/decoders/promptable_decoder.py':DECODER_SHA256}
    for name,sha in hashes.items():
        if digest(root/name)!=sha:raise ValueError('source changed '+name)
    wheel=a.roma_wheel.resolve();sys.path.insert(0,str(wheel));import roma
    if roma.__version__!='1.6.1' or not str(roma.__file__).startswith(str(wheel)+'/'):raise ValueError('unexpected RoMa import')
    for name in ['sam_3d_body','sam_3d_body.models','sam_3d_body.models.heads','sam_3d_body.models.decoders']:
        package=types.ModuleType(name);package.__path__=[str(a.upstream/Path(*name.split('.')))];sys.modules[name]=package
    os.environ['MOMENTUM_ENABLED']='0'
    Head=importlib.import_module('sam_3d_body.models.heads.mhr_head').MHRHead
    CameraHead=importlib.import_module('sam_3d_body.models.heads.camera_head').PerspectiveHead
    utils=importlib.import_module('sam_3d_body.models.modules.mhr_utils')
    Prompt=importlib.import_module('sam_3d_body.models.decoders.prompt_encoder').PromptEncoder
    Camera=importlib.import_module('sam_3d_body.models.modules.camera_embed').CameraEncoder
    Decoder=importlib.import_module('sam_3d_body.models.decoders.promptable_decoder').PromptableDecoder
    FFN=importlib.import_module('sam_3d_body.models.modules.transformer').FFN
    MLP=importlib.import_module('sam_3d_body.models.modules.transformer').MLP
    Position=importlib.import_module('sam_3d_body.models.decoders.prompt_encoder').PositionEmbeddingRandom
    namespace={'torch':torch,'F':F,'Dict':Dict,'Optional':Optional};selected={}
    methods=['forward_decoder','camera_project','_full_to_crop','keypoint_token_update_fn','keypoint3d_token_update_fn']
    if a.hand_branch:methods=[name if name=='_full_to_crop' else name+'_hand' for name in methods]
    for filename,classname,names in [('sam3d_body.py','SAM3DBody',methods),('base_model.py','BaseModel',['_flatten_person'])]:
        source=root/'models/meta_arch'/filename;tree=ast.parse(source.read_text(),filename=str(source));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
        nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
        if len(nodes)!=len(names) or any(n.decorator_list for n in nodes):raise ValueError('unexpected methods')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace);selected[filename]=names
    pose_source=root/'models/heads/mhr_head.py';body_source=root/'models/meta_arch/sam3d_body.py';camera_source=root/'models/heads/camera_head.py'
    tree=ast.parse(pose_source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MHRHead');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='mhr_forward')
    call=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='self' and n.value.func.attr=='mhr');after=method.body[call+1].lineno
    torch.set_num_threads(a.threads);torch.manual_seed(8611);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(8611);a.output.mkdir(parents=True,exist_ok=True);cases=[];tensors={};full={};norm_cases=[]
    # B,depth,hand tokens,previous estimate,repeat PE,intrinsics center.
    configurations=[(1,6,0,0,1,0)] if a.full_width else [(2,2,0,0,1,0),(1,6,1,1,0,1)]
    if trained_cfg is not None:configurations=[(1,6,1,0,1,1)]
    for index,(b,depth,hand,has_prev,repeat,center_flag) in enumerate(configurations):
        h,w,patch,c,d,n,heads,dh,hidden,ph,pdepth,ch,cdepth=4,6,2,8,8,2,2,4,16,4,2,4,2
        if a.full_width:h,w,patch,c,d,n,heads,dh,hidden,ph,pdepth,ch,cdepth=512,512,16,1280,1024,1,8,64,4096,128,2,128,2
        twoway=True
        if trained_cfg is not None:
            cfg=trained_cfg.MODEL;dc=cfg.DECODER
            h,w=cfg.IMAGE_SIZE;d,depth,heads,dh,hidden=dc.DIM,dc.DEPTH,dc.HEADS,dc.DIM_HEAD,dc.MLP_DIM
            ph,pdepth=d//cfg.MHR_HEAD.MLP_CHANNEL_DIV_FACTOR,cfg.MHR_HEAD.MLP_DEPTH
            ch,cdepth=d//cfg.CAMERA_HEAD.MLP_CHANNEL_DIV_FACTOR,cfg.CAMERA_HEAD.MLP_DEPTH
            twoway=dc.ENABLE_TWOWAY
        print(json.dumps({'case':index,'phase':'construct','context_dim':c,'token_dim':d,'layers':depth}),flush=True)
        scale=float(np.float32(trained_cfg.MODEL.CAMERA_HEAD.get('DEFAULT_SCALE_FACTOR_HAND',1.) if a.hand_branch else .8 if index else 1.));holder=torch.nn.Module()
        holder.cfg=CfgNode({'MODEL':{'BACKBONE':{'TYPE':'dinov3'},'DECODER':{'DO_KEYPOINT_TOKENS':True,'DO_KEYPOINT3D_TOKENS':True,'DO_HAND_DETECT_TOKENS':bool(hand),'USE_INTRIN_CENTER':bool(center_flag)}}})
        if trained_cfg is not None:holder.cfg=trained_cfg.clone()
        for names in selected.values():
            for name in names:setattr(holder,name,types.MethodType(namespace[name],holder))
        holder._batch_size=b;holder._max_num_person=1;holder.body_batch_idx=torch.arange(b,device=a.device);holder.pelvis_idx=[9,10]
        holder.init_pose=torch.nn.Embedding(1,519);holder.init_camera=torch.nn.Embedding(1,3)
        holder.init_to_token_mhr=torch.nn.Linear(525,d);holder.prev_to_token_mhr=torch.nn.Linear(522,d)
        holder.prompt_encoder=Prompt(c,70,mask_embed_type='v2' if trained_cfg is not None else None);holder.prompt_to_token=torch.nn.Linear(c,d);holder.ray_cond_emb=Camera(c,patch)
        holder.keypoint_embedding=torch.nn.Embedding(70,d);holder.keypoint3d_embedding=torch.nn.Embedding(70,d)
        holder.keypoint_embedding_idxs=list(range(70));holder.keypoint3d_embedding_idxs=list(range(70))
        holder.keypoint_posemb_linear=FFN(embed_dims=2,feedforward_channels=d,output_dims=d,num_fcs=2,add_identity=False)
        holder.keypoint3d_posemb_linear=FFN(embed_dims=3,feedforward_channels=d,output_dims=d,num_fcs=2,add_identity=False)
        holder.keypoint_feat_linear=torch.nn.Linear(c,d)
        if hand:holder.hand_box_embedding=torch.nn.Embedding(2,d)
        if trained_cfg is not None and (not a.hand_branch or a.image_pipeline):
            holder.hand_cls_embed=torch.nn.Linear(d,2);holder.bbox_embed=MLP(d,d,4,3)
        holder.decoder=Decoder(d,c,depth,num_heads=heads,head_dims=dh,mlp_dims=hidden,repeat_pe=bool(repeat),enable_twoway=twoway,do_interm_preds=True,keypoint_token_update='v2' if trained_cfg is not None else True)
        holder.head_pose=Head(d,mlp_depth=pdepth,mhr_model_path=str(a.model),ffn_zero_bias=False,mlp_channel_div_factor=d//ph,enable_hand_model=a.hand_branch)
        holder.head_camera=CameraHead(d,(w,h),mlp_depth=cdepth,mlp_channel_div_factor=d//ch,default_scale_factor=scale)
        holder=holder.float().to(a.device).eval()
        def random(shape,scale=1):return torch.from_numpy(rng.normal(scale=scale,size=shape).astype(np.float32)).to(a.device)
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            # Never replace/randomize the released MHR state.
            for name,v in holder.state_dict().items():
                if name.startswith('head_pose.'):continue
                if name.endswith('norm_final.weight') or '.ln' in name and name.endswith('weight') or name.endswith('norm.weight'):v.copy_(1+random(tuple(v.shape),.05))
                elif name.startswith('head_camera.'):v.copy_(random(tuple(v.shape),.01))
                else:v.copy_(random(tuple(v.shape),.2/np.sqrt(v.shape[-1]) if v.ndim>=2 else .03))
            head=holder.head_pose;holder.init_pose.weight.copy_(head.get_zero_pose_init());holder.init_camera.weight.copy_(torch.tensor([[-1.,.02,-.03]],device=a.device))
            for name,v in head.proj.named_parameters():v.copy_(random(tuple(v.shape),.01/np.sqrt(v.shape[1]) if v.ndim==2 else .003))
            head.scale_mean.copy_(random((68,),.01));head.scale_comps.copy_(random((28,68),.01))
            head.hand_pose_mean.copy_(utils.compact_model_params_to_cont_hand(torch.zeros(1,27))[0].to(a.device));head.hand_pose_comps.copy_(torch.eye(54,device=a.device))
            indices=rng.permutation(np.arange(68,122,dtype=np.int32));head.hand_joint_idxs_left.copy_(torch.from_numpy(indices[:27].astype(np.int64)).to(a.device));head.hand_joint_idxs_right.copy_(torch.from_numpy(indices[27:].astype(np.int64)).to(a.device));head.faces.copy_(head.mhr.character_torch.mesh.faces)
            mapping=np.zeros((308,18439+127),dtype=np.float32);pool=np.concatenate([np.linspace(0,18438,16,dtype=np.int32),18439+np.array([0,4,12,36,57,83,126])])
            for row in range(308):
                cols=rng.choice(pool,5,replace=False);weights=rng.uniform(-.2,1.,5).astype(np.float32);weights/=np.sum(weights);mapping[row,cols]=weights
            head.keypoint_mapping.copy_(torch.from_numpy(mapping).to(a.device))
            # Overwrite and validate EVERY branch tensor before any inference.
            # No synthetic initialization is allowed to survive in trained mode.
            if a.hand_branch:
                # Retain local references for observation only. The original
                # methods resolve their real *_hand attributes, never body aliases.
                independent=['init_pose','init_camera','init_to_token_mhr','prev_to_token_mhr','ray_cond_emb','keypoint_embedding','keypoint3d_embedding','keypoint_posemb_linear','keypoint_feat_linear','keypoint3d_posemb_linear','decoder','head_pose','head_camera']
                for name in independent:
                    module=getattr(holder,name);delattr(holder,name);setattr(holder,name+'_hand',module)
                holder.hand_pe_layer=Position(c//2).float().to(a.device)
                holder.hand_batch_idx=holder.body_batch_idx;del holder.body_batch_idx
                holder.keypoint_embedding_idxs_hand=holder.keypoint_embedding_idxs;holder.keypoint3d_embedding_idxs_hand=holder.keypoint3d_embedding_idxs
                from trained_body_state import load_hand
                indices,trained_metadata=load_hand(holder,a.trained_extraction)
            elif trained_cfg is not None:indices,trained_metadata=load_trained(holder,a.trained_extraction)
            decoder=holder.decoder_hand if a.hand_branch else holder.decoder
            camera_head=holder.head_camera_hand if a.hand_branch else holder.head_camera
            init_projection=holder.init_to_token_mhr_hand if a.hand_branch else holder.init_to_token_mhr
            prev_projection=holder.prev_to_token_mhr_hand if a.hand_branch else holder.prev_to_token_mhr
            features=random((b,c,h//patch,w//patch));rays=random((b,2,h,w),.6);cliff=random((b,3),.2)
            prompts=torch.tensor([[[0.,0.,-2],[.6,.4,9.]]]*b,device=a.device)
            if a.full_width:prompts=prompts[:,:1].contiguous()
            previous=random((b,1,522),.1) if has_prev else None
            image_size=torch.tensor([[997.,613.],[1920.,1080.]][:b],device=a.device);box_center=image_size*torch.tensor([.43,.61],device=a.device);box_size=torch.tensor([450.,500.][:b],device=a.device)
            intrinsics=torch.tensor([[[850+90*z,3,.46*float(image_size[z,0])],[-2,925+80*z,.52*float(image_size[z,1])],[0,0,1]] for z in range(b)],dtype=torch.float32,device=a.device)
            # Crop is original box -> decoder input pixels; asymmetric shear in
            # case 1 exercises nontrivial full-to-crop beyond a scalar resize.
            affine=torch.zeros((b,2,3),device=a.device);affine[:,0,0]=w/box_size;affine[:,1,1]=h/box_size
            affine[:,0,2]=w*.5-affine[:,0,0]*box_center[:,0];affine[:,1,2]=h*.5-affine[:,1,1]*box_center[:,1]
            if index:affine[:,0,1]=.0002
            crop_size=torch.tensor([[float(w),float(h)]]*b,device=a.device)
            batch={'ray_cond':rays,'bbox_center':box_center[:,None],'bbox_scale':box_size[:,None,None].expand(-1,1,2).contiguous(),'ori_img_size':image_size[:,None],
                   'cam_int':intrinsics,'img':torch.zeros((b,1,3,h,w),device=a.device),'affine_trans':affine[:,None],'img_size':crop_size[:,None]}
            if a.hand_branch:batch['ray_cond_hand']=batch.pop('ray_cond')
            pipeline=None
            if a.image_pipeline:
                from body_image_pipeline import OriginalImagePipeline
                pipeline=OriginalImagePipeline(a,holder)
            def run():
                if pipeline is not None:return pipeline.run()
                forward=holder.forward_decoder_hand if a.hand_branch else holder.forward_decoder
                return forward(features,keypoints=prompts,prev_estimate=previous,condition_info=cliff,batch=batch)
            def clone(value):return (value[0].detach().clone(),[{k:v.detach().clone() for k,v in o.items() if isinstance(v,torch.Tensor)} for o in value[1]])
            if a.benchmark_precision:
                from benchmark_body import benchmark
                benchmark(a,pipeline,trained_metadata)
                pipeline.close()
                return
            print(json.dumps({'case':index,'phase':'uninstrumented_warmup_and_baseline'}),flush=True)
            run();baseline=clone(run())
            if trained_cfg is not None:
                repeat_value=clone(run())
                if not torch.equal(baseline[0],repeat_value[0]) or any(not torch.equal(v,repeat_value[1][i][k]) for i,o in enumerate(baseline[1]) for k,v in o.items()):raise ValueError('trained branch reference is not repeatable')
            taps={};hooks=[];active={'i':0,'batch':b,'projection':0}
            print(json.dumps({'case':index,'phase':'observed'}),flush=True)
            if pipeline is not None:pipeline.observe=True
            def keep(name,v):taps[f'layer.{active["i"]}.'+name]=v.detach().clone()
            def cond(name,v):taps['condition.'+name]=v.detach().clone()
            decoder_observer=None
            if a.decoder_operations:
                from observe_body_decoder import BodyDecoderObserver
                decoder_observer=BodyDecoderObserver(decoder,lambda name,v:taps.__setitem__(name,v.detach().clone()))
            def trace(frame,event,value):
                if frame.f_code.co_filename!=str(pose_source) or frame.f_code.co_name!='mhr_forward':return None
                if event=='line' and frame.f_lineno==after:keep('pose.mhr.vertices_cm',frame.f_locals['curr_skinned_verts']);keep('pose.mhr.skeleton',frame.f_locals['curr_skel_state'])
                return trace
            def profile(frame,event,value):
                filename,fn=frame.f_code.co_filename,frame.f_code.co_name;l=frame.f_locals
                if filename==str(camera_source) and fn=='perspective_projection' and event=='call':active['projection']=l['points_3d'].shape[1]
                if event!='return':return
                if filename==str(pose_source):
                    if fn=='mhr_forward':
                        for name,key in [('00.vertices_m','curr_skinned_verts'),('01.joints_m','curr_joint_coords'),('02.quaternions','curr_joint_quats'),('03.joint_rotations','curr_joint_rots'),('04.vertex_joints','model_vert_joints'),('07.keypoints308','model_keypoints_pred')]:keep('pose.map.'+name,l[key])
                        keep('pose.map.08.first70',l['model_keypoints_pred'][:,:70]);keep('pose.pose.90.model_params',l['model_params'])
                    if fn=='forward':
                        keep('pose.pose.10.pred',l['pred'])
                        for name,key in [('24.shape','shape'),('25.scale','scale'),('26.hand','hand'),('27.face','face')]:keep('pose.pose.'+name,value[key])
                        for name,key in [('90.vertices','pred_vertices'),('91.joints','pred_joint_coords'),('92.keypoints','pred_keypoints_3d')]:keep('pose.map.'+name,value[key])
                        for key in ['pred_pose_raw','global_rot','body_pose']:keep('pose.'+key,value[key])
                if filename==str(camera_source):
                    if fn=='forward':keep('camera.10.pred_cam',value)
                    if fn=='perspective_projection':
                        if active['projection']==70:
                            for name,key in [('11.corrected_cam','pred_cam'),('12.scaled_box','bs'),('13.focal','focal_length'),('15.translation','pred_cam_t'),('16.camera_points','j3d_cam')]:keep('camera.'+name,l[key])
                            keep('camera.14.offset',torch.stack([l['cx'],l['cy']],dim=-1));keep('camera.17.depth',value['pred_keypoints_2d_depth']);keep('camera.20.pixels',value['pred_keypoints_2d'])
                        else:keep('03.vertex_pixels',value['pred_keypoints_2d'])
                        active['projection']=0
                if filename==str(root/'models/modules/geometry_utils.py') and fn=='perspective_projection' and active['projection']==70:keep('camera.19.intrinsic_projection',l['y'])
                if filename==str(body_source):
                    if fn=='_full_to_crop':keep('04.crop_points',value)
                    if fn=='keypoint_token_update_fn_comb':keep('05.feedback_tokens',value[0]);keep('06.feedback_augment',value[1])
            for i,layer in enumerate(decoder.layers):
                hooks.append(layer.register_forward_pre_hook(lambda _m,_a,i=i:active.update(i=i)))
                hooks.append(layer.register_forward_hook(lambda _m,_a,v:(keep('00.tokens',v[0]),keep('01.context',v[1])) and None))
            hooks.append(decoder.norm_final.register_forward_hook(lambda _m,_a,v:keep('02.normalized',v)))
            for module,pre,post in [(init_projection,'00.init_input','01.pose_token'),(prev_projection,'02.prev_input','03.prev_token')]:
                hooks.append(module.register_forward_pre_hook(lambda _m,a,pre=pre:cond(pre,a[0])));hooks.append(module.register_forward_hook(lambda _m,_a,v,post=post:cond(post,v)))
            hooks.append(holder.prompt_encoder.register_forward_hook(lambda _m,_a,v:(cond('10.prompt_input',v[0]),cond('11.prompt_mask',v[1])) and None))
            hooks.append(holder.prompt_to_token.register_forward_hook(lambda _m,_a,v:cond('12.prompt_token',v)))
            def decoder_pre(_m,args):
                if args[4] is not None:raise ValueError('unexpected token mask')
                for name,v in zip(['30.tokens','20.image','31.token_pe','21.image_pe'],args[:4]):cond(name,v)
            hooks.append(decoder.register_forward_pre_hook(decoder_pre))
            for i in range(cdepth):
                module=camera_head.proj.layers[i] if i+1==cdepth else camera_head.proj.layers[i][0]
                hooks.append(module.register_forward_hook(lambda _m,_a,v,i=i:keep(f'camera.00.ffn.{i}.linear',v)))
                if i+1<cdepth:hooks.append(camera_head.proj.layers[i][1].register_forward_hook(lambda _m,_a,v,i=i:keep(f'camera.00.ffn.{i}.relu',v)))
            observer=Observe(keep,active);old_trace,old_profile=sys.gettrace(),sys.getprofile()
            try:
                sys.settrace(trace);sys.setprofile(profile)
                with observer:observed=clone(run())
            finally:
                sys.settrace(old_trace);sys.setprofile(old_profile)
                for hook in hooks:hook.remove()
                if decoder_observer:decoder_observer.close()
            if observer.mappings!=depth:raise ValueError('missing MHR mapping calls')
            taps['90.output_tokens']=observed[0]
            if pipeline is not None:taps.update(pipeline.taps)
            equivalence={'tokens':error(baseline[0].cpu().numpy(),observed[0].cpu().numpy())}
            for i,output in enumerate(baseline[1]):
                for key,v in output.items():equivalence[f'layer.{i}.{key}']=error(v.cpu().numpy(),observed[1][i][key].cpu().numpy())
            if any(e['max_abs']>1e-3 or e['relative_l2']>2e-5 for e in equivalence.values()):raise ValueError('observation changed original decoder')
            if trained_cfg is not None and any(e['max_abs']!=0 for e in equivalence.values()):raise ValueError('trained observation is not byte-exact')
        head_prefix='head_pose_hand.' if a.hand_branch else 'head_pose.'
        state={k:v for k,v in holder.state_dict().items() if not k.startswith((head_prefix,'backbone.'))}
        # Mask convolution is executed upstream, but its output is unselected
        # for this no-mask case. Keep the selected no-mask embedding, never drop it.
        state={k:v for k,v in state.items() if not k.startswith('prompt_encoder.mask_downscaling.')}
        if a.hand_branch and pipeline is None:state.pop('prompt_encoder.no_mask_embed.weight')
        state.update({head_prefix+'proj.'+k:v for k,v in head.proj.state_dict().items()})
        for key in ['scale_mean','scale_comps','hand_pose_mean','hand_pose_comps','keypoint_mapping']:state[head_prefix+key]=getattr(head,key)
        if a.hand_branch:
            for key in ['local_to_world_wrist','right_wrist_coords','root_coords']:state[head_prefix+key]=getattr(head,key)
        shape=(b,h,w,patch,c,d,n,70,519,70,1,hand,depth,heads,dh,hidden,ph,pdepth,ch,cdepth,repeat,int(twoway),center_flag,has_prev)
        prefix=f'case.{index:04d}';filename=prefix+'.input'
        with (a.output/filename).open('wb') as f:
            f.write((b'S3DHRG01' if a.hand_branch and pipeline is not None else b'S3DHFL01' if a.hand_branch else b'S3DRGB03' if a.decoder_operations else b'S3DRGB02' if trained_cfg is not None else b'S3DRGB01' if pipeline is not None else b'S3DFLW01')+struct.pack('<24If',*shape,scale)+indices.astype('<i4').tobytes())
            if a.hand_branch:f.write(head.nonhand_param_idxs.detach().cpu().numpy().astype('<i4').tobytes())
            if pipeline is not None:
                pipeline.write_input(f);values=[prompts,*([previous] if has_prev else []),*[state[k] for k in sorted(state)]]
            else:values=[features,rays,cliff,prompts,*([previous] if has_prev else []),box_center,box_size,image_size,intrinsics,affine,crop_size,*[state[k] for k in sorted(state)]]
            for v in values:f.write(v.detach().cpu().numpy().astype('<f4').tobytes())
        for name,v in taps.items():tensors[prefix+'.'+name]=v.cpu().numpy().copy()
        full[prefix+'.tokens']=baseline[0].cpu().numpy().copy()
        for i,output in enumerate(baseline[1]):
            for key,v in output.items():full[prefix+f'.layer.{i}.'+key]=v.cpu().numpy().copy()
        # Small, original norm_final inputs/outputs for normal no-weights tests.
        if not a.full_width:
            for i in range(depth):norm_cases.append((b,2+n+2*hand+140,d,taps[f'layer.{i}.00.tokens'],state['decoder.norm_final.weight'],state['decoder.norm_final.bias'],taps[f'layer.{i}.02.normalized']))
        cases.append({'prefix':prefix,'input':filename,'depth':depth,'shape':shape,'scale':scale,'order':sorted(taps),'shapes':{k:list(v.shape) for k,v in taps.items()},'unobserved_vs_observed':equivalence,
            **({'image_pipeline':pipeline.metadata} if pipeline is not None else {})})
        print(json.dumps({'case':index,'depth':depth,'taps':len(taps),'observer_max_abs':max(e['max_abs'] for e in equivalence.values())}),flush=True)
        if pipeline is not None:pipeline.close()
        del holder,head,pipeline
    if a.device=='cpu' and not a.full_width:
        with (a.output/'decoder-norm.txt').open('w') as f:
            f.write(f'S3D_DECODER_NORM_V1 {len(norm_cases)}\n')
            for b,n,d,*values in norm_cases:
                f.write(f'{b} {n} {d}\n')
                for v in values:f.write(' '.join(format(x,'.9g') for x in v.cpu().numpy().flat)+'\n')
    save_file(tensors,a.output/'upstream.safetensors');save_file(full,a.output/'full.safetensors')
    scope='original complete Body forward_decoder with real MHR and synthetic SAM state/features; no injected intermediate pose, camera, mesh or feedback; NOT trained SAM image inference'
    if a.image_pipeline:scope='original image transforms, streamed original full backbone and unchanged pose branch/decoder on official dancing RGB; synthetic SAM state plus real MHR; own intermediates, NOT learned image reconstruction or performance acceptance'
    if trained_cfg is not None:scope='trained F32 original Body pose branch from dancing RGB through DINOv3, no-mask embedding, six decoder/geometry feedback layers and hand boxes; no injected intermediates; NOT hand refinement or published BF16/performance acceptance'
    if a.hand_branch:scope='trained F32 original forward_decoder_hand with synthetic features/rays/CLIFF, real hand state and six own-intermediate MHR/camera/feedback layers; NOT hand-image or final refinement acceptance'
    if a.hand_branch and a.image_pipeline:scope='trained F32 original hand image branch from official dancing RGB and a fixed original-derived hand ROI, original shared DINOv3/no-mask embedding, six hand decoder/geometry/feedback layers and shared box heads; own intermediates after the supplied ROI; NOT body-to-hand crop inference, unmirroring, final refinement or performance acceptance'
    manifest={'schema_version':1,'scope':scope,
        'model_sha256':MODEL_SHA,'source_sha256':hashes,'selected_unchanged_ast_methods':selected,'roma_sha256':ROMA_SHA,'script_sha256':digest(Path(__file__)),
        'device':a.device,'torch':torch.__version__,'threads':a.threads,'sdpa_backend':'MATH','tf32':False,'full_width':a.full_width,'decoder_operations':a.decoder_operations,'hand_branch':a.hand_branch,'learned_sam_checkpoint_loaded':trained_cfg is not None,'trained_state':trained_metadata,'cases':cases,
        'artifacts':{f.name:digest(f) for f in sorted(a.output.iterdir()) if f.is_file() and f.name!='manifest.json'}}
    if helper_hashes!={name:digest(Path(__file__).with_name(name)) for name in helpers}:raise ValueError('reference helper changed during capture')
    manifest['reference_helper_sha256']=helper_hashes
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
