#!/usr/bin/env python3
"""Original crop + 2D/3D keypoint feedback methods with synthetic state/pose inputs."""
import argparse,ast,hashlib,importlib,json,struct,sys,types
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from safetensors.numpy import save_file
from yacs.config import CfgNode
from capture_camera_encoder import HASHES


class Observe(TorchDispatchMode):
    def __init__(self,taps):super().__init__();self.taps=taps;self.crop=True
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        value=func(*args,**(kwargs or {}))
        if self.crop and func==torch.ops.aten.cat.default:self.taps['00.homogeneous']=value.detach().clone()
        if self.crop and func==torch.ops.aten.bmm.default:self.taps['01.crop_pixels']=value.detach().clone()
        if func==torch.ops.aten.grid_sampler_2d.default:self.taps['14.sampled']=value.squeeze(3).permute(0,2,1).detach().clone()
        return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--small-regression',action='store_true');args=p.parse_args()
    args.upstream=args.upstream.resolve();root=args.upstream/'sam_3d_body'
    hashes={f'models/modules/{k}':v for k,v in HASHES.items()}
    hashes|={'models/meta_arch/sam3d_body.py':'851b7475f18b56891aa02606e7c0ee9e03120fa208cc85df5127b792e1abfeee',
        'models/meta_arch/base_model.py':'baf4c93ab865e6e9f4f498056a673698e59bafe89b17f969833884a3b352e8bc'}
    for name,digest in hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'source changed: {name}')
    for name in ['sam_3d_body','sam_3d_body.models','sam_3d_body.models.modules']:
        mod=types.ModuleType(name);mod.__path__=[str(args.upstream/Path(*name.split('.')))];sys.modules[name]=mod
    FFN=importlib.import_module('sam_3d_body.models.modules.transformer').FFN
    namespace={'torch':torch,'F':F,'Dict':Dict};selected={}
    for filename,classname,names in [('sam3d_body.py','SAM3DBody',['_full_to_crop','keypoint_token_update_fn','keypoint3d_token_update_fn']),('base_model.py','BaseModel',['_flatten_person'])]:
        source=root/'models/meta_arch'/filename;tree=ast.parse(source.read_text(),filename=str(source));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
        nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
        if len(nodes)!=len(names) or any(n.decorator_list for n in nodes):raise ValueError('unexpected method definitions')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace);selected[filename]=names
    torch.set_num_threads(1);torch.manual_seed(8605);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;rng=np.random.default_rng(8605)
    # B,T,D,C,H,W,J,K2,K3,start2,start3,hip0,hip1,layer,depth.
    shapes=[(2,16,8,8,3,5,9,6,4,2,9,1,2,0,6),(1,10,8,4,1,1,7,7,0,1,0,0,1,2,6),
        (2,16,8,8,4,4,9,6,4,2,9,1,2,5,6),(1,16,16,8,5,3,9,6,4,2,9,1,2,1,6)]
    if not args.small_regression:shapes.append((1,143,1024,1280,32,32,70,70,70,3,73,9,10,0,6))
    args.output.mkdir(parents=True,exist_ok=True);cases,tensors,rules=[],{},[];lines=['S3D_FEEDBACK_REGRESSION_V1',str(len(shapes))]
    for index,shape in enumerate(shapes):
        b,t,d,c,h,w,j,k,k3,start,start3,hip0,hip1,layer,depth=shape
        holder=torch.nn.Module();holder.cfg=CfgNode({'MODEL':{'BACKBONE':{'TYPE':'dinov3'}}});holder.decoder=types.SimpleNamespace(layers=[None]*depth)
        holder._batch_size=b;holder._max_num_person=1;holder._flatten_person=types.MethodType(namespace['_flatten_person'],holder)
        holder.pelvis_idx=[hip0,hip1];idx2=np.roll(np.arange(j,dtype=np.int32),1)[:k].copy();idx3=np.arange(j,dtype=np.int32)[::-1][:k3].copy()
        holder.keypoint_embedding=torch.nn.Embedding(k,d);holder.keypoint_embedding_idxs=idx2.tolist()
        holder.keypoint_posemb_linear=FFN(embed_dims=2,feedforward_channels=d,output_dims=d,num_fcs=2,add_identity=False)
        holder.keypoint_feat_linear=torch.nn.Linear(c,d)
        if k3:
            holder.keypoint3d_embedding=torch.nn.Embedding(k3,d);holder.keypoint3d_embedding_idxs=idx3.tolist()
            holder.keypoint3d_posemb_linear=FFN(embed_dims=3,feedforward_channels=d,output_dims=d,num_fcs=2,add_identity=False)
        holder=holder.float().to(args.device).eval()
        def random(shape,scale=1):return torch.from_numpy(rng.normal(scale=scale,size=shape).astype(np.float32)).to(args.device)
        with torch.no_grad():
            for name,v in holder.named_parameters():v.copy_(random(tuple(v.shape),1/np.sqrt(v.shape[1]) if v.ndim==2 else .1))
            tokens=random((b,t,d));augment=random((b,t,d));image=random((b,c,h,w));world=random((b,j,3))
            locations=rng.uniform(-.45,.45,(b,j,2)).astype(np.float32)
            edge=np.array([[-.5,-.5],[.5,.5],[0,0],[.50001,0],[0,-.50001],[.49,.3],[-.3,.2]],dtype=np.float32)
            locations[:,:min(j,7)]=edge[:min(j,7)];pixels=torch.from_numpy((locations+.5)*512).to(args.device)
            depths=torch.ones((b,j),device=args.device);depths[:,2]=1e-5;depths[:,3]=float(np.nextafter(np.float32(1e-5),np.float32(-np.inf)));depths[:,4]=-1
            affine=torch.tensor([[[1.,0,0],[0,1.,0]]]*b,device=args.device);size=torch.tensor([[512.,512.]]*b,device=args.device)
            if index==3:affine=torch.tensor([[[.8,-.2,32],[.2,.8,-16]]],device=args.device);size=torch.tensor([[384.,512.]],device=args.device)
            batch={'affine_trans':affine[:,None],'img_size':size[:,None]}
            def run(observer=None):
                crop=namespace['_full_to_crop'](holder,batch,pixels)
                if observer is not None:observer.crop=False;observer.taps['02.crop_points']=crop.detach().clone()
                pose={'pred_keypoints_2d_cropped':crop,'pred_keypoints_2d_depth':depths,'pred_keypoints_3d':world}
                value=namespace['keypoint_token_update_fn'](holder,start,image,tokens,augment,pose,layer)
                if k3:value=namespace['keypoint3d_token_update_fn'](holder,start3,*value)
                return value[0],value[1]
            baseline=tuple(v.clone() for v in run());taps={};hooks=[]
            def capture(name,v):taps[name]=v.detach().clone()
            if layer+1<depth:
                def ffn(module,prefix):
                    for sub,name in [(module.layers[0][0],'.0.linear'),(module.layers[0][1],'.1.relu'),(module.layers[1],'.2.output')]:
                        hooks.append(sub.register_forward_hook(lambda _m,_a,v,name=name:capture(prefix+name,v)))
                ffn(holder.keypoint_posemb_linear,'20.pose2d')
                if k3:ffn(holder.keypoint3d_posemb_linear,'31.pose3d')
                hooks.append(holder.keypoint_feat_linear.register_forward_pre_hook(lambda _m,a:capture('15.masked_features',a[0])))
                hooks.append(holder.keypoint_feat_linear.register_forward_hook(lambda _m,_a,v:capture('21.feature_update',v)))
            def profile(frame,event,value):
                if event!='return' or frame.f_code.co_filename!=str(root/'models/meta_arch/sam3d_body.py'):return
                local=frame.f_locals
                if frame.f_code.co_name=='keypoint_token_update_fn' and 'invalid_mask' in local:
                    for name,key in [('10.selected_2d','pred_keypoints_2d_cropped'),('11.selected_depth','pred_keypoints_2d_depth'),('13.grid','pred_keypoints_2d_cropped_sample_points')]:capture(name,local[key])
                    capture('12.invalid',local['invalid_mask'].float())
                if frame.f_code.co_name=='keypoint3d_token_update_fn' and 'pred_keypoints_3d' in local:capture('30.relative_3d',local['pred_keypoints_3d'])
            old=sys.getprofile();observer=Observe(taps)
            try:
                sys.setprofile(profile)
                with observer:observed=run(observer)
            finally:sys.setprofile(old)
            for hook in hooks:hook.remove()
            if not all(torch.equal(x,y) for x,y in zip(baseline,observed)):raise ValueError('feedback observation changed output')
            capture('90.tokens',observed[0]);capture('91.augment',observed[1])
        inputs={'image':image,'tokens':tokens,'augment':augment,'pixels':pixels,'depths':depths,'world':world,'affine':affine,'crop_size':size}
        state={name:v for name,v in holder.state_dict().items() if 'linear.' in name}
        prefix=f'case.{index:04d}';filename=prefix+'.input'
        with (args.output/filename).open('wb') as stream:
            stream.write(b'S3DFBK01'+struct.pack('<15I',*shape)+idx2.astype('<i4').tobytes()+idx3.astype('<i4').tobytes())
            for name in ['image','tokens','augment','pixels','depths','world','affine','crop_size']:stream.write(inputs[name].cpu().numpy().astype('<f4').tobytes())
            for name in sorted(state):stream.write(state[name].cpu().numpy().astype('<f4').tobytes())
        if args.small_regression:
            lines.append(' '.join(map(str,shape)));lines.append(' '.join(map(str,idx2)));lines.append(' '.join(map(str,idx3)))
            for group in [inputs,state,taps]:
                lines.append(str(len(group)))
                for name,v in sorted(group.items()):
                    a=v.cpu().numpy().reshape(-1);lines.append(f'{name} {a.size}')
                    for start in range(0,len(a),8):lines.append(' '.join(format(float(v),'.9g') for v in a[start:start+8]))
        order=sorted(taps);cases.append({'prefix':prefix,'input':filename,'order':order,'shapes':{k:list(v.shape) for k,v in taps.items()}})
        for name in order:
            key=prefix+'.'+name;tensors[key]=taps[name].cpu().numpy().copy()
            exact=name in ['00.homogeneous','11.selected_depth','12.invalid'] or layer+1==depth and name in ['90.tokens','91.augment']
            rules.append({'name':key,'mode':'exact'} if exact else {'name':key,'mode':'float','max_abs':1e-3 if name=='01.crop_pixels' else 1e-4,'relative_l2':2e-5,'zero_reference_floor':1e-12})
        print(f'captured {prefix}, B={b}, D={d}, K={k}/{k3}, last={layer+1==depth}, taps={len(taps)}',flush=True)
    if args.small_regression:(args.output/'regression.txt').write_text('\n'.join(lines)+'\n')
    save_file(tensors,args.output/'upstream.safetensors')
    boundary='original full_to_crop and 2D/3D keypoint feedback method ASTs, original FFNs/grid_sample; synthetic pose/state inputs, not complete pose geometry or decoder parity'
    (args.output/'rules.json').write_text(json.dumps({'schema_version':1,'boundary':boundary,'tensors':rules},indent=2)+'\n')
    manifest={'boundary':boundary,'revision':'b5c765a0d89d789985e186d396315e7590887b94','source_hashes':hashes,'selected_methods':selected,
        'method_changes':'none; unchanged AST bodies including original BaseModel flatten','capture_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'torch':torch.__version__,'numpy':np.__version__,'device':args.device,'threads':1,'tf32_matmul':False,'tf32_cudnn':False,
        'cuda_device':torch.cuda.get_device_name() if args.device=='cuda' else None,'unobserved_vs_observed':'exact_equal_all_cases',
        'cases':cases,'artifacts':{}}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name!='manifest.json':manifest['artifacts'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
