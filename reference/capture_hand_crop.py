#!/usr/bin/env python3
"""Original _get_hand_box and hand transforms; component capture, not refinement."""
import argparse,ast,copy,importlib,json,struct,sys,types
from pathlib import Path
import cv2,numpy as np,torch
from torchvision.transforms import ToTensor
from safetensors import safe_open
from safetensors.numpy import save_file
from capture_mhr import digest
from capture_body_camera import HASHES
PHOTO_SHA='0112b0a32ea5860db6a6fc700804751528341a02bf07fed0329a0c2610f905d3'
ESTIMATOR_SHA='8bfa34316d82eabbe185c1cd64a77c47670785cdb0fa489b1481fb468266a145'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','trained-reference','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=a.upstream/'sam_3d_body';hashes=dict(HASHES,**{'sam_3d_body_estimator.py':ESTIMATOR_SHA})
    for name,sha in hashes.items():
        if digest(root/name)!=sha:raise ValueError('unverified original source '+name)
    photo=a.upstream/'notebook/images/dancing.jpg'
    if digest(photo)!=PHOTO_SHA:raise ValueError('unverified original photograph')
    manifest=json.loads((a.trained_reference/'manifest.json').read_text());path=a.trained_reference/'upstream.safetensors'
    if not manifest['learned_sam_checkpoint_loaded'] or digest(path)!=manifest['artifacts']['upstream.safetensors']:raise ValueError('unverified trained reference')
    with safe_open(path,framework='numpy') as safe:
        trained_boxes=safe.get_tensor('case.0000.branch.hand_box').copy()
        trained_affine=safe.get_tensor('case.0000.prepare.affine').reshape(2,3).copy()
    for name in ['sam_3d_body','sam_3d_body.data','sam_3d_body.data.utils','sam_3d_body.models','sam_3d_body.models.meta_arch','sam_3d_body.models.optim']:
        module=types.ModuleType(name);module.__path__=[str(a.upstream/Path(*name.split('.')))];sys.modules[name]=module
    transforms=importlib.import_module('sam_3d_body.data.transforms.common')
    prepare=importlib.import_module('sam_3d_body.data.utils.prepare_batch').prepare_batch
    Base=importlib.import_module('sam_3d_body.models.meta_arch.base_model').BaseModel
    source=root/'models/meta_arch/sam3d_body.py';tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SAM3DBody')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_get_hand_box')
    namespace={'np':np};exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
    # Execute the exact left flip/crop statements from the original full method.
    run=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run_inference')
    start=next(i for i,n in enumerate(run.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='flipped_img')
    end=next(i for i,n in enumerate(run.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='batch_lhand')
    left_statements=compile(ast.Module(body=run.body[start:end+1],type_ignores=[]),str(source),'exec')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rgb=cv2.cvtColor(cv2.imread(str(photo)),cv2.COLOR_BGR2RGB);height,width=rgb.shape[:2]
    focal=np.float32((height**2+width**2)**.5);camera=torch.tensor([[[focal,0,width/2],[0,focal,height/2],[0,0,1]]],dtype=torch.float32)
    yy,xx=np.indices((43,67));pattern=np.stack([(xx*31+yy*7)%256,(xx*3+yy*53)%256,(xx^yy)%256],-1).astype(np.uint8)
    cases=[(rgb,trained_boxes,trained_affine,512,camera),
           (pattern,np.array([[[.05,.7,.12,.21],[.95,.25,.2,.15]]],np.float32),np.array([[.8,0,-4],[0,.8,3]],np.float32),32,torch.tensor([[[70,0,32],[0,80,20],[0,0,1]]],dtype=torch.float32)),
           (pattern,np.array([[[.5,.5,.9,.8],[.45,.55,.95,.2]]],np.float32),np.array([[1.1,0,7],[0,1.1,-6]],np.float32),32,torch.tensor([[[70,0,32],[0,80,20],[0,0,1]]],dtype=torch.float32))]
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise FileExistsError('output must be empty')
    tensors={};rules=[];descriptors=[]
    for i,(img,boxes,affine,crop,cam) in enumerate(cases):
        holder=types.SimpleNamespace(cfg=types.SimpleNamespace(MODEL=types.SimpleNamespace(IMAGE_SIZE=(crop,crop))),image_mean=torch.tensor([.485,.456,.406]).view(3,1,1),image_std=torch.tensor([.229,.224,.225]).view(3,1,1))
        transform=transforms.Compose([transforms.GetBBoxCenterScale(padding=.9),transforms.TopdownAffine(input_size=(crop,crop)),transforms.VisionTransformWrapper(ToTensor())])
        def capture():
            batch={'affine_trans':torch.from_numpy(affine.copy())[None,None]}
            left,right=namespace['_get_hand_box'](holder,{'mhr':{'hand_box':torch.from_numpy(boxes.copy())}},batch)
            result={}
            for side,xyxy in [('left',left),('right',right)]:
                for key in ['center','scale']:result[side+'.'+key]=batch[side+'_'+key].reshape(-1).copy()
                result[side+'.full_xyxy']=xyxy.reshape(-1).copy()
            local=dict(img=img.copy(),left_xyxy=left.copy(),width=img.shape[1],transform_hand=transform,cam_int=cam.clone(),prepare_batch=prepare)
            exec(left_statements,{},local);batches=[local['batch_lhand'],prepare(img.copy(),transform,right.copy(),cam_int=cam.clone())]
            for side,b,xyxy in zip(['left','right'],batches,[local['left_xyxy'],right]):
                result[side+'.input_xyxy']=xyxy.reshape(-1).copy()
                result[side+'.normalized_rgb']=Base.data_preprocess(holder,b['img'].flatten(0,1),crop_width=False)[0].numpy().copy()
                for key,original in [('box_center','bbox_center'),('affine','affine_trans'),('image_size','ori_img_size'),('crop_size','img_size'),('intrinsics','cam_int')]:result[side+'.'+key]=b[original].numpy().reshape(-1).copy()
                result[side+'.box_size']=b['bbox_scale'].numpy().reshape(-1)[:1].copy()
            result['left.unflipped_center']=result['left.box_center'].copy();result['left.unflipped_center'][0]=img.shape[1]-result['left.unflipped_center'][0]-1
            return result
        reference=capture();repeat=capture()
        if any(reference[k].tobytes()!=repeat[k].tobytes() for k in reference):raise ValueError('nonrepeatable hand preparation')
        prefix=f'case.{i:04d}';filename=prefix+'.input';h,w=img.shape[:2];stride=w*3+7
        raw=np.full((h,stride),193,np.uint8);raw[:,:w*3]=img.reshape(h,w*3)
        intr=np.array([cam[0,0,0],cam[0,1,1],cam[0,0,2],cam[0,1,2]],np.float32)
        with (a.output/filename).open('xb') as f:
            f.write(b'S3DHCP01'+struct.pack('<4I',w,h,stride,crop)+affine.astype('<f4').tobytes()+boxes.astype('<f4').tobytes()+intr.astype('<f4').tobytes()+raw.tobytes())
        for key,value in reference.items():
            tensors[prefix+'.'+key]=np.ascontiguousarray(value,dtype=np.float32)
            rules.append(dict(name=prefix+'.'+key,mode='exact'))
        descriptors.append(dict(prefix=prefix,input=filename,order=sorted(reference),shapes={k:list(v.shape) for k,v in reference.items()},origin='trained official dancing hand boxes' if i==0 else 'synthetic boundary case'))
        print(prefix,'captured',len(reference),'tensors',flush=True)
    save_file(tensors,a.output/'upstream.safetensors')
    (a.output/'rules.json').write_text(json.dumps(dict(schema_version=1,boundary=__doc__,tensors=rules),indent=2)+'\n')
    record=dict(scope=__doc__,source_hashes=hashes,photo_sha256=PHOTO_SHA,trained_reference_sha256=digest(path),script_sha256=digest(Path(__file__)),
                oracle='unchanged _get_hand_box AST, original full-method left crop statements, prepare_batch and transforms; injected upstream hand boxes, no hand inference',
                repeated='all bytes exact across two executions',torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__,cases=descriptors)
    record['artifacts']={p.name:digest(p) for p in a.output.iterdir() if p.is_file()}
    (a.output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':main()
