#!/usr/bin/env python3
"""Invoke actual upstream crop transforms and BaseModel normalization, no weights.

Package namespace paths avoid unrelated root __init__ model/detector imports.
No function/class is replaced or reimplemented; normalization is invoked as the
original unbound method with only its mean/std inputs on a lightweight holder.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import cv2
import numpy as np
import torch
import torchvision
from safetensors.numpy import save_file

EXPECTED = {
    "data/transforms/common.py": "165697629df2fe23a62bb90daa2ac8a0f67cac09d70f36844b3c089a9af5a45f",
    "data/transforms/bbox_utils.py": "0f49a3f857a09f98d1fd9c609df42f875899515f5e10a0ea6b25acd7a9197d79",
    "models/meta_arch/base_model.py": "baf4c93ab865e6e9f4f498056a673698e59bafe89b17f969833884a3b352e8bc",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.upstream/"sam_3d_body"
    for relative, expected in EXPECTED.items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"upstream source hash mismatch: {relative}")
    for name in ["sam_3d_body", "sam_3d_body.data", "sam_3d_body.models",
                 "sam_3d_body.models.meta_arch", "sam_3d_body.models.optim"]:
        package = types.ModuleType(name)
        package.__path__ = [str(args.upstream/Path(*name.split('.')))]
        sys.modules[name] = package
    transforms = importlib.import_module("sam_3d_body.data.transforms.common")
    BaseModel = importlib.import_module("sam_3d_body.models.meta_arch.base_model").BaseModel
    holder = types.SimpleNamespace(image_mean=torch.tensor([.485,.456,.406]).view(-1,1,1),
                                   image_std=torch.tensor([.229,.224,.225]).view(-1,1,1))
    rng = np.random.default_rng(724)
    yy, xx = np.indices((81, 127))
    pattern = np.stack([(xx*31+yy*7)%256,(xx*3+yy*53)%256,(xx^yy)%256],axis=-1).astype(np.uint8)
    images = [pattern, rng.integers(0,256,(79,113,3),dtype=np.uint8),
              np.zeros((33,31,3),dtype=np.uint8),np.ones((33,31,3),dtype=np.uint8),
              np.full((35,37,3),255,dtype=np.uint8)]
    photo = args.upstream/"notebook/images/dancing.jpg"
    images.append(cv2.cvtColor(cv2.imread(str(photo)),cv2.COLOR_BGR2RGB))
    outputs = [(512,512),(93,111),(32,32),(32,32),(65,63),(512,512)]
    tensors, cases = {}, []
    rules = {"schema_version":1,"boundary":"actual Body RGB crop/ToTensor/normalization only; no model inference",
             "tensors":[]}
    args.output.mkdir(parents=True,exist_ok=True)
    lines = ["SAM3D_IMAGE_CASES_V1"]
    geometry_lines = ["SAM3D_CROP_CASES_V1"]
    for index in range(12):
        image = images[index%len(images)]
        h,w = image.shape[:2]
        ow,oh = outputs[index%len(images)]
        padding = float(np.float32(.9 if index%3==1 else 1.25))
        rotation = float(np.float32(17.5 if index>=6 else 0))
        box = np.asarray([-.35*w,-.1*h,1.12*w,.97*h] if index>=6 else [0,0,w,h],dtype=np.float32)
        data = {"img":image.copy(),"bbox":box,"bbox_format":"xyxy","bbox_rotation":rotation}
        data = transforms.GetBBoxCenterScale(padding=padding)(data)
        data = transforms.TopdownAffine(input_size=(ow,oh),use_udp=False)(data)
        cropped = data['img'].copy()
        unit = torchvision.transforms.ToTensor()(cropped)
        normalized = BaseModel.data_preprocess(holder,unit[None],crop_width=False)[0].numpy()
        prefix=f"case.{index:04d}"
        tensors[prefix+'.rgb']=cropped
        tensors[prefix+'.normalized']=np.ascontiguousarray(normalized)
        rules['tensors'].append({"name":prefix+'.rgb',"mode":"exact"})
        rules['tensors'].append({"name":prefix+'.normalized',"mode":"float","max_abs":1e-6,
                                "relative_l2":1e-6,"zero_reference_floor":1e-12})
        # Include row padding to exercise native stride validation/data addressing.
        stride=w*3+7
        raw=np.full((h,stride),193,dtype=np.uint8); raw[:,:w*3]=image.reshape(h,w*3)
        raw.tofile(args.output/(prefix+'.input.rgb'))
        row=[index,w,h,stride,ow,oh,padding,rotation,*box.tolist()]
        lines.append(' '.join(map(str,row)))
        geometry_lines.append(' '.join(map(str,[index,ow,oh,padding,.75,rotation,*box.tolist()])))
        cases.append({"id":index,"input_shape":list(image.shape),"output_size":[ow,oh],
                      "padding":padding,"rotation":rotation,"box":box.tolist(),
                      "affine":data['affine_trans'].tolist()})
    save_file(tensors,args.output/'upstream.safetensors')
    (args.output/'cases.txt').write_text('\n'.join(lines)+'\n')
    (args.output/'crop-cases.txt').write_text('\n'.join(geometry_lines)+'\n')
    (args.output/'rules.json').write_text(json.dumps(rules,indent=2)+'\n')
    manifest={"boundary":rules['boundary'],"oracle":"original TopdownAffine and BaseModel.data_preprocess methods",
              "upstream_revision":"b5c765a0d89d789985e186d396315e7590887b94",
              "source_hashes":EXPECTED,"torch":torch.__version__,"torchvision":torchvision.__version__,
              "numpy":np.__version__,"opencv":cv2.__version__,"opencv_optimized":cv2.useOptimized(),
              "photo_sha256":hashlib.sha256(photo.read_bytes()).hexdigest(),"cases":cases,
              "artifacts":{}}
    for file in sorted(args.output.iterdir()):
        if file.name=='manifest.json': continue
        manifest['artifacts'][file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f"Captured {len(cases)} cases, {len(tensors)} tensors using actual upstream transforms")


if __name__=='__main__': main()
