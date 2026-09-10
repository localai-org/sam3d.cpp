#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file

parser=argparse.ArgumentParser(description='Pack native RGB/CHW image diagnostic output')
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
lines=(args.input/'results.txt').read_text().splitlines()
if not lines or lines[0]!='SAM3D_IMAGE_RESULTS_V1': raise ValueError('bad result header')
tensors={}
for line in lines[1:]:
    index,w,h=map(int,line.split())
    prefix=f'case.{index:04d}'
    if index<0 or w<=0 or h<=0 or prefix+'.rgb' in tensors: raise ValueError('bad/duplicate case')
    tensors[prefix+'.rgb']=np.fromfile(args.input/(prefix+'.rgb'),dtype=np.uint8).reshape(h,w,3)
    tensors[prefix+'.normalized']=np.fromfile(args.input/(prefix+'.normalized.f32'),dtype='<f4').reshape(3,h,w)
if not tensors: raise ValueError('empty result')
save_file(tensors,args.output)
