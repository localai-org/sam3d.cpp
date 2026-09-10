"""Original Objects SSI + PyTorch3D ASTs; execute only in isolated reference.

Method bodies and numerical expressions are unchanged. Postponed annotations
avoid importing unused pose/rotation classes; they do not change execution.
"""
import __future__,ast,hashlib,logging,sys,types
from collections import namedtuple
from pathlib import Path
from typing import Dict,List,Optional,Union
import torch
import torchvision

P3D_REV='75ebeeaea0908c5527e7b1e305fbc7681382db47'
HASHES={
 'pytorch3d/transforms/transform3d.py':'b8fdd5c9aa5dcc89b41994984472d6117cc9ccf49f8b225334197f2a373dfd84',
 'pytorch3d/common/datatypes.py':'1e1f48bea108b715b4dfcbc0b79c915bd3e78fbe4768645c52cc2fb3919aeb6e',
 'sam3d_objects/data/dataset/tdfy/pose_target.py':'d56de3ea82f9c16363ba5d101fe487183bdef30ecb948fbe0f529282c7888ec3',
 'sam3d_objects/data/dataset/tdfy/img_and_mask_transforms.py':'e5ecdb46c12568a1d91e09c83842a16afaca8d1e22d73f49f0b9ded62cc3cc26',
 'sam3d_objects/model/backbone/dit/embedder/point_remapper.py':'fae4708d051a7f11c398252643e550daea69dd21873440d870533fddc11540c0',
}
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load_original(objects,p3d):
 ns={'torch':torch,'torchvision':torchvision,'nn':torch.nn,'F':torch.nn.functional,
     'Dict':Dict,'List':List,'Union':Union,'Optional':Optional,'namedtuple':namedtuple,
     'logger':logging.getLogger('original-ssi')}
 def execute(key,names):
  path=Path(p3d if key.startswith('pytorch3d/') else objects)/key
  if digest(path)!=HASHES[key]:raise ValueError('unverified original source '+key)
  nodes=[]
  for n in ast.parse(path.read_text()).body:
   name=n.name if isinstance(n,(ast.FunctionDef,ast.ClassDef)) else n.targets[0].id if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) else None
   if name in names:nodes.append(n)
  if len(nodes)!=len(names):raise ValueError('missing original definitions')
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),ns)
 execute('pytorch3d/common/datatypes.py',['Device','make_device','get_device'])
 execute('pytorch3d/transforms/transform3d.py',['Transform3d','Translate','Scale','_handle_coord','_handle_input','_broadcast_bmm'])
 execute('sam3d_objects/data/dataset/tdfy/pose_target.py',['PoseTargetConvention','ScaleShiftInvariant'])
 name='sam3d_objects.data.dataset.tdfy.pose_target';module=types.ModuleType(name)
 module.ScaleShiftInvariant=ns['ScaleShiftInvariant'];sys.modules[name]=module
 execute('sam3d_objects/model/backbone/dit/embedder/point_remapper.py',['PointRemapper'])
 execute('sam3d_objects/data/dataset/tdfy/img_and_mask_transforms.py',[
  'SSINormalizedPointmap','SSIPointmapNormalizer','ObjectCentricSSI','ObjectApparentSizeSSI',
  'NormalizedDisparitySpaceSSI','normalize_pointmap_ssi','_apply_metric_to_ssi'])
 return ns
