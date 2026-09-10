"""Hash-verified original PointPatchEmbed and pinned timm class definitions.

AST extraction avoids unrelated timm model registries/imports and never changes
method bodies. Only eval/no stochastic dropout is supported by the capture.
Execute only in the reviewed offline reference container, not on the host.
"""
import ast,collections,hashlib,logging,zipfile
from functools import partial
from itertools import repeat
from pathlib import Path
from typing import Final,Optional
import torch

WHEEL_SHA='bf5704014476ab011589d3c14172ee4c901fd18f9110a928019cac5be2945914'
TIMM_HASHES={
 'timm/models/vision_transformer.py':'8dc607d12266dbe12c0b35e2e9832cf132b9ad052eb2f2ae50d402fa07f88ba3',
 'timm/layers/mlp.py':'daf34147a30be416d2d70ae5784baa99567ba511af3da659c27dab58f055f8bb',
 'timm/layers/helpers.py':'f552ea203f238ddc3f527dbbd05deb40bbcdcfdbd032137d9adb3b924fcc6bf6',
 'timm/layers/config.py':'ff5d6394c57d703c3efb0d7348ac4778920d39feb437a203aa7b349b680f1adb',
}
OBJECT_HASHES={
 'point_remapper.py':'fae4708d051a7f11c398252643e550daea69dd21873440d870533fddc11540c0',
 'pointmap.py':'d0bdcbefdd9403b1fb9fb7b0e9b9b118fb6239bbf3e11028fa6284d6300cb45b',
}
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load_original(upstream,wheel):
 if digest(wheel)!=WHEEL_SHA:raise ValueError('unverified timm wheel')
 ns={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'Final':Final,'Optional':Optional,
     'partial':partial,'repeat':repeat,'collections':collections,'logger':logging.getLogger('original-pointpatch')}
 def execute(source,path,names):
  nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
  if len(nodes)!=len(names):raise ValueError('missing original class/function')
  exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
 with zipfile.ZipFile(wheel) as z:
  sources={name:z.read(name).decode() for name in TIMM_HASHES}
  for name,source in sources.items():
   if hashlib.sha256(source.encode()).hexdigest()!=TIMM_HASHES[name]:raise ValueError('unverified timm source')
  execute(sources['timm/layers/helpers.py'],'timm/layers/helpers.py',['_ntuple']);ns['to_2tuple']=ns['_ntuple'](2)
  # Pinned timm config defaults on PyTorch >=2; invoke its original selector.
  ns.update(_HAS_FUSED_ATTN=hasattr(torch.nn.functional,'scaled_dot_product_attention'),_EXPORTABLE=False,_USE_FUSED_ATTN=1)
  execute(sources['timm/layers/config.py'],'timm/layers/config.py',['use_fused_attn'])
  execute(sources['timm/layers/mlp.py'],'timm/layers/mlp.py',['Mlp'])
  execute(sources['timm/models/vision_transformer.py'],'timm/models/vision_transformer.py',['Attention','Block'])
 root=Path(upstream)/'sam3d_objects/model/backbone/dit/embedder'
 for name,cls in [('point_remapper.py','PointRemapper'),('pointmap.py','PointPatchEmbed')]:
  if digest(root/name)!=OBJECT_HASHES[name]:raise ValueError('unverified Objects pointmap source')
  execute((root/name).read_text(),str(root/name),[cls])
 return ns['PointPatchEmbed'],ns['PointRemapper']
