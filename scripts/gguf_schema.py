"""Versioned conversion contracts; not a model implementation or checkpoint oracle."""
ARCHITECTURE = 'sam3d.body.dinov3.vith16plus'
BODY_REVISION = 'b5c765a0d89d789985e186d396315e7590887b94'
DINO_REVISION = '6876159a11b4df116f30f667f8c9888617df0751'
CHECKPOINT_SHA256 = 'b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf'
CONFIG_SHA256 = '1012fc3f39cb5e90e3f8fbadf7bded31604bfafdce0321d17a7c1a2d3f08b88d'


def block_shapes(dim, hidden):
    return {
        'norm1.weight':(dim,), 'norm1.bias':(dim,),
        'attn.qkv.weight':(3*dim,dim), 'attn.qkv.bias':(3*dim,),
        'attn.qkv.bias_mask':(3*dim,), 'attn.proj.weight':(dim,dim),
        'attn.proj.bias':(dim,), 'ls1.gamma':(dim,),
        'norm2.weight':(dim,), 'norm2.bias':(dim,),
        'mlp.w1.weight':(hidden,dim), 'mlp.w1.bias':(hidden,),
        'mlp.w2.weight':(hidden,dim), 'mlp.w2.bias':(hidden,),
        'mlp.w3.weight':(dim,hidden), 'mlp.w3.bias':(dim,), 'ls2.gamma':(dim,),
    }


def body_dino_shapes():
    dim, hidden = 1280, 5120
    shapes = {'cls_token':(1,1,dim),'storage_tokens':(1,4,dim),
              'mask_token':(1,dim),'patch_embed.proj.weight':(dim,3,16,16),
              'patch_embed.proj.bias':(dim,),'rope_embed.periods':(16,),
              'norm.weight':(dim,),'norm.bias':(dim,)}
    for index in range(32):
        shapes.update({f'blocks.{index}.{key}':shape for key,shape in block_shapes(dim,hidden).items()})
    return shapes
