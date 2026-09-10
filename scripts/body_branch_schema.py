"""Pinned trained Body pose-branch contract, excluding hand-crop refinement.

Derived from the original model configuration/constructors; dimensions are not
inferred from an input file. Tensor axes retain original PyTorch order.
"""
from gguf_schema import BODY_REVISION,CHECKPOINT_SHA256,CONFIG_SHA256
ARCHITECTURE='sam3d.body.pose_branch'
SAFE_SHA='4c6b3f63ce8a050f6587cf833a036bad3f68377d86cfe69d591502c1373ba0a3'
MHR_SHA='352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc'
INTEGER_NAMES={'head_pose.hand_joint_idxs_left','head_pose.hand_joint_idxs_right','head_pose.faces'}

def shapes():
    d,c=1024,1280
    result={}
    def add(name,shape):
        if name in result:raise ValueError('duplicate schema name')
        result[name]=tuple(shape)
    def linear(name,inputs,outputs):add(name+'.weight',(outputs,inputs));add(name+'.bias',(outputs,))
    def norm(name,width):add(name+'.weight',(width,));add(name+'.bias',(width,))
    def ffn(name,inputs,outputs):linear(name+'.layers.0.0',inputs,d);linear(name+'.layers.1',d,outputs)
    add('init_pose.weight',(1,519));add('init_camera.weight',(1,3))
    linear('init_to_token_mhr',525,d);linear('prev_to_token_mhr',522,d);linear('prompt_to_token',c,d)
    for name in ['keypoint_embedding','keypoint3d_embedding']:add(name+'.weight',(70,d))
    add('hand_box_embedding.weight',(2,d))
    add('ray_cond_emb.conv.weight',(c,c+99,1,1));norm('ray_cond_emb.norm',c)
    add('prompt_encoder.pe_layer.positional_encoding_gaussian_matrix',(2,c//2))
    for name in ['invalid_point_embed','not_a_point_embed','no_mask_embed']:add('prompt_encoder.'+name+'.weight',(1,c))
    for i in range(70):add(f'prompt_encoder.point_embeddings.{i}.weight',(1,c))
    for i in range(6):
        prefix=f'decoder.layers.{i}.'
        for name,width in [('ln_pe_1',d),('ln_pe_2',c),('ln1',d),('ln2_1',d),('ln2_2',c),('ln3',d)]:norm(prefix+name,width)
        for name,width in [('self_attn',d),('cross_attn',c)]:
            linear(prefix+name+'.q_proj',d,512);linear(prefix+name+'.k_proj',width,512)
            linear(prefix+name+'.v_proj',width,512);linear(prefix+name+'.proj',512,d)
        ffn(prefix+'ffn',d,d)
    norm('decoder.norm_final',d)
    ffn('head_pose.proj',d,519);ffn('head_camera.proj',d,3)
    for name,shape in [('scale_mean',(68,)),('scale_comps',(28,68)),('hand_pose_mean',(54,)),('hand_pose_comps',(54,54)),
                       ('keypoint_mapping',(308,18566)),('hand_joint_idxs_left',(27,)),('hand_joint_idxs_right',(27,)),('faces',(36874,3))]:add('head_pose.'+name,shape)
    ffn('keypoint_posemb_linear',2,d);ffn('keypoint3d_posemb_linear',3,d);linear('keypoint_feat_linear',c,d)
    for i in range(3):linear(f'bbox_embed.layers.{i}',d,4 if i==2 else d)
    linear('hand_cls_embed',d,2)
    return result

def metadata():
    return {'general.architecture':ARCHITECTURE,'general.alignment':32,'sam3d.schema_version':1,'sam3d.component':'body.pose_branch',
            'sam3d.source.body_repository':'https://github.com/facebookresearch/sam-3d-body','sam3d.source.body_revision':BODY_REVISION,
            'sam3d.source.checkpoint_sha256':CHECKPOINT_SHA256,'sam3d.source.config_sha256':CONFIG_SHA256,'sam3d.source.safetensors_sha256':SAFE_SHA,
            'sam3d.converted.precision':'F32,I32','sam3d.tensor_layout':'torch_contiguous; ggml_dimensions_reversed; no_data_transpose',
            'sam3d.decoder':'depth=6;dim=1024;heads=8;head_dim=64;mlp=1024;repeat_pe=1;twoway=0;norm_eps=1e-6',
            'sam3d.condition':'cliff_intrinsics_center;keypoints70;keypoints3d70;hand_tokens;no_mask_embedding',
            'sam3d.required_backbone':'sam3d.body.dinov3.vith16plus','sam3d.required_mhr_sha256':MHR_SHA,
            'sam3d.output':'body_vertices_m;keypoints70;camera;hand_boxes;not_hand_refinement',
            'sam3d.source.precision':'F32,I64'}
