"""Exact public MHR v1.0.1 asset inference contract; names shortened for GGUF."""
ARCHITECTURE='sam3d.mhr.lod1'
ASSET_SHA='352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc'
RELEASE_SHA='e4f4f205cd87c0fa106577ba1de4fc763e4eb197c924461d2ef7e6944e9d6b94'
RELEASE_URL='https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip'
# canonical name: (original safetensors key, original dtype, GGML dtype, shape)
TENSORS={
 'identity.vectors':('character_torch.blend_shape.shape_vectors','F32',0,(45,18439,3)),
 'identity.base':('character_torch.blend_shape.base_shape','F32',0,(18439,3)),
 'expression.vectors':('face_expressions_model.shape_vectors','F32',0,(72,18439,3)),
 'pose.sparse.indices':('pose_correctives_model.pose_dirs_predictor.0.sparse_indices','I64',26,(2,53136)),
 'pose.sparse.weight':('pose_correctives_model.pose_dirs_predictor.0.sparse_weight','F32',0,(53136,)),
 'pose.dense.weight':('pose_correctives_model.pose_dirs_predictor.2.weight','F32',0,(55317,3000)),
 'skeleton.offsets':('character_torch.skeleton.joint_translation_offsets','F32',0,(127,3)),
 'skeleton.prerotations':('character_torch.skeleton.joint_prerotations','F32',0,(127,4)),
 'skeleton.prefix':('character_torch.skeleton.pmi','I64',26,(2,266)),
 'skeleton.parents':('character_torch.skeleton.joint_parents','I32',26,(127,)),
 'skin.inverse_bind':('character_torch.linear_blend_skinning.inverse_bind_pose','F32',0,(127,8)),
 'skin.joints':('character_torch.linear_blend_skinning.skin_indices_flattened','I32',26,(51337,)),
 'skin.weights':('character_torch.linear_blend_skinning.skin_weights_flattened','F32',0,(51337,)),
 'skin.vertices':('character_torch.linear_blend_skinning.vert_indices_flattened','I64',26,(51337,)),
 'mesh.faces':('character_torch.mesh.faces','I32',26,(36874,3)),
 'mesh.texcoords':('character_torch.mesh.texcoords','F32',0,(19455,2)),
 'mesh.texcoord_faces':('character_torch.mesh.texcoord_faces','I32',26,(36874,3)),
 'parameter.matrix':('character_torch.parameter_transform.parameter_transform','F32',0,(889,249)),
}
