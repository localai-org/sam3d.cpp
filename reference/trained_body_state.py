"""Verified trained Body branch state; safe tensors only, no random fallback."""
from pathlib import Path
import torch
from safetensors import safe_open
from yacs.config import CfgNode
from capture_mhr import digest

STATE_SHA = '4c6b3f63ce8a050f6587cf833a036bad3f68377d86cfe69d591502c1373ba0a3'
BACKBONE_SHA = '98afde8ac5c13c68f3b7c7baf7cc8fa525df901fe1cec44b57d018fba11d53cc'
CONFIG_SHA = '1012fc3f39cb5e90e3f8fbadf7bded31604bfafdce0321d17a7c1a2d3f08b88d'

def config(path):
    if digest(path) != CONFIG_SHA:
        raise ValueError('unverified trained Body configuration')
    return CfgNode.load_cfg(Path(path).read_text())

def load(holder, extraction):
    path = extraction/'body-other-state.safetensors'
    if digest(path) != STATE_SHA:
        raise ValueError('unverified trained Body state')
    # The reviewed real TorchScript MHR is already loaded separately. Every
    # other tensor belonging to this branch must exist in the safe checkpoint.
    selected = {k:v for k,v in holder.state_dict().items() if not k.startswith('head_pose.mhr.')}
    with safe_open(path, framework='pt') as safe, torch.no_grad():
        for name, target in selected.items():
            if name not in safe.keys():
                raise ValueError('missing learned tensor '+name)
            value = safe.get_tensor(name)
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError('learned tensor layout/dtype mismatch '+name)
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError('nonfinite learned tensor '+name)
            target.copy_(value)
            if not torch.equal(target.detach().cpu().contiguous().view(torch.uint8),value.contiguous().view(torch.uint8)):
                raise ValueError('learned tensor assignment changed bytes '+name)
    indices = torch.cat([holder.head_pose.hand_joint_idxs_left, holder.head_pose.hand_joint_idxs_right]).cpu().numpy().astype('<i4')
    return indices, dict(state_sha256=STATE_SHA, config_sha256=CONFIG_SHA,
                         extraction_sha256=digest(extraction/'extraction.json'),
                         loaded_names=sorted(selected), missing_initialized_tensors=[],
                         assignment_readback='all tensor bytes equal safe source',loader_sha256=digest(Path(__file__)))

def load_hand(holder, extraction):
    """Populate the actual hand-named original modules, with no body alias fallback."""
    path=extraction/'body-other-state.safetensors'
    if digest(path)!=STATE_SHA:raise ValueError('unverified trained hand state')
    selected={k:v for k,v in holder.state_dict().items() if not k.startswith('head_pose_hand.mhr.')}
    with safe_open(path,framework='pt') as safe,torch.no_grad():
        for name,target in selected.items():
            if name not in safe.keys():raise ValueError('missing learned hand tensor '+name)
            value=safe.get_tensor(name)
            if value.shape!=target.shape or value.dtype!=target.dtype:raise ValueError('hand layout mismatch '+name)
            if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('nonfinite hand state')
            target.copy_(value)
            if not torch.equal(target.detach().cpu().contiguous().view(torch.uint8),value.contiguous().view(torch.uint8)):
                raise ValueError('hand assignment changed bytes '+name)
    head=holder.head_pose_hand
    indices=torch.cat([head.hand_joint_idxs_left,head.hand_joint_idxs_right]).cpu().numpy().astype('<i4')
    if not torch.equal(head.faces,head.mhr.character_torch.mesh.faces):raise ValueError('hand topology differs from MHR')
    return indices,dict(state_sha256=STATE_SHA,config_sha256=CONFIG_SHA,extraction_sha256=digest(extraction/'extraction.json'),
                       loaded_names=sorted(selected),missing_initialized_tensors=[],assignment_readback='all tensor bytes equal safe source',loader_sha256=digest(Path(__file__)))
