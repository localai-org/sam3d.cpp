# Live SAM3D to G1 through SONIC

Investigation: 2026-09-18. Local SAM3D revision `0aa27dc`, sibling
motion-bricks.cpp revision `587ba93`. This is an interface feasibility review,
not a tested bridge or a robot-control result.

SAM3D has the skeleton information needed to attempt this. The preferred first
experiment is:

```text
live camera -> resident SAM3D Body -> MHR-to-SMPL joint adapter
            -> SONIC ZMQ v3 -> upstream SMPL encoder + G1 policy -> MuJoCo
```

GEM-X is not required for that experiment. Whether SAM3D gives better control
than GEM-X remains unmeasured: removing temporal estimation can reduce latency
but also increase jitter and inconsistent lower-body motion.

## Existing interfaces

- [Body output](API.md#result-schema-and-coordinates) includes 127 joint
  positions, global rotations and global transforms. It already contains more
  than a point skeleton; no orientation reconstruction from pixels is required.
- Prefer `joint_transforms [1,127,8]`: original MHR Y-up, translation in
  centimetres, XYZW quaternion, uniform scale. Public `joints` instead have metre
  units and the `diag(1,-1,-1)` axis conversion. Do not mix those conventions.
- [Skeleton export](SKELETON-EXPORT.md) already preserves raw transforms and
  timestamps in 4,092-byte `.pose` files. Saved live takes can test conversion
  reproducibly before connecting a camera stream. GLB need not be an intermediate.
- `demo/tracking.go`, immediately after native result parsing/validation, is a
  suitable live integration point. Publish through a bounded latest-result
  queue outside the application lock. Browser presentation interpolation should
  not become the control input.

Upstream now has a [live GEM-X tutorial][live] and a [SOMA-to-SMPL bridge][bridge].
Our sibling GEM-X demo is offline, but that is not a limitation of the current
upstream integration. The bridge is useful as an interface reference even if
GEM-X is excluded from the estimator pipeline.

The upstream bridge sends selected human joint positions rather than running
a full SMPL mesh fit. This supports trying an MHR joint adapter first; it does
not prove that a name-based MHR mapping matches the encoder's training inputs.

## Packet and model contract

The upstream bridge publishes topic `pose`, default port 5556, protocol v3:

| Field | Shape for one frame | Meaning |
| --- | --- | --- |
| `smpl_joints` | `[1,24,3]` F32 | SMPL-ordered positions, metres, root-local |
| `body_quat` | `[1,4]` F32 | WXYZ root orientation in SONIC's convention |
| `smpl_pose` | `[1,21,3]` F32 | Zero in the upstream camera example |
| `joint_pos` | `[1,29]` F32 | Zero except optional six G1 wrist angles |
| `joint_vel` | `[1,29]` F32 | Zero in that example |
| `frame_index` | `[1]` I64 | Monotonically advancing sequence index |

Wire framing is topic bytes, a 1,280-byte padded JSON header and little-endian
binary arrays. Use the upstream packer for initial interoperability validation.
The source is the authority for exact field semantics and framing.

The local upstream deployment source explicitly maps protocols 2/3 to encoder
mode 2 (`smpl`). Its checked policy observation configuration consumes ten
SMPL joint samples, ten anchor orientations and ten wrist samples; it does not
use `smpl_pose` in that branch. This explains the camera example's zero poses,
but should not be generalized to arbitrary policy releases.

The local native port is a separate limitation:
`../motion-bricks.cpp/include/motionbricks/sonic.h` and its conversion script
support **G1 mode 0 only**. They cannot consume this SMPL stream by changing
the mode number. The original ONNX policy and observation config are already
present under `../motion-bricks.cpp/generated/sonic/policy/`, including the SMPL
mode declaration. Native support would require extracting its encoder branch,
implementing its observation packing and validating tokens/actions against the
same upstream model. Matching decoder identity is necessary when reusing tokens.

An alternative is MHR-to-G1 constrained IK followed by the existing mode-0
encoder. That adds robot retargeting and reference-velocity construction, so it
is a larger first experiment than using the existing learned human encoder.

## Mapping work and remaining uncertainties

The named MHR hierarchy is embedded in `demo/web/skeleton.json`. Anatomical
candidates are:

| SMPL landmarks | MHR candidates |
| --- | --- |
| pelvis | `root` (index 1), not artificial `body_world` |
| hips / knees | `l/r_upleg`, `l/r_lowleg` |
| ankles / feet | `l/r_foot`, `l/r_ball` |
| collar / shoulder / elbow / wrist | `l/r_clavicle`, `l/r_uparm`, `l/r_lowarm`, `l/r_wrist` |
| neck / head | `c_neck`, `c_head` |
| hand endpoints | candidate `l/r_middle1` |
| three SMPL spine landmarks | calibrate against MHR's four `c_spine0..3` levels |

These are candidates, not a validated index table. Spine locations, hand
endpoints, foot pivots and body proportions need geometric checks. Stabilizing
bone lengths or using a calibrated target skeleton may improve compatibility;
framewise MHR shape estimates should not make the robot's reference limbs grow
and shrink.

The upstream SOMA adapter centres joints on the pelvis, rotates Y-up to Z-up,
applies its SMPL root/base-orientation convention and removes that orientation
from the positions. MHR's root rest frame is not automatically the SMPL rest
frame. Derive a fixed alignment from canonical poses and check neutral, left
arm raise, knee bend and a quarter turn. Simply reordering an MHR quaternion
from XYZW to WXYZ is insufficient evidence of correctness. Keep quaternion
signs continuous when filtering.

This path carries local pose and root orientation, not a world-space root
trajectory. Camera translation should not be interpreted as commanded walking.
World-relative locomotion would additionally need floor/camera calibration,
root/contact estimation or a separate locomotion command path.

## Live performance and evaluation

[Existing live measurements](../reference/BODY_LIVE_PERFORMANCE.md) report
8.1 Hz and 176 ms median to first render submission; the latter is not measured
camera-to-robot latency. Warm native BF16 inference is about 86 ms. That supports
testing slow movements, but does not establish responsive whole-body control.

[Fast inference measurements](../reference/BODY_FAST_INFERENCE.md) reduce native
inference to about 55 ms with `fast384`, while moving some joints about 20 cm
relative to the default on the evaluated inputs. Start with the default BF16
512-pixel mode and evaluate speed variants only after the mapping is checked.
Those distances are estimator differences, not ground-truth errors.

Keep the controller running at its required rate independently of camera
inference. Preserve capture timestamps, drop superseded results and bound
filtering delay. Resampling does not create fresh observations. The checked
upstream SMPL observation builder looks forward in a reference sequence and
clamps unavailable samples to the last pose. A live stream must explicitly test
that repeated-pose horizon and receiver buffering; it cannot supply true future
motion. Include stale-input handling rather than assuming a socket timeout
stops tracking the last pose.

Recommended experiment:

1. Convert a recorded `.pose` take into SMPL-ordered positions and a debug
   skeleton view. Check axes, units, neutral heading, limb lengths and side
   labels independently of the robot policy.
2. Replay those frames through upstream v3 in MuJoCo. Begin with slow arm
   motion, then torso/knee movement. Record reference versus achieved motion,
   falls, foot slip, unwanted steps and output discontinuities.
3. Feed the same adapter directly from live native results. Measure capture to
   controller-consumption age (median and tail), dropped frames, controller
   overruns and stop/reacquisition behavior. Compare raw versus light causal
   filtering and fixed versus framewise body proportions.
4. Compare SAM3D and GEM-X on identical video if assessing motion quality.
   This investigation does not diagnose the poor GEM-X conversion result.
5. If the mapping works, convert the SMPL encoder for an all-native runtime.
   Keep initial physical-robot testing separate from proving simulation behavior.

No bridge, inference benchmark, simulation trial or hardware trial was run in
this investigation. Findings come from the local API, demo, model conversion,
deployment source/configuration and upstream documentation/source inspection.

[live]: https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/live_camera_teleop.html
[bridge]: https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/examples/live_camera_teleop/soma_to_smpl.py
