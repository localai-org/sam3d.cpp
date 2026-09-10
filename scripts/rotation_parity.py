"""Well-conditioned Euler-coordinate comparison; never changes inference data.

Raw relative error divides by an angle norm that vanishes at identity. Original
CPU/CUDA controls demonstrate that this can reject normal elementary-function
rounding. Compare each radian coordinate's (sin, cos) unit-circle embedding for
relative L2, while still enforcing the original raw absolute-angle bound. This
is a coordinate-wise check, not a geodesic metric or permission to wrap outputs.
"""
import numpy as np

POLICY = 'euler-radians-unit-circle-v1'

def compare_euler(reference, candidate):
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 3 or reference.shape[0] == 0:
        raise ValueError('Euler tensors must have matching nonempty B,3 shapes')
    if reference.dtype != np.float32 or candidate.dtype != np.float32:
        raise ValueError('Euler tensors must be F32')
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError('non-finite Euler tensor')
    r, v = reference.astype(np.float64), candidate.astype(np.float64)
    delta = v-r
    maximum = float(np.max(np.abs(delta)))
    raw_relative = float(np.linalg.norm(delta)/max(np.linalg.norm(r), 1e-12))
    ref_circle = np.stack((np.sin(r), np.cos(r)), axis=-1)
    got_circle = np.stack((np.sin(v), np.cos(v)), axis=-1)
    relative = float(np.linalg.norm(got_circle-ref_circle)/np.linalg.norm(ref_circle))
    return {'policy': POLICY, 'max_abs': maximum, 'max_abs_limit': 1e-4,
            'raw_euler_relative_l2': raw_relative,
            'relative_l2': relative, 'relative_l2_representation': 'per-coordinate sin/cos, evaluated in F64',
            'relative_l2_limit': 2e-5, 'pass': maximum <= 1e-4 and relative <= 2e-5}
