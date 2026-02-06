# EZGripper IK Optimization Fix

This document describes the fix for the EZGripper fingertip closure issue in IK optimization.

---

## Problem Description

When running vis_denoise.py, test_diff_v3.py, or test_diff_v3_ce.py with ezgripper, the fingertip links (`left_ezgripper_finger_L2_1` and `left_ezgripper_finger_L2_2`) don't close properly during grasp - they remain in a straight/extended position.

---

## Root Cause

The L2 joints (`left_ezgripper_knuckle_L1_L2_1` and `left_ezgripper_knuckle_L1_L2_2`) are **pure-rotation joints** that rotate around the Y-axis:

1. **IK limitation**: The IK solver optimizes link **positions**, not rotations. For pure-rotation joints, changing the joint angle doesn't significantly change the link position (only the rotation changes).

2. **Same issue as LeapHand**: This is identical to the LeapHand fingertip joint problem. LeapHand fingertip joints rotate around Z-axis, while EZGripper L2 joints rotate around Y-axis.

### URDF Structure

```
left_ezgripper_palm_link
├─ left_ezgripper_finger_L1_1 (q_idx=6, revolute, axis=[0,1,0])
│  └─ left_ezgripper_finger_L2_1 (q_idx=7, revolute, axis=[0,1,0]) ← PURE ROTATION
│     └─ extra_virtual_link_1 (FIXED, offset xyz=0.01,0.01,0.01)
└─ left_ezgripper_finger_L1_2 (q_idx=8, mimic of 6)
   └─ left_ezgripper_finger_L2_2 (q_idx=9, mimic of 7) ← PURE ROTATION
      └─ extra_virtual_link_2 (FIXED, offset xyz=0.01,0.01,0.01)
```

### Why TIP_MAPPING Didn't Work

Initial attempt used TIP_MAPPING to map L2 links to extra_virtual_link_1/2. This failed because:
- `extra_virtual_link_1/2` are **NOT** in the point cloud data
- Point cloud only contains: `palm_link`, `L1_1`, `L1_2`, `L2_1`, `L2_2`
- IK solver cannot target links that don't exist in `target_links`

---

## Solution: FINGERTIP_JOINTS Approach

Extract L2 joint angles from the **SE3 rotation** in the diffusion output, similar to LeapHand fingertip handling.

### Key Difference: Joint Axis

| Robot | Joint Axis | Rotation Matrix | Angle Formula |
|-------|------------|-----------------|---------------|
| LeapHand | (0, 0, -1) | Rz(θ) | `atan2(R[1,0], R[0,0])` |
| EZGripper | (0, 1, 0) | Ry(θ) = [[cos,0,sin],[0,1,0],[-sin,0,cos]] | `atan2(R[0,2], R[2,2])` |

### Configuration

```python
FINGERTIP_JOINTS = {
    'leaphand': {...},
    'leaphand_graph_1': {...},
    'ezgripper': {
        ('left_ezgripper_finger_L1_1', 'left_ezgripper_finger_L2_1'): 7,
        ('left_ezgripper_finger_L1_2', 'left_ezgripper_finger_L2_2'): 9,
    },
}
```

### Updated extract_fingertip_joints()

```python
if is_ezgripper:
    # Y-axis rotation: R = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
    angle = torch.atan2(R_rel[:, 0, 2], R_rel[:, 2, 2])
    predict_q[:, q_idx] = angle
else:
    # LeapHand: Z-axis rotation
    angle = torch.atan2(R_rel[:, 1, 0], R_rel[:, 0, 0])
    predict_q[:, q_idx] = -angle  # Negate for axis (0,0,-1)
```

---

## Files Modified

| File | Change |
|------|--------|
| `test_diff_v3.py` | Added ezgripper to FINGERTIP_JOINTS, updated extract_fingertip_joints() for Y-axis |
| `test_diff_v3_ce.py` | Same changes |
| `vis_denoise.py` | Same changes |
| `docs/EZGRIPPER_IK_FIX.md` | This documentation |

---

## Verification

### Test with vis_denoise.py

```bash
# Clear cache first
rm outputs/vis_cache.pt

# Update config to use ezgripper
# In config/vis_denoise.yaml:
# dataset.robot_names: ['ezgripper']
# vis.embodiment: 'ezgripper'

python vis_denoise.py --config config/vis_denoise.yaml
```

### Test with test_diff_v3.py

```bash
python test_diff_v3.py --config config/test_diff_v3.yaml --hands ezgripper --gpu 0
```

### Test with test_diff_v3_ce.py

```bash
python test_diff_v3_ce.py --config config/test_diff_v3_ce.yaml --hands ezgripper --gpu 0
```

Expected: L2 fingertip links should now close properly based on the diffusion output rotation.

---

## Technical Notes

### Mimic Joints

- L1_2 mimics L1_1 (q_idx 8 mimics 6)
- L2_2 mimics L2_1 (q_idx 9 mimics 7)
- PyTorch-kinematics handles mimic joints internally
- We extract angles for **both** L2 joints (indices 7 and 9) since they're both in the q tensor

### Joint Order (verified)

```python
['virtual_joint_x', 'virtual_joint_y', 'virtual_joint_z',
 'virtual_joint_roll', 'virtual_joint_pitch', 'virtual_joint_yaw',
 'left_ezgripper_knuckle_palm_L1_1',      # q_idx=6
 'left_ezgripper_knuckle_L1_L2_1',         # q_idx=7 ← L2_1 joint
 'left_ezgripper_knuckle_palm_L1_2',      # q_idx=8
 'left_ezgripper_knuckle_L1_L2_2']         # q_idx=9 ← L2_2 joint
```
