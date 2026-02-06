# Robotiq 3-Finger IK Optimization Fix

This document describes the fix for the Robotiq 3-finger fingertip alignment issue in IK optimization.

---

## Problem Description

When running vis_denoise.py, test_diff_v3.py, or test_diff_v3_ce.py with robotiq_3finger, the fingertip links (`gripper_fingerA_dist`, `gripper_fingerB_dist`, `gripper_fingerC_dist`) have poor alignment with the diffusion output - they don't rotate properly.

---

## Root Cause

The `*_joint_4` joints are **pure-rotation joints** (same issue as EZGripper/LeapHand):

1. **IK limitation**: The IK solver optimizes link **positions**, not rotations
2. The `*_dist` link origin is at the joint location
3. When joint_4 rotates, the link origin stays in place (only orientation changes)
4. IK cannot optimize joints that don't affect position

### URDF Structure

```
gripper_palm
├─ gripper_fingerA_base (FIXED)
│  └─ gripper_fingerA_prox (joint_2, q_idx=6, Y-axis)
│     └─ gripper_fingerA_med (joint_3, q_idx=7, Y-axis)
│        └─ gripper_fingerA_dist (joint_4, q_idx=8, Y-axis) ← PURE ROTATION
├─ gripper_fingerB_base (knuckle, q_idx=9, X-axis)
│  └─ gripper_fingerB_prox (joint_2, q_idx=10, Y-axis)
│     └─ gripper_fingerB_med (joint_3, q_idx=11, Y-axis)
│        └─ gripper_fingerB_dist (joint_4, q_idx=12, Y-axis) ← PURE ROTATION
└─ gripper_fingerC_base (knuckle, q_idx=13, -X-axis)
   └─ gripper_fingerC_prox (joint_2, q_idx=14, Y-axis)
      └─ gripper_fingerC_med (joint_3, q_idx=15, Y-axis)
         └─ gripper_fingerC_dist (joint_4, q_idx=16, Y-axis) ← PURE ROTATION
```

### Additional Complication: Fixed Rotation Offset

Unlike EZGripper (which has `rpy="0 0 0"`), Robotiq has a fixed rotation offset:

```xml
<joint name="gripper_fingerA_joint_4" type="revolute">
    <origin rpy="0 -0.436332312999 0" xyz="0 0 0.03810000"/>
    <axis xyz="0 1 0"/>
</joint>
```

The `-0.436332312999` radians (≈-25°) offset must be compensated when extracting joint angles.

---

## Solution: FINGERTIP_JOINTS Approach

Extract `*_joint_4` angles from SE3 rotation with offset compensation.

### Configuration

```python
ROBOTIQ_JOINT4_OFFSET = -0.436332312999  # rpy offset in URDF

FINGERTIP_JOINTS = {
    'robotiq_3finger': {
        ('gripper_fingerA_med', 'gripper_fingerA_dist'): 8,
        ('gripper_fingerB_med', 'gripper_fingerB_dist'): 12,
        ('gripper_fingerC_med', 'gripper_fingerC_dist'): 16,
    },
}
```

### Angle Extraction Formula

```python
# Y-axis rotation with offset
# R_rel = Ry(offset + joint_angle)
# joint_angle = extracted - offset

angle_with_offset = torch.atan2(R_rel[:, 0, 2], R_rel[:, 2, 2])
joint_angle = angle_with_offset - ROBOTIQ_JOINT4_OFFSET
```

---

## Files Modified

| File | Change |
|------|--------|
| `vis_denoise.py` | Added ROBOTIQ_JOINT4_OFFSET, robotiq_3finger to FINGERTIP_JOINTS, updated extract_fingertip_joints() |
| `test_diff_v3.py` | Same changes |
| `test_diff_v3_ce.py` | Same changes |
| `docs/ROBOTIQ_3FINGER_IK_FIX.md` | This documentation |

---

## Verification

```bash
# Clear cache
rm outputs/vis_cache.pt

# Test vis_denoise.py (update config first)
python vis_denoise.py --config config/vis_denoise.yaml

# Test test_diff_v3.py
python test_diff_v3.py --config config/test_diff_v3.yaml --hands robotiq_3finger --gpu 0

# Test test_diff_v3_ce.py
python test_diff_v3_ce.py --config config/test_diff_v3_ce.yaml --hands robotiq_3finger --gpu 0
```

---

## Technical Notes

### Joint Order (verified)
```python
['virtual_joint_x', 'virtual_joint_y', 'virtual_joint_z',
 'virtual_joint_roll', 'virtual_joint_pitch', 'virtual_joint_yaw',  # 0-5
 'gripper_fingerA_joint_2',  # 6
 'gripper_fingerA_joint_3',  # 7
 'gripper_fingerA_joint_4',  # 8 ← fingerA_dist joint
 'gripper_fingerB_knuckle',  # 9
 'gripper_fingerB_joint_2',  # 10
 'gripper_fingerB_joint_3',  # 11
 'gripper_fingerB_joint_4',  # 12 ← fingerB_dist joint
 'gripper_fingerC_knuckle',  # 13
 'gripper_fingerC_joint_2',  # 14
 'gripper_fingerC_joint_3',  # 15
 'gripper_fingerC_joint_4']  # 16 ← fingerC_dist joint
```

### Why Offset Compensation is Needed

For Y-axis rotation with offset φ:
```
R_rel = Ry(φ + θ)  where φ = -0.436332312999, θ = joint angle
extracted = atan2(R[0,2], R[2,2]) = φ + θ
θ = extracted - φ = extracted + 0.436332312999
```

### Comparison with Other Grippers

| Gripper | Joint Axis | Offset | Formula |
|---------|------------|--------|---------|
| LeapHand | Z (0,0,-1) | π for thumb | `atan2(R[1,0], R[0,0])` |
| EZGripper | Y (0,1,0) | None | `atan2(R[0,2], R[2,2])` |
| Robotiq 3-finger | Y (0,1,0) | -0.436332312999 | `atan2(R[0,2], R[2,2]) - offset` |
