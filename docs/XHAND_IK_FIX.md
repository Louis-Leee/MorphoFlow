# XHand IK Optimization Fix

This document describes the fix for the XHand fingertip alignment issue in IK optimization.

---

## Problem Description

When running vis_denoise.py, test_diff_v3.py, or test_diff_v3_ce.py with xhand, the `*_link2` links (last controllable links on each finger) have poor alignment with the diffusion output - they don't rotate properly.

---

## Root Cause

The `*_joint2` joints are **pure-rotation joints** from IK's perspective (same issue as LeapHand/EZGripper/Robotiq):

1. **IK limitation**: The IK solver optimizes link **positions**, not rotations
2. The `*_link2` link origin is at the joint location
3. When joint2 rotates, the link origin stays in place (only orientation changes)
4. IK cannot optimize joints that don't affect position

### URDF Structure

```
right_hand_link (palm)
├─ Thumb Chain:
│  └─ thumb_bend_link (joint q=6, Z-axis)
│     └─ thumb_rota_link1 (joint q=7, Y-axis)
│        └─ thumb_rota_link2 (joint q=8, Y-axis) ← PURE ROTATION
│           └─ thumb_rota_tip (FIXED)
│
├─ Index Chain:
│  └─ index_bend_link (joint q=9, Y-axis)
│     └─ index_rota_link1 (joint q=10, X-axis)
│        └─ index_rota_link2 (joint q=11, X-axis) ← PURE ROTATION
│           └─ index_rota_tip (FIXED)
│
├─ Middle Chain:
│  └─ mid_link1 (joint q=12, X-axis)
│     └─ mid_link2 (joint q=13, X-axis) ← PURE ROTATION
│        └─ mid_tip (FIXED)
│
├─ Ring Chain:
│  └─ ring_link1 (joint q=14, X-axis)
│     └─ ring_link2 (joint q=15, X-axis) ← PURE ROTATION
│        └─ ring_tip (FIXED)
│
└─ Pinky Chain:
   └─ pinky_link1 (joint q=16, X-axis)
      └─ pinky_link2 (joint q=17, X-axis) ← PURE ROTATION
         └─ pinky_tip (FIXED)
```

### Key Difference: Mixed Rotation Axes

Unlike other hands, XHand has different rotation axes for thumb vs other fingers:
- **Thumb**: Y-axis rotation (0, 1, 0)
- **Other fingers**: X-axis rotation (1, 0, 0)

---

## Solution: FINGERTIP_JOINTS Approach

Extract `*_joint2` angles from SE3 rotation with axis-specific formulas.

### Configuration

```python
FINGERTIP_JOINTS = {
    'xhand': {
        # (parent_link, child_link): q_index
        ('right_hand_thumb_rota_link1', 'right_hand_thumb_rota_link2'): 8,
        ('right_hand_index_rota_link1', 'right_hand_index_rota_link2'): 11,
        ('right_hand_mid_link1', 'right_hand_mid_link2'): 13,
        ('right_hand_ring_link1', 'right_hand_ring_link2'): 15,
        ('right_hand_pinky_link1', 'right_hand_pinky_link2'): 17,
    },
}
```

### Angle Extraction Formulas

**Y-axis rotation (thumb)**:
```python
# Ry = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
angle = torch.atan2(R_rel[:, 0, 2], R_rel[:, 2, 2])
```

**X-axis rotation (other fingers)**:
```python
# Rx = [[1, 0, 0], [0, cos, -sin], [0, sin, cos]]
angle = torch.atan2(R_rel[:, 2, 1], R_rel[:, 1, 1])
```

---

## Files Modified

| File | Change |
|------|--------|
| `vis_denoise.py` | Added xhand to FINGERTIP_JOINTS, updated extract_fingertip_joints() |
| `test_diff_v3.py` | Same changes |
| `test_diff_v3_ce.py` | Same changes |
| `docs/XHAND_IK_FIX.md` | This documentation |

---

## Verification

```bash
# Clear cache
rm outputs/vis_cache.pt

# Test vis_denoise.py (update config first)
python vis_denoise.py --config config/vis_denoise.yaml

# Test test_diff_v3.py
python test_diff_v3.py --config config/test_diff_v3.yaml --hands xhand --gpu 0

# Test test_diff_v3_ce.py
python test_diff_v3_ce.py --config config/test_diff_v3_ce.yaml --hands xhand --gpu 0
```

---

## Technical Notes

### Joint Order (verified)
```python
['virtual_x_translation_joint',      # 0
 'virtual_y_translation_joint',      # 1
 'virtual_z_translation_joint',      # 2
 'virtual_x_rotation_joint',         # 3
 'virtual_y_rotation_joint',         # 4
 'virtual_z_rotation_joint',         # 5
 'right_hand_thumb_bend_joint',      # 6  (Z-axis)
 'right_hand_thumb_rota_joint1',     # 7  (Y-axis)
 'right_hand_thumb_rota_joint2',     # 8  (Y-axis) ← thumb_rota_link2 joint
 'right_hand_index_bend_joint',      # 9  (Y-axis)
 'right_hand_index_joint1',          # 10 (X-axis)
 'right_hand_index_joint2',          # 11 (X-axis) ← index_rota_link2 joint
 'right_hand_mid_joint1',            # 12 (X-axis)
 'right_hand_mid_joint2',            # 13 (X-axis) ← mid_link2 joint
 'right_hand_ring_joint1',           # 14 (X-axis)
 'right_hand_ring_joint2',           # 15 (X-axis) ← ring_link2 joint
 'right_hand_pinky_joint1',          # 16 (X-axis)
 'right_hand_pinky_joint2']          # 17 (X-axis) ← pinky_link2 joint
```

### Comparison with Other Hands

| Hand | Link Affected | Joint Axis | Offset |
|------|--------------|-----------|--------|
| LeapHand | fingertip, fingertip_2, etc. | Z (0,0,-1) | π for thumb |
| EZGripper | finger_L2_1, finger_L2_2 | Y (0,1,0) | None |
| Robotiq 3-finger | fingerA/B/C_dist | Y (0,1,0) | -0.436332312999 |
| **XHand** | thumb_rota_link2 | **Y (0,1,0)** | None |
| **XHand** | index/mid/ring/pinky_link2 | **X (1,0,0)** | None |
