# LeapHand Variants Creation Guide

This document describes how to create LeapHand variants by removing or modifying fingers.

## Existing Variants

| Variant | Modification | Actuated Joints | DOF | Links |
|---------|-------------|-----------------|-----|-------|
| leaphand | None | 16 (0-15) | 22 | 17 |
| leaphand_graph_1 | Middle finger removed | 12 (0,1,2,3,8-15) | 18 | 13 |
| leaphand_graph_2 | Index finger removed | 12 (4-15) | 18 | 13 |
| leaphand_morpho_1 | All DIP links removed (shortened fingers) | 12 (0,1,3,4,5,7,8,9,11,12,13,15) | 18 | 13 |
| leaphand_morpho_2 | Extra DIP links added (elongated fingers) | 20 (0-3,2_1,4-7,6_1,8-11,10_1,12-15,14_1) | 26 | 21 |
| leaphand_morpho_3 | Non-thumb DIP removed (shortened non-thumb) | 13 (0,1,3,4,5,7,8,9,11,12,13,14,15) | 19 | 14 |
| leaphand_graph_morpho_1 | No index + elongated thumb | 13 (4-11,12,13,14,14_1,15) | 19 | 14 |

---

## Overview

LeapHand variants are created by removing specific links and joints from the base URDF. Each variant requires updates across multiple files:

1. **URDF file** - Define the robot structure
2. **Metadata files** - Register the new variant
3. **Config files** - Add link counts and embodiment lists
4. **Code files** - Add IK configurations and controller logic

---

## leaphand_graph_1 (No Middle Finger Variant)

### Basic Parameters

| Parameter | leaphand (original) | leaphand_graph_1 |
|-----------|---------------------|------------------|
| Actuated joints | 16 (0-15) | 12 (0,1,2,3,8-15) |
| Total DOF (with virtual) | 22 | 18 |
| robot_links | 17 | 13 |
| Fingers | 4 (index, middle, ring, thumb) | 3 (index, ring, thumb) |

### Removed Components

**Joints removed:** 4, 5, 6, 7 (middle finger kinematic chain)

**Links removed:**
- `mcp_joint_2` - Middle finger MCP
- `pip_2` - Middle finger PIP
- `dip_2` - Middle finger DIP
- `fingertip_2` - Middle finger tip
- `extra_middle_tip_head` - Middle finger extra tip link

---

## Files Modified for leaphand_graph_1

### 1. URDF File

**Location:** `data/data_urdf/robot/leaphand/leap_hand_right_extended_graph_1.urdf`

Created by copying `leap_hand_right_extended.urdf` and removing:
- All `<link>` elements for middle finger links
- All `<joint>` elements for joints 4, 5, 6, 7
- Keep original joint numbering (0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15)

### 2. Metadata Files

**`utils/palm_centric.py`:**
```python
PALM_LINK_NAMES: Dict[str, str] = {
    # ...
    "leaphand": "palm_lower",
    "leaphand_graph_1": "palm_lower",  # Same palm as leaphand
    # ...
}
```

**`data/data_urdf/robot/urdf_assets_meta.json`:**
```json
{
  "urdf_path": {
    "leaphand_graph_1": "data/data_urdf/robot/leaphand/leap_hand_right_extended_graph_1.urdf"
  },
  "ee_link": {
    "leaphand_graph_1": "palm_lower"
  },
  "palm_link": {
    "leaphand_graph_1": "palm_lower"
  }
}
```

**`data/data_urdf/robot/removed_links.json`:**
```json
{
  "leaphand_graph_1": []
}
```

### 3. Config Files

**`config/test_diff_v3.yaml` and `config/vis_denoise.yaml`:**

```yaml
model:
  robot_links:
    leaphand: 17
    leaphand_graph_1: 13  # 4 fewer links (middle finger)

  embodiment:
    - 'leaphand'
    - 'leaphand_graph_1'
```

### 4. Point Cloud Generation

```bash
python dataset/generate_pc.py --robot_name leaphand_graph_1
```

Output: `data/PointCloud/robot/leaphand_graph_1.pt`

### 5. Code Files

#### `test_diff_v3.py` / `vis_denoise.py`

**FINGERTIP_JOINTS** - Maps parent-child link pairs to q_index:
```python
FINGERTIP_JOINTS = {
    'leaphand': {
        ('dip', 'fingertip'): 9,
        ('dip_2', 'fingertip_2'): 13,
        ('dip_3', 'fingertip_3'): 17,
        ('thumb_dip', 'thumb_fingertip'): 21,
    },
    'leaphand_graph_1': {
        ('dip', 'fingertip'): 9,           # Index: unchanged
        ('dip_3', 'fingertip_3'): 13,      # Ring: 17 -> 13
        ('thumb_dip', 'thumb_fingertip'): 17,  # Thumb: 21 -> 17
    },
}
```

**TIP_MAPPING** - Maps revolute tip links to fixed extra tip links for IK:
```python
LEAPHAND_GRAPH_1_TIP_MAPPING = {
    'fingertip': 'extra_index_tip_head',
    'fingertip_3': 'extra_ring_tip_head',
    # No fingertip_2 (middle finger removed)
}
```

**LINK_WEIGHTS** - Per-link IK weights:
```python
LEAPHAND_GRAPH_1_LINK_WEIGHTS = {
    'palm_lower': 3.0,
    'extra_ring_tip_head': 0.5,
    'extra_index_tip_head': 0.8,
    'thumb_fingertip': 3.0,
    # No extra_middle_tip_head (middle finger removed)
}
```

#### `utils/controller.py`

```python
elif robot_name == 'leaphand_graph_1':
    if joint_name in ['13']:  # Thumb base (originally joint 13 in leaphand)
        return None
    if joint_name in ['0', '8']:  # X-axis joints (no '4' - middle finger)
        link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)
    elif joint_name in ['1', '9', '12', '14']:  # Y-axis joints (no '5' - middle finger)
        link_dir = torch.tensor([0, 1, 0], dtype=torch.float32)
    else:
        link_dir = torch.tensor([0, -1, 0], dtype=torch.float32)
```

---

## leaphand_graph_2 (No Index Finger Variant)

### Basic Parameters

| Parameter | leaphand (original) | leaphand_graph_2 |
|-----------|---------------------|------------------|
| Actuated joints | 16 (0-15) | 12 (4-15) |
| Total DOF (with virtual) | 22 | 18 |
| robot_links | 17 | 13 |
| Fingers | 4 (index, middle, ring, thumb) | 3 (middle, ring, thumb) |

### Removed Components

**Joints removed:** 0, 1, 2, 3, index_tip (index finger kinematic chain)

**Links removed:**
- `mcp_joint` - Index finger MCP
- `pip` - Index finger PIP
- `dip` - Index finger DIP
- `fingertip` - Index finger tip
- `extra_index_tip_head` - Index finger extra tip link

---

## Files Modified for leaphand_graph_2

### 1. URDF File

**Location:** `data/data_urdf/robot/leaphand/leap_hand_right_extended_graph_2.urdf`

Created by copying `leap_hand_right_extended.urdf` and removing:
- All `<link>` elements for index finger links
- All `<joint>` elements for joints 0, 1, 2, 3, index_tip
- Keep original joint numbering (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

### 2. Metadata Files

Same structure as leaphand_graph_1, with paths updated for leaphand_graph_2.

### 3. Config Files

Same as leaphand_graph_1: `robot_links: 13`, added to `embodiment` list.

### 4. Point Cloud Generation

```bash
python dataset/generate_pc.py --robot_name leaphand_graph_2
```

Output: `data/PointCloud/robot/leaphand_graph_2.pt`

### 5. Code Files

#### FINGERTIP_JOINTS

```python
'leaphand_graph_2': {
    ('dip_2', 'fingertip_2'): 9,       # Middle: joint '7' at index 9
    ('dip_3', 'fingertip_3'): 13,      # Ring: joint '11' at index 13
    ('thumb_dip', 'thumb_fingertip'): 17,  # Thumb: joint '15' at index 17
},
```

#### TIP_MAPPING

```python
LEAPHAND_GRAPH_2_TIP_MAPPING = {
    'fingertip_2': 'extra_middle_tip_head',
    'fingertip_3': 'extra_ring_tip_head',
    # No fingertip (index finger removed)
}
```

#### LINK_WEIGHTS

```python
LEAPHAND_GRAPH_2_LINK_WEIGHTS = {
    'palm_lower': 3.0,
    'extra_ring_tip_head': 0.5,
    'extra_middle_tip_head': 0.7,
    'thumb_fingertip': 3.0,
    # No extra_index_tip_head (index finger removed)
}
```

#### controller.py

```python
elif robot_name == 'leaphand_graph_2':
    # LeapHand variant without index finger (joints 0-3 removed)
    if joint_name in ['13']:
        return None
    if joint_name in ['4', '8']:  # X-axis joints (no '0')
        link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)
    elif joint_name in ['5', '9', '12', '14']:  # Y-axis joints (no '1')
        link_dir = torch.tensor([0, 1, 0], dtype=torch.float32)
    else:
        link_dir = torch.tensor([0, -1, 0], dtype=torch.float32)
```

### Joint Order (pytorch-kinematics topological sort)

```python
from utils.hand_model import create_hand_model
hand = create_hand_model('leaphand_graph_2')
print(hand.get_joint_orders())
# ['virtual_joint_x', 'virtual_joint_y', 'virtual_joint_z',
#  'virtual_joint_roll', 'virtual_joint_pitch', 'virtual_joint_yaw',
#  '5', '4', '6', '7', '9', '8', '10', '11', '12', '13', '14', '15']
```

---

## FINGERTIP_JOINTS Index Calculation

The `q_idx` values in FINGERTIP_JOINTS are determined by pytorch-kinematics' topological sort order, NOT by joint numbering in URDF.

### Method

1. Load the hand model and get joint order:
```python
from utils.hand_model import create_hand_model
hand = create_hand_model('leaphand_graph_1')
joint_names = hand.get_joint_orders()
print(joint_names)
# ['virtual_joint_x', 'virtual_joint_y', 'virtual_joint_z',
#  'virtual_joint_roll', 'virtual_joint_pitch', 'virtual_joint_yaw',
#  '1', '0', '2', '3', '9', '8', '10', '11', '12', '13', '14', '15']
```

2. Find the index of each fingertip joint:
   - Index finger tip (joint '3'): index 9
   - Ring finger tip (joint '11'): index 13
   - Thumb tip (joint '15'): index 17

### Why Ring Finger Changes from 17 to 13

In `leaphand`:
- Joints 0-7 occupy indices 6-13
- Joint '11' (ring tip) is at index 17

In `leaphand_graph_1`:
- Joints 4-7 are removed (middle finger)
- Remaining joints shift: Joint '11' moves to index 13

---

## Creating a New Variant

### Step 1: Create URDF

1. Copy base URDF: `leap_hand_right_extended.urdf`
2. Remove unwanted `<link>` and `<joint>` elements
3. Keep original joint numbering for remaining joints
4. Save as `leap_hand_right_extended_graph_N.urdf`

### Step 2: Update Metadata

1. Add to `urdf_assets_meta.json`:
   - `urdf_path`
   - `ee_link`
   - `palm_link`

2. Add to `removed_links.json`

### Step 3: Generate Point Cloud

```bash
python dataset/generate_pc.py --robot_name leaphand_graph_N
```

### Step 4: Update Config Files

Add to all relevant config files:
- `robot_links` with correct link count
- `embodiment` list

### Step 5: Update Code

1. **FINGERTIP_JOINTS**: Calculate new q_idx values using the method above
2. **TIP_MAPPING**: Remove entries for deleted fingers
3. **LINK_WEIGHTS**: Remove entries for deleted finger links
4. **controller.py**: Add variant with correct joint names

### Step 6: Validate

```bash
# Test model loading
python -c "
from utils.hand_model import create_hand_model
import torch
hand = create_hand_model('leaphand_graph_N', torch.device('cuda:0'))
print(f'DOF: {hand.dof}')
print(f'Links: {list(hand.vertices.keys())}')
print(f'Joint names: {hand.get_joint_orders()}')
"

# Run evaluation
python test_diff_v3.py --config config/test_diff_v3.yaml --hands leaphand_graph_N --ckpt <checkpoint> --gpu 0

# Run visualization
python vis_denoise.py --config config/vis_denoise.yaml
```

---

## leaphand_morpho_1 (Shortened Fingers Variant)

### Basic Parameters

| Parameter | leaphand (original) | leaphand_morpho_1 |
|-----------|---------------------|-------------------|
| Actuated joints | 16 (0-15) | 12 (0,1,3,4,5,7,8,9,11,12,13,15) |
| Total DOF (with virtual) | 22 | 18 |
| robot_links | 17 | 13 |
| Fingers | 4 (all present) | 4 (all present, shortened by one segment) |

### Design

Unlike graph_1/graph_2 which remove entire fingers, morpho_1 keeps all 4 fingers but **shortens each by removing the DIP link**. The fingertip joints (3, 7, 11, 15) are reconnected directly to the PIP links, using the original DIP joint origins. This tests whether the model can generalize to morphological changes.

### Removed Components

**Joints removed:** 2, 6, 10, 14 (DIP joints for all four fingers)

**Links removed:**
- `dip` - Index finger DIP
- `dip_2` - Middle finger DIP
- `dip_3` - Ring finger DIP
- `thumb_dip` - Thumb DIP

### Joint Reconnection

When DIP links are removed, fingertip joints reconnect to PIP links with the DIP joint's origin:

| Fingertip Joint | Old Parent | New Parent | New Origin (from old DIP joint) |
|-----------------|-----------|------------|-------------------------------|
| 3 (index) | dip | pip | xyz="0.015 0.0143 -0.013" rpy="1.57079 -1.57079 0" |
| 7 (middle) | dip_2 | pip_2 | xyz="0.015 0.0143 -0.013" rpy="1.57079 -1.57079 0" |
| 11 (ring) | dip_3 | pip_3 | xyz="0.015 0.0143 -0.013" rpy="1.57079 -1.57079 0" |
| 15 (thumb) | thumb_dip | thumb_pip | xyz="0 0.0145 -0.017" rpy="-1.57079 0 0" |

### FINGERTIP_JOINTS (R_origin handling)

Because the reconnected joints have non-identity `rpy` origins, the angle extraction requires removing the R_origin offset:

```python
# Index/Middle/Ring: rpy="1.57079 -1.57079 0"
R_origin = [[0,-1,0],[0,0,-1],[1,0,0]]

# Thumb: rpy="-1.57079 0 0"
R_origin = [[1,0,0],[0,0,1],[0,-1,0]]

# Extract: R_joint = R_origin^T @ R_rel, angle = atan2(R_joint[1,0], R_joint[0,0])
# predict_q[:, q_idx] = -angle  (negate for axis (0,0,-1))
```

```python
'leaphand_morpho_1': {
    ('pip', 'fingertip'): 8,            # Index: joint '3'
    ('pip_2', 'fingertip_2'): 11,       # Middle: joint '7'
    ('pip_3', 'fingertip_3'): 14,       # Ring: joint '11'
    ('thumb_pip', 'thumb_fingertip'): 17,  # Thumb: joint '15'
},
```

### Joint Order (pytorch-kinematics topological sort)

```python
hand = create_hand_model('leaphand_morpho_1')
print(hand.get_joint_orders())
# ['virtual_joint_x', 'virtual_joint_y', 'virtual_joint_z',
#  'virtual_joint_roll', 'virtual_joint_pitch', 'virtual_joint_yaw',
#  '1', '0', '3', '5', '4', '7', '9', '8', '11', '12', '13', '15']
```

### controller.py

```python
elif robot_name == 'leaphand_morpho_1':
    if joint_name in ['13']:
        return None
    if joint_name in ['0', '4', '8']:  # X-axis MCP joints (all 4 fingers present)
        link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)
    elif joint_name in ['1', '5', '9', '12']:  # Y-axis joints (no '14' - removed)
        link_dir = torch.tensor([0, 1, 0], dtype=torch.float32)
    else:
        link_dir = torch.tensor([0, -1, 0], dtype=torch.float32)
```

---

## leaphand_morpho_2 (Elongated Fingers Variant)

### Basic Parameters

| Parameter | leaphand (original) | leaphand_morpho_2 |
|-----------|---------------------|-------------------|
| Actuated joints | 16 (0-15) | 20 (0-15, 2_1, 6_1, 10_1, 14_1) |
| Total DOF (with virtual) | 22 | 26 |
| robot_links | 17 | 21 |
| Fingers | 4 (all present) | 4 (all present, each elongated by one DIP segment) |

### Design

The inverse of morpho_1: instead of removing DIP links, morpho_2 **adds an extra DIP link** to each finger chain. This tests whether the model can generalize to longer fingers.

### Added Components

**New Joints:** 2_1, 6_1, 10_1, 14_1 (extra DIP joints for all four fingers)

**New Links:**
- `dip_1` - Index finger extra DIP (mesh: `dip.obj`)
- `dip_2_1` - Middle finger extra DIP (mesh: `dip.obj`)
- `dip_3_1` - Ring finger extra DIP (mesh: `dip.obj`)
- `thumb_dip_1` - Thumb extra DIP (mesh: `thumb_dip.obj`)

### Chain Changes

Each finger chain has a new DIP link inserted between the existing DIP and the fingertip:

| Finger | Original Chain | Morpho 2 Chain |
|--------|---------------|----------------|
| Index | pip → dip → fingertip | pip → dip → **dip_1** → fingertip |
| Middle | pip_2 → dip_2 → fingertip_2 | pip_2 → dip_2 → **dip_2_1** → fingertip_2 |
| Ring | pip_3 → dip_3 → fingertip_3 | pip_3 → dip_3 → **dip_3_1** → fingertip_3 |
| Thumb | thumb_dip → thumb_fingertip | thumb_dip → **thumb_dip_1** → thumb_fingertip |

### New Joint Properties

| New Joint | Origin (from existing joint) | Limits (from existing joint) |
|-----------|------------------------------|------------------------------|
| 2_1 | j3: xyz="0 -0.0361 0.0002" rpy="0 0 0" | j2: [-0.506, 1.885] |
| 6_1 | j7: xyz="0 -0.0361 0.0002" rpy="0 0 0" | j6: [-0.506, 1.885] |
| 10_1 | j11: xyz="0 -0.03609 0.0002" rpy="0 0 0" | j10: [-0.506, 1.885] |
| 14_1 | j15: xyz="0 0.0466 0.0002" rpy="0 0 0" | j14: [-1.20, 1.90] |

### FINGERTIP_JOINTS

Since joints 3/7/11 still have rpy="0 0 0" and joint 15 still has rpy="0 0 3.14159", the extraction uses the same logic as base leaphand (with `child == 'thumb_fingertip'` detection for the Rz(pi) offset).

```python
'leaphand_morpho_2': {
    ('dip_1', 'fingertip'): 10,           # Index: joint '3' at index 10
    ('dip_2_1', 'fingertip_2'): 15,       # Middle: joint '7' at index 15
    ('dip_3_1', 'fingertip_3'): 20,       # Ring: joint '11' at index 20
    ('thumb_dip_1', 'thumb_fingertip'): 25,  # Thumb: joint '15' at index 25
},
```

### Joint Order (pytorch-kinematics topological sort)

```python
hand = create_hand_model('leaphand_morpho_2')
print(hand.get_joint_orders())
# ['virtual_joint_x', 'virtual_joint_y', 'virtual_joint_z',
#  'virtual_joint_roll', 'virtual_joint_pitch', 'virtual_joint_yaw',
#  '1', '0', '2', '2_1', '3', '5', '4', '6', '6_1', '7',
#  '9', '8', '10', '10_1', '11', '12', '13', '14', '14_1', '15']
```

### controller.py

```python
elif robot_name == 'leaphand_morpho_2':
    if joint_name in ['13']:
        return None
    if joint_name in ['0', '4', '8']:  # X-axis MCP joints
        link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)
    elif joint_name in ['1', '5', '9', '12', '14']:  # Y-axis joints
        link_dir = torch.tensor([0, 1, 0], dtype=torch.float32)
    else:  # 2, 2_1, 3, 6, 6_1, 7, 10, 10_1, 11, 14_1, 15
        link_dir = torch.tensor([0, -1, 0], dtype=torch.float32)
```

### Visualization Q Mapping

Since morpho_2 has MORE DOF (26) than leaphand (22), the visualization script uses an "expand" mapping. Shared joints are mapped by name, and new joints (2_1, 6_1, 10_1, 14_1) copy values from their source joints (2, 6, 10, 14).

---

## leaphand_morpho_3 (Shortened Non-Thumb Fingers Variant)

### Basic Parameters

| Parameter | leaphand (original) | leaphand_morpho_3 |
|-----------|--------------------|--------------------|
| DOF | 22 (6 virtual + 16) | 19 (6 virtual + 13) |
| Links | 17 | 14 |
| Removed Joints | — | 2, 6, 10 |
| Removed Links | — | dip, dip_2, dip_3 |

### Key Difference from morpho_1

morpho_3 is a hybrid of morpho_1 and base leaphand:
- **Non-thumb fingers**: Same as morpho_1 — DIP removed, fingertip reparented to pip
- **Thumb**: Unchanged from base leaphand — full length with thumb_dip

### URDF Changes

1. Remove links: `dip`, `dip_2`, `dip_3` (keep `thumb_dip`)
2. Remove joints: `2`, `6`, `10` (keep `14`, `15`)
3. Reparent joints 3, 7, 11: parent changes from dip → pip, origin `rpy="1.57079 -1.57079 0"`

### Joint Order (pytorch-kinematics topological sort)

```
morpho_3 (19): [vx,vy,vz,vr,vp,vy, '1','0','3', '5','4','7', '9','8','11', '12','13','14','15']
```

### FINGERTIP_JOINTS

```python
'leaphand_morpho_3': {
    ('pip', 'fingertip'): 8,              # Index (same as morpho_1)
    ('pip_2', 'fingertip_2'): 11,         # Middle (same as morpho_1)
    ('pip_3', 'fingertip_3'): 14,         # Ring (same as morpho_1)
    ('thumb_dip', 'thumb_fingertip'): 18, # Thumb: q_idx=18 (NOT 17 like morpho_1!)
}
```

### R_origin/R_offset Extraction (HYBRID)

- **Non-thumb** (joints 3, 7, 11): `rpy="1.57079 -1.57079 0"` → `R_origin = [[0,-1,0],[0,0,-1],[1,0,0]]` (same as morpho_1)
- **Thumb** (joint 15): `rpy="0 0 3.14159"` → `R_offset = [[-1,0,0],[0,-1,0],[0,0,1]]` (same as base leaphand)

### Controller

```python
elif robot_name == 'leaphand_morpho_3':
    if joint_name in ['13']: return None
    if joint_name in ['0', '4', '8']: link_dir = [1, 0, 0]
    elif joint_name in ['1', '5', '9', '12', '14']: link_dir = [0, 1, 0]  # includes '14'
    else: link_dir = [0, -1, 0]
```

---

## Reference: LeapHand Finger Structure

| Finger | Joints | Links |
|--------|--------|-------|
| Index | 0, 1, 2, 3 | mcp_joint, pip, dip, fingertip, extra_index_tip_head |
| Middle | 4, 5, 6, 7 | mcp_joint_2, pip_2, dip_2, fingertip_2, extra_middle_tip_head |
| Ring | 8, 9, 10, 11 | mcp_joint_3, pip_3, dip_3, fingertip_3, extra_ring_tip_head |
| Thumb | 12, 13, 14, 15 | thumb_mcp, thumb_pip, thumb_dip, thumb_fingertip, extra_thumb_tip_head |

---

## Test Results

### leaphand_graph_1 @ epoch=489

```
Total success rate: 14.50%
Total diversity: 0.5449
Grasp generation time: 0.018 s/grasp
```

Per-object breakdown:
- contactdb+apple: 19.0%
- contactdb+camera: 9.0%
- contactdb+cylinder_medium: 1.0%
- contactdb+door_knob: 9.0%
- contactdb+rubber_duck: 25.0%
- contactdb+water_bottle: 3.0%
- ycb+baseball: 27.0%
- ycb+pear: 34.0%
- ycb+potted_meat_can: 3.0%
- ycb+tomato_soup_can: 15.0%

---

## leaphand_graph_morpho_1 (No Index Finger + Elongated Thumb)

### Basic Parameters

| Parameter | leaphand (original) | leaphand_graph_morpho_1 |
|-----------|---------------------|-------------------------|
| Actuated joints | 16 (0-15) | 13 (4-11, 12-15, 14_1) |
| Total DOF (with virtual) | 22 | 19 |
| robot_links | 17 | 14 |
| Fingers | 4 (index, middle, ring, thumb) | 3 (middle, ring, elongated thumb) |

### Modification Summary

Combines **graph_2** (no index finger) with **morpho_2** thumb elongation:

**From graph_2 (removed):**
- Joints: 0, 1, 2, 3 (index finger chain)
- Links: mcp_joint, pip, dip, fingertip, extra_index_tip_head

**From morpho_2 (added):**
- Link: `thumb_dip_1` (uses thumb_dip.obj mesh)
- Joint: `14_1` (parent=thumb_dip, child=thumb_dip_1, origin="0 0.0466 0.0002" rpy="0 0 0")
- Joint `15` reparented: parent thumb_dip -> thumb_dip_1

### Joint Order (pytorch-kinematics topological sort)

```
graph_morpho_1 (19): [vx,vy,vz,vr,vp,vy, '5','4','6','7', '9','8','10','11', '12','13','14','14_1','15']
```

### FINGERTIP_JOINTS

```python
'leaphand_graph_morpho_1': {
    ('dip_2', 'fingertip_2'): 9,            # Middle finger
    ('dip_3', 'fingertip_3'): 13,           # Ring finger
    ('thumb_dip_1', 'thumb_fingertip'): 18, # Thumb (elongated chain)
}
```

Only 3 fingertip joints (no index finger). Uses DEFAULT leaphand extraction (same as base leaphand, graph_2, morpho_2).

### Q Mapping (from leaphand 22 DOF)

Expand type: 19 DOF with 1 extra joint (`14_1`).
- 18 shared joints mapped by name from leaphand
- 1 copied joint: `14_1` copies value from joint `14` (via MORPHO_2_COPY_MAP)

### Controller

Based on graph_2: no joint '0' (index removed), adds '14_1' to the else group (Y-axis negative joints).

### Files Modified

- URDF: `data/data_urdf/robot/leaphand/leap_hand_right_extended_graph_morpho_1.urdf`
- Metadata: `urdf_assets_meta.json`, `removed_links.json`, `palm_centric.py`
- Configs: `test_diff_v3.yaml`, `test_diff_v3_ce.yaml`, `vis_denoise.yaml`, `train_diff_v3_ce_leaphand.yaml`
- Code: `vis_denoise.py`, `test_diff_v3.py`, `test_diff_v3_ce.py`, `controller.py`
- Visualization: `vis_leaphand_variants.py`
- Dataset: `generate_variant_datasets.py`
