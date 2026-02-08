# LeapHand Variant Datasets

Generation of variant datasets by Q-mapping original leaphand grasps to each variant's joint space.

## Overview

| Dataset File | Robot | DOF | Entries | Source |
|-------------|-------|-----|---------|--------|
| `leaphand_dataset_filtered.pt` | leaphand | 22 | 7,800 | Original (human-generated) |
| `leaphand_graph_1_dataset.pt` | leaphand_graph_1 | 18 | 7,800 | Q-mapped from leaphand |
| `leaphand_graph_2_dataset.pt` | leaphand_graph_2 | 18 | 7,800 | Q-mapped from leaphand |
| `leaphand_morpho_1_dataset.pt` | leaphand_morpho_1 | 18 | 7,800 | Q-mapped from leaphand |
| `leaphand_morpho_2_dataset.pt` | leaphand_morpho_2 | 26 | 7,800 | Q-mapped from leaphand |
| `leaphand_morpho_3_dataset.pt` | leaphand_morpho_3 | 19 | 7,800 | Q-mapped from leaphand |
| `leaphand_graph_morpho_1_dataset.pt` | leaphand_graph_morpho_1 | 19 | 7,800 | Q-mapped from leaphand |
| `leaphand_all.pt` | All 7 variants | Mixed | 54,600 | Merged |

All files are in `data/CMapDataset_filtered/`.

---

## Dataset Structure

Each `.pt` file contains `{'metadata': [...]}` where each entry is a 3-tuple:

```python
(target_q: Tensor[DOF], object_name: str, robot_name: str)
```

- `target_q`: Joint angle vector (DOF depends on variant)
- `object_name`: e.g. `'ycb+power_drill'`, `'contactdb+alarm_clock'`
- `robot_name`: e.g. `'leaphand_graph_1'`

78 objects, 100 grasps each = 7,800 entries per variant. morpho_3 is a subset variant (19 DOF) with shortened non-thumb fingers but full-length thumb.

---

## Q Mapping

Joint values are mapped from leaphand (DOF=22) to each variant using **joint name matching** via pytorch-kinematics topological sort order.

### Subset Mapping (graph_1, graph_2, morpho_1: 22 -> 18 DOF)

Every variant joint exists in leaphand. Direct indexing by name:

```python
src_joints = hands['leaphand'].get_joint_orders()       # 22 joints
dst_joints = hands[variant].get_joint_orders()           # 18 joints
indices = [src_joints.index(j) for j in dst_joints]
q_variant = q_leaphand[indices]
```

### Expand Mapping (morpho_2: 22 -> 26 DOF)

4 extra joints not in leaphand. Shared joints mapped by name, new joints copy from source:

```python
COPY_MAP = {'2_1': '2', '6_1': '6', '10_1': '10', '14_1': '14'}
```

Virtual joints [0:6] (palm translation + rotation) are identical across all variants.

### Joint Order Reference

```
leaphand  (22): [vx,vy,vz,vr,vp,vy, '1','0','2','3','5','4','6','7','9','8','10','11','12','13','14','15']
graph_1   (18): [vx,vy,vz,vr,vp,vy, '1','0','2','3','9','8','10','11','12','13','14','15']
graph_2   (18): [vx,vy,vz,vr,vp,vy, '5','4','6','7','9','8','10','11','12','13','14','15']
morpho_1  (18): [vx,vy,vz,vr,vp,vy, '1','0','3','5','4','7','9','8','11','12','13','15']
morpho_2  (26): [vx,vy,vz,vr,vp,vy, '1','0','2','2_1','3','5','4','6','6_1','7','9','8','10','10_1','11','12','13','14','14_1','15']
morpho_3  (19): [vx,vy,vz,vr,vp,vy, '1','0','3','5','4','7','9','8','11','12','13','14','15']
graph_morpho_1 (19): [vx,vy,vz,vr,vp,vy, '5','4','6','7','9','8','10','11','12','13','14','14_1','15']
```

---

## Generation

```bash
# Generate all 4 variant datasets + merged leaphand_all.pt
conda run -n rpf python dataset/generate_variant_datasets.py

# Custom source/output
python dataset/generate_variant_datasets.py --source data/CMapDataset_filtered/leaphand_dataset_filtered.pt --output-dir data/CMapDataset_filtered

# Skip merged file
python dataset/generate_variant_datasets.py --no-merge
```

---

## Merging

Individual datasets can be merged using `merge_dataset.py`:

```bash
# Merge all leaphand variants
python merge_dataset.py \
    data/CMapDataset_filtered/leaphand_dataset_filtered.pt \
    data/CMapDataset_filtered/leaphand_graph_1_dataset.pt \
    data/CMapDataset_filtered/leaphand_graph_2_dataset.pt \
    data/CMapDataset_filtered/leaphand_morpho_1_dataset.pt \
    data/CMapDataset_filtered/leaphand_morpho_2_dataset.pt \
    data/CMapDataset_filtered/leaphand_morpho_3_dataset.pt \
    data/CMapDataset_filtered/leaphand_graph_morpho_1_dataset.pt \
    --out data/CMapDataset_filtered/leaphand_all.pt

# Merge with other robots for full multi-robot training
python merge_dataset.py \
    data/CMapDataset_filtered/cmap_full_dataset.pt \
    data/CMapDataset_filtered/leaphand_graph_1_dataset.pt \
    data/CMapDataset_filtered/leaphand_graph_2_dataset.pt \
    data/CMapDataset_filtered/leaphand_morpho_1_dataset.pt \
    data/CMapDataset_filtered/leaphand_morpho_2_dataset.pt \
    data/CMapDataset_filtered/leaphand_morpho_3_dataset.pt \
    data/CMapDataset_filtered/leaphand_graph_morpho_1_dataset.pt \
    --out data/CMapDataset_filtered/cmap_full_with_variants.pt
```

---

## Training Config

### Cross-Embodiment Training (leaphand + variants)

Config: `config/train_diff_v3_ce_leaphand.yaml`

```yaml
dataset:
  dataset_path: 'data/CMapDataset_filtered/leaphand_all.pt'
  gt_robot_names:
    - 'leaphand'
  no_gt_robot_names:
    - 'leaphand_graph_1'
    - 'leaphand_graph_2'
    - 'leaphand_morpho_1'
    - 'leaphand_morpho_2'
    - 'leaphand_morpho_3'
    - 'leaphand_graph_morpho_1'
```

Launch:
```bash
python train_diff_v3_ce.py --config-name train_diff_v3_ce_leaphand
```

### How It Works

- **GT robot (leaphand)**: Trains with object-conditioned diffusion using real grasp data
- **No-GT robots (variants)**: Train with self-reconstruction of FK poses; object conditioning is dropped (classifier-free guidance with `p_uncond`)
- `CrossEmbodimentDataset` reads from `dataset_path` (configurable; defaults to `cmap_full_dataset.pt`)
- The `dataset_path` field is optional — existing configs without it still work as before

---

## Visualization

Load from `leaphand_all.pt` to visualize any variant:

```python
import torch
from utils.hand_model import create_hand_model

dataset_path = 'data/CMapDataset_filtered/leaphand_all.pt'
metadata = torch.load(dataset_path, map_location='cpu', weights_only=False)['metadata']

robot_name = 'leaphand_graph_1'
object_name = 'contactdb+alarm_clock'
grasps = [m[0] for m in metadata if m[1] == object_name and m[2] == robot_name]

hand = create_hand_model(robot_name)
mesh = hand.get_trimesh_q(grasps[0])['visual']
```

For interactive comparison of all 5 variants:
```bash
python visualization/vis_leaphand_variants.py
```
