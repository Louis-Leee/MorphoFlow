# V5 CE VAE: Dual Object Split + GT/No-GT Overlap

## Overview

V5 extends V4 CE VAE with two key dataset changes:

1. **Separate object splits**: GT robots use `split_train_validate_objects.json` (48 train objects), while no-GT robots can use `split_train_validate_objects_no_gt.json` (58 train objects). Since no-GT robots never see object features, including validation objects in their training set does not cause data leakage.

2. **GT/No-GT overlap**: The same robot can appear in both `gt_robot_names` and `no_gt_robot_names`. When selected as GT, it trains with real object conditioning; when selected as no-GT, it trains without. These are separate training entries.

## Model

Uses `RobotGraphV4CE` (same as V4). No model changes.

## New Files

| File | Purpose |
|------|---------|
| `dataset/CrossEmbodimentDatasetV5.py` | Dataset with dual-split + overlap |
| `train_diff_v5_ce_vae.py` | Training script |
| `config/train_diff_v5_ce_vae.yaml` | Training config |
| `test_diff_v5_ce_vae.py` | Test script |
| `config/test_diff_v5_ce_vae.yaml` | Test config |

## Key Changes in CrossEmbodimentDatasetV5

### New Parameter: `no_gt_split_json_path`

```python
CrossEmbodimentDatasetV5(
    ...,
    no_gt_split_json_path='data/CMapDataset_filtered/split_train_validate_objects_no_gt.json',
)
```

If `None`, falls back to the same split as GT (backward compatible).

### Overlap via (robot, mode) Tuples

```python
# V4: flat list, no overlap support
self.all_robot_names = gt_robot_names + no_gt_robot_names  # duplicates break routing

# V5: (robot, mode) tuples
self.robot_entries = [('allegro', 'gt'), ('barrett', 'gt'), ('barrett', 'nogt')]
# __getitem__ picks a tuple, dispatches on mode
```

### Dual Metadata Filtering

- GT metadata filtered by `gt_object_names` (48 objects) + `gt_robot_set`
- No-GT metadata filtered by `nogt_object_names` (58 objects) + `no_gt_robot_set`
- Object PCs loaded for the union of both sets

## Config Example

```yaml
dataset:
  gt_robot_names: ['allegro', 'shadowhand']
  no_gt_robot_names: ['barrett']
  no_gt_split_json_path: 'data/CMapDataset_filtered/split_train_validate_objects_no_gt.json'

model:
  ce_config:
    no_gt_robot_names: ['barrett']
```

With overlap:

```yaml
dataset:
  gt_robot_names: ['allegro', 'barrett']
  no_gt_robot_names: ['barrett']
  # barrett appears in both: GT path uses 48 objects, no-GT uses 58
```

## Usage

```bash
# Training
python train_diff_v5_ce_vae.py

# Testing (same as V4 — model is identical)
python test_diff_v5_ce_vae.py --hands allegro barrett shadowhand --ckpt <path> --gpu 0
```
