# LeapHand Variant Comparison Visualization

Interactive viser-based tool for comparing how 6 LeapHand variants render the same grasp.

## Usage

```bash
python visualization/vis_leaphand_variants.py
python visualization/vis_leaphand_variants.py --port 8081
python visualization/vis_leaphand_variants.py --dataset data/CMapDataset_filtered/cmap_full_dataset.pt
```

Open `http://localhost:8080` in a browser after launching.

---

## Variants & Color Legend

| Variant | Color | Opacity | Modification |
|---------|-------|---------|-------------|
| leaphand | Blue (70, 130, 230) | 1.0 | Original, 16 actuated joints, 22 DOF |
| leaphand_graph_1 | Green (50, 200, 100) | 0.6 | Middle finger removed (joints 4-7), 12 actuated, 18 DOF |
| leaphand_graph_2 | Orange (240, 160, 50) | 0.6 | Index finger removed (joints 0-3), 12 actuated, 18 DOF |
| leaphand_morpho_1 | Purple (180, 80, 220) | 0.6 | All DIP links removed (joints 2,6,10,14), 12 actuated, 18 DOF |
| leaphand_morpho_2 | Red (220, 60, 60) | 0.6 | Extra DIP links added (joints 2_1,6_1,10_1,14_1), 20 actuated, 26 DOF |
| leaphand_morpho_3 | Teal (100, 200, 200) | 0.6 | Non-thumb DIP removed (joints 2,6,10), thumb kept, 13 actuated, 19 DOF |
| leaphand_graph_morpho_1 | Gold (200, 150, 50) | 0.6 | No index + elongated thumb (joints 0-3 removed, 14_1 added), 13 actuated, 19 DOF |
| Object mesh | Pink (239, 132, 167) | 0.7 | - |

---

## GUI Controls

### Grasp Selection
- **Object**: Dropdown to select from all available objects in the dataset
- **Grasp Index**: Slider to browse different grasps for the selected object

### Variant Visibility
- **Checkboxes**: Toggle each variant on/off independently
- **Show All / Hide All**: Buttons for quick toggling

### Display Options
- **Show Object Mesh**: Toggle the object mesh visibility
- **Spread Layout**: Offset variants spatially along the X axis for clearer comparison
- **Spread Distance**: Control the spacing when spread mode is enabled (0.05m - 0.4m)

---

## Q Mapping: How Joint Values Transfer Between Variants

The dataset contains grasps for the original `leaphand` (22 DOF). To render these on variants, we map joint values by **joint name matching**.

### Subset Mapping (fewer DOF variants)

For graph_1, graph_2, morpho_1, morpho_3 (18-19 DOF < 22 DOF): every variant joint exists in leaphand, so we index directly.

```python
# Precomputed at init:
src_joints = hands['leaphand'].get_joint_orders()       # length 22
dst_joints = hands['leaphand_graph_1'].get_joint_orders()  # length 18
indices = [src_joints.index(j) for j in dst_joints]

# At runtime (fast):
q_variant = q_leaphand[indices]
```

### Expand Mapping (more DOF variants)

For morpho_2 (26 DOF > 22 DOF): it has extra joints (2_1, 6_1, 10_1, 14_1) not in leaphand. Shared joints are mapped by name, and new joints copy values from source joints:

```python
COPY_MAP = {'2_1': '2', '6_1': '6', '10_1': '10', '14_1': '14'}
```

### Joint Order (pytorch-kinematics topological sort)

The q vector is NOT ordered by joint number. The actual ordering:

```
leaphand (22):     [vx,vy,vz,vr,vp,vy, '1','0','2','3','5','4','6','7','9','8','10','11','12','13','14','15']
graph_1  (18):     [vx,vy,vz,vr,vp,vy, '1','0','2','3','9','8','10','11','12','13','14','15']
graph_2  (18):     [vx,vy,vz,vr,vp,vy, '5','4','6','7','9','8','10','11','12','13','14','15']
morpho_1 (18):     [vx,vy,vz,vr,vp,vy, '1','0','3','5','4','7','9','8','11','12','13','15']
morpho_2 (26):     [vx,vy,vz,vr,vp,vy, '1','0','2','2_1','3','5','4','6','6_1','7','9','8','10','10_1','11','12','13','14','14_1','15']
morpho_3 (19):     [vx,vy,vz,vr,vp,vy, '1','0','3','5','4','7','9','8','11','12','13','14','15']
graph_morpho_1(19):[vx,vy,vz,vr,vp,vy, '5','4','6','7','9','8','10','11','12','13','14','14_1','15']
```

---

## Notes

- The script loads all 6 hand models at startup (~2s)
- Each render update calls `get_trimesh_q()` for visible variants (~0.14s per variant)
- Object meshes are cached after first load
- The script only reads data, it does not modify any files
