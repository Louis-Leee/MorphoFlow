"""
LeapHand variant comparison visualization.

Loads original leaphand grasps from the dataset and simultaneously displays
the same grasp on all 5 LeapHand variants (original, graph_1, graph_2, morpho_1, morpho_2)
using joint name-based Q mapping.

Usage:
    python visualization/vis_leaphand_variants.py
    python visualization/vis_leaphand_variants.py --port 8081
    python visualization/vis_leaphand_variants.py --dataset data/CMapDataset_filtered/cmap_full_dataset.pt
"""

import os
import sys
import time
import argparse
from collections import defaultdict

import torch
import trimesh
import numpy as np
import viser

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model


# ── Variant configuration ────────────────────────────────────────────────

VARIANT_CONFIG = {
    'leaphand': {
        'color': (70, 130, 230),
        'opacity': 1.0,
        'label': 'leaphand (Original)',
    },
    'leaphand_graph_1': {
        'color': (50, 200, 100),
        'opacity': 0.6,
        'label': 'graph_1 (No Middle)',
    },
    'leaphand_graph_2': {
        'color': (240, 160, 50),
        'opacity': 0.6,
        'label': 'graph_2 (No Index)',
    },
    'leaphand_morpho_1': {
        'color': (180, 80, 220),
        'opacity': 0.6,
        'label': 'morpho_1 (Shortened)',
    },
    'leaphand_morpho_2': {
        'color': (220, 60, 60),
        'opacity': 0.6,
        'label': 'morpho_2 (Elongated)',
    },
    'leaphand_morpho_3': {
        'color': (100, 200, 200),
        'opacity': 0.6,
        'label': 'morpho_3 (Short non-thumb)',
    },
    'leaphand_graph_morpho_1': {
        'color': (200, 150, 50),
        'opacity': 0.6,
        'label': 'graph_morpho_1 (No Index+Long Thumb)',
    },
}

OBJECT_COLOR = (239, 132, 167)

EMPTY_VERTS = np.zeros((3, 3), dtype=np.float32)
EMPTY_FACES = np.zeros((1, 3), dtype=np.int32)


# ── Data loading ─────────────────────────────────────────────────────────

def load_grasps(dataset_path):
    """Load dataset and return leaphand grasps grouped by object name.

    Returns:
        dict: object_name -> list of q tensors (leaphand DOF=22)
    """
    data = torch.load(dataset_path, map_location=torch.device('cpu'), weights_only=False)
    metadata = data['metadata']

    grasps_by_object = defaultdict(list)
    for entry in metadata:
        q, object_name, robot_name = entry[0], entry[1], entry[2]
        if robot_name == 'leaphand':
            grasps_by_object[object_name].append(q)

    print(f"Loaded {sum(len(v) for v in grasps_by_object.values())} leaphand grasps "
          f"across {len(grasps_by_object)} objects")
    return dict(grasps_by_object)


# ── Q mapping ────────────────────────────────────────────────────────────

# Morpho 2 has extra joints not in leaphand — copy values from source joints
MORPHO_2_COPY_MAP = {'2_1': '2', '6_1': '6', '10_1': '10', '14_1': '14'}


def build_q_mappings(hands):
    """Precompute joint index mappings from leaphand to each variant.

    For variants with FEWER joints (graph_1, graph_2, morpho_1): subset indexing.
    For variants with MORE joints (morpho_2): shared joint indexing + copy map.

    Returns:
        dict: variant_name -> mapping info (list of indices, or dict with expand info)
    """
    src_joints = hands['leaphand'].get_joint_orders()
    mappings = {}

    for variant_name in VARIANT_CONFIG:
        if variant_name == 'leaphand':
            continue
        dst_joints = hands[variant_name].get_joint_orders()

        # Check if all dst joints exist in src (subset case)
        extra_joints = [j for j in dst_joints if j not in src_joints]

        if not extra_joints:
            # Subset case: all variant joints exist in leaphand
            indices = [src_joints.index(j) for j in dst_joints]
            assert indices[:6] == list(range(6)), (
                f"Virtual joints should map 1:1 for {variant_name}, got {indices[:6]}"
            )
            mappings[variant_name] = {'type': 'subset', 'indices': indices}
            print(f"  {variant_name}: {len(dst_joints)} DOF, "
                  f"maps {len(indices)} joints from leaphand ({len(src_joints)} DOF)")
        else:
            # Expand case: variant has joints not in leaphand (e.g. morpho_2)
            # Build: shared_indices (dst_idx -> src_idx for shared joints)
            #        copy_indices (dst_idx -> dst_idx of source joint for new joints)
            shared_indices = {}  # dst_pos -> src_pos
            copy_indices = {}    # dst_pos -> dst_pos (of the source joint to copy from)
            for dst_idx, j in enumerate(dst_joints):
                if j in src_joints:
                    shared_indices[dst_idx] = src_joints.index(j)
                elif j in MORPHO_2_COPY_MAP:
                    copy_src_name = MORPHO_2_COPY_MAP[j]
                    copy_indices[dst_idx] = dst_joints.index(copy_src_name)
                else:
                    raise ValueError(
                        f"Joint '{j}' in {variant_name} not in leaphand and no copy mapping"
                    )
            mappings[variant_name] = {
                'type': 'expand',
                'dst_dof': len(dst_joints),
                'shared_indices': shared_indices,
                'copy_indices': copy_indices,
            }
            print(f"  {variant_name}: {len(dst_joints)} DOF (expand), "
                  f"{len(shared_indices)} shared + {len(copy_indices)} copied "
                  f"from leaphand ({len(src_joints)} DOF)")

    return mappings


def map_q(q_leaphand, variant_name, q_mappings):
    """Map leaphand q vector to a variant's q vector by joint name matching."""
    if variant_name == 'leaphand':
        return q_leaphand

    mapping = q_mappings[variant_name]

    if mapping['type'] == 'subset':
        return q_leaphand[mapping['indices']]

    # Expand case: build full q vector
    q_variant = torch.zeros(mapping['dst_dof'], dtype=q_leaphand.dtype)
    for dst_idx, src_idx in mapping['shared_indices'].items():
        q_variant[dst_idx] = q_leaphand[src_idx]
    for dst_idx, copy_dst_idx in mapping['copy_indices'].items():
        q_variant[dst_idx] = q_variant[copy_dst_idx]
    return q_variant


# ── Object mesh cache ────────────────────────────────────────────────────

class ObjectMeshCache:
    def __init__(self):
        self._cache = {}

    def get(self, object_name):
        if object_name not in self._cache:
            parts = object_name.split('+')
            path = os.path.join(
                ROOT_DIR,
                f'data/data_urdf/object/{parts[0]}/{parts[1]}/{parts[1]}.stl',
            )
            if os.path.exists(path):
                self._cache[object_name] = trimesh.load_mesh(path)
            else:
                print(f"Warning: Object mesh not found: {path}")
                self._cache[object_name] = None
        return self._cache[object_name]


# ── Visualization ────────────────────────────────────────────────────────

def launch_viewer(grasps_by_object, hands, q_mappings, args):
    """Launch interactive viser viewer for variant comparison."""
    server = viser.ViserServer(host='0.0.0.0', port=args.port)
    print(f"Viser server running at http://localhost:{args.port}")

    object_names = sorted(grasps_by_object.keys())
    mesh_cache = ObjectMeshCache()

    # ── GUI: Grasp Selection ──
    with server.gui.add_folder("Grasp Selection", expand_by_default=True):
        object_dropdown = server.gui.add_dropdown(
            "Object", options=object_names, initial_value=object_names[0],
        )
        max_grasps = len(grasps_by_object[object_names[0]]) - 1
        grasp_slider = server.gui.add_slider(
            "Grasp Index", min=0, max=max(max_grasps, 0), step=1, initial_value=0,
        )

    # ── GUI: Variant Visibility ──
    visibility_cbs = {}
    with server.gui.add_folder("Variant Visibility", expand_by_default=True):
        for variant_name, cfg in VARIANT_CONFIG.items():
            cb = server.gui.add_checkbox(cfg['label'], initial_value=True)
            visibility_cbs[variant_name] = cb

        show_all_btn = server.gui.add_button("Show All")
        hide_all_btn = server.gui.add_button("Hide All")

    # ── GUI: Display Options ──
    with server.gui.add_folder("Display Options", expand_by_default=False):
        show_object_cb = server.gui.add_checkbox("Show Object Mesh", initial_value=True)
        spread_cb = server.gui.add_checkbox("Spread Layout", initial_value=False)
        spread_dist = server.gui.add_slider(
            "Spread Distance", min=0.05, max=0.4, step=0.01, initial_value=0.15,
        )

    # ── Scene update ──
    def on_update(_=None):
        obj_name = object_dropdown.value
        grasp_idx = int(grasp_slider.value)
        grasps = grasps_by_object[obj_name]
        grasp_idx = min(grasp_idx, len(grasps) - 1)
        q_leaphand = grasps[grasp_idx]

        print(f"{obj_name} | grasp {grasp_idx}/{len(grasps)-1}")

        # ── Object mesh ──
        obj_mesh = mesh_cache.get(obj_name)
        if obj_mesh is not None and show_object_cb.value:
            server.scene.add_mesh_simple(
                'object', obj_mesh.vertices, obj_mesh.faces,
                color=OBJECT_COLOR, opacity=0.7,
            )
        else:
            server.scene.add_mesh_simple(
                'object', EMPTY_VERTS, EMPTY_FACES, visible=False,
            )

        # ── Variant meshes ──
        variant_names = list(VARIANT_CONFIG.keys())
        for i, variant_name in enumerate(variant_names):
            scene_name = f'robot/{variant_name}'
            cfg = VARIANT_CONFIG[variant_name]

            if not visibility_cbs[variant_name].value:
                server.scene.add_mesh_simple(
                    scene_name, EMPTY_VERTS, EMPTY_FACES, visible=False,
                )
                continue

            q_variant = map_q(q_leaphand, variant_name, q_mappings)
            mesh_result = hands[variant_name].get_trimesh_q(q_variant)
            mesh = mesh_result['visual']
            verts = np.array(mesh.vertices, dtype=np.float32)

            # Spatial offset in spread mode
            if spread_cb.value:
                offset_x = (i - (len(variant_names) - 1) / 2.0) * spread_dist.value
                verts = verts.copy()
                verts[:, 0] += offset_x

            server.scene.add_mesh_simple(
                scene_name, verts, mesh.faces,
                color=cfg['color'], opacity=cfg['opacity'],
            )

    # ── Callbacks ──
    def on_object_change(_):
        obj_name = object_dropdown.value
        new_max = len(grasps_by_object[obj_name]) - 1
        grasp_slider.max = max(new_max, 0)
        if grasp_slider.value > new_max:
            grasp_slider.value = 0
        on_update()

    object_dropdown.on_update(on_object_change)
    grasp_slider.on_update(on_update)
    show_object_cb.on_update(on_update)
    spread_cb.on_update(on_update)
    spread_dist.on_update(on_update)

    for cb in visibility_cbs.values():
        cb.on_update(on_update)

    @show_all_btn.on_click
    def _(_):
        for cb in visibility_cbs.values():
            cb.value = True
        on_update()

    @hide_all_btn.on_click
    def _(_):
        for cb in visibility_cbs.values():
            cb.value = False
        on_update()

    # ── Initial render ──
    on_update()

    print("Viewer ready. Use controls to browse grasps and toggle variants.")
    while True:
        time.sleep(1)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LeapHand variant comparison visualization",
    )
    parser.add_argument(
        '--port', type=int, default=8080,
        help='Viser server port (default: 8080)',
    )
    parser.add_argument(
        '--dataset', type=str,
        default='data/CMapDataset_filtered/cmap_full_dataset.pt',
        help='Dataset path relative to project root',
    )
    args = parser.parse_args()

    dataset_path = os.path.join(ROOT_DIR, args.dataset)

    print("Loading dataset...")
    grasps_by_object = load_grasps(dataset_path)

    print("Loading hand models...")
    hands = {}
    for variant_name in VARIANT_CONFIG:
        hands[variant_name] = create_hand_model(variant_name, torch.device('cpu'))
        print(f"  {variant_name}: DOF={hands[variant_name].dof}")

    print("Building Q mappings...")
    q_mappings = build_q_mappings(hands)

    print("Launching viewer...")
    launch_viewer(grasps_by_object, hands, q_mappings, args)


if __name__ == '__main__':
    main()
