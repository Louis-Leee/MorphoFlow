"""
Generate LeapHand variant datasets by Q-mapping original leaphand grasps.

Reads leaphand_dataset_filtered.pt (7800 grasps, DOF=22) and generates
dataset files for each variant by mapping joint values through joint name
matching (same logic as visualization/vis_leaphand_variants.py).

Usage:
    python dataset/generate_variant_datasets.py
    python dataset/generate_variant_datasets.py --source data/CMapDataset_filtered/leaphand_dataset_filtered.pt
    python dataset/generate_variant_datasets.py --no-merge  # skip leaphand_all.pt generation
"""

import os
import sys
import argparse

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model


# ── Variant definitions ──────────────────────────────────────────────────

VARIANTS = [
    'leaphand_graph_1',
    'leaphand_graph_2',
    'leaphand_morpho_1',
    'leaphand_morpho_2',
    'leaphand_morpho_3',
    'leaphand_graph_morpho_1',
]

# Morpho 2 has extra joints not in leaphand — copy values from source joints
MORPHO_2_COPY_MAP = {'2_1': '2', '6_1': '6', '10_1': '10', '14_1': '14'}


# ── Q mapping (reused from visualization/vis_leaphand_variants.py) ───────

def build_q_mappings(hands):
    """Precompute joint index mappings from leaphand to each variant.

    For variants with FEWER joints (graph_1, graph_2, morpho_1): subset indexing.
    For variants with MORE joints (morpho_2): shared joint indexing + copy map.

    Returns:
        dict: variant_name -> mapping info dict
    """
    src_joints = hands['leaphand'].get_joint_orders()
    mappings = {}

    for variant_name in VARIANTS:
        dst_joints = hands[variant_name].get_joint_orders()
        extra_joints = [j for j in dst_joints if j not in src_joints]

        if not extra_joints:
            # Subset case: all variant joints exist in leaphand
            indices = [src_joints.index(j) for j in dst_joints]
            assert indices[:6] == list(range(6)), (
                f"Virtual joints should map 1:1 for {variant_name}, got {indices[:6]}"
            )
            mappings[variant_name] = {'type': 'subset', 'indices': indices}
            print(f"  {variant_name}: {len(dst_joints)} DOF, "
                  f"subset of leaphand ({len(src_joints)} DOF)")
        else:
            # Expand case: variant has joints not in leaphand (e.g. morpho_2)
            shared_indices = {}
            copy_indices = {}
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
                  f"{len(shared_indices)} shared + {len(copy_indices)} copied")

    return mappings


def map_q(q_leaphand, variant_name, q_mappings):
    """Map leaphand q vector to a variant's q vector by joint name matching."""
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


# ── Joint limit validation ───────────────────────────────────────────────

def validate_joint_limits(q, hand, variant_name):
    """Check if q is within URDF joint limits. Returns (clipped_q, n_clipped)."""
    lower, upper = hand.pk_chain.get_joint_limits()
    lower_t = torch.tensor(lower, dtype=q.dtype)
    upper_t = torch.tensor(upper, dtype=q.dtype)

    clipped = q.clone()
    clipped = torch.max(clipped, lower_t)
    clipped = torch.min(clipped, upper_t)
    n_clipped = (clipped != q).sum().item()
    return clipped, n_clipped


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate LeapHand variant datasets via Q mapping",
    )
    parser.add_argument(
        '--source', type=str,
        default='data/CMapDataset_filtered/leaphand_dataset_filtered.pt',
        help='Source leaphand dataset path (relative to project root)',
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='data/CMapDataset_filtered',
        help='Output directory for variant datasets (relative to project root)',
    )
    parser.add_argument(
        '--no-merge', action='store_true',
        help='Skip generating merged leaphand_all.pt',
    )
    args = parser.parse_args()

    source_path = os.path.join(ROOT_DIR, args.source)
    output_dir = os.path.join(ROOT_DIR, args.output_dir)

    # ── Load source dataset ──
    print(f"Loading source dataset: {source_path}")
    data = torch.load(source_path, map_location='cpu', weights_only=False)
    metadata = data['metadata']
    print(f"  {len(metadata)} entries, DOF={metadata[0][0].shape[0]}, "
          f"robot_name='{metadata[0][2]}'")

    # ── Load hand models ──
    print("Loading hand models...")
    hands = {}
    for name in ['leaphand'] + VARIANTS:
        hands[name] = create_hand_model(name, torch.device('cpu'))
        print(f"  {name}: DOF={hands[name].dof}")

    # ── Build Q mappings ──
    print("Building Q mappings...")
    q_mappings = build_q_mappings(hands)

    # ── Generate variant datasets ──
    all_metadata = list(metadata)  # start with original leaphand entries

    for variant_name in VARIANTS:
        print(f"\nGenerating {variant_name} dataset...")
        hand = hands[variant_name]
        variant_metadata = []
        total_clipped = 0

        for q_leaphand, object_name, robot_name in metadata:
            q_variant = map_q(q_leaphand, variant_name, q_mappings)
            q_variant, n_clipped = validate_joint_limits(q_variant, hand, variant_name)
            total_clipped += n_clipped
            variant_metadata.append((q_variant, object_name, variant_name))

        # Save individual dataset
        out_path = os.path.join(output_dir, f'{variant_name}_dataset.pt')
        torch.save({'metadata': variant_metadata}, out_path)
        print(f"  Saved {out_path}: {len(variant_metadata)} entries, DOF={variant_metadata[0][0].shape[0]}")
        if total_clipped > 0:
            print(f"  WARNING: {total_clipped} joint values clipped to URDF limits")

        all_metadata.extend(variant_metadata)

    # ── Generate merged dataset ──
    if not args.no_merge:
        merged_path = os.path.join(output_dir, 'leaphand_all.pt')
        torch.save({'metadata': all_metadata}, merged_path)
        print(f"\nMerged dataset: {merged_path}")
        print(f"  Total entries: {len(all_metadata)}")

        # Count per robot
        from collections import Counter
        counts = Counter(m[2] for m in all_metadata)
        for rn, cnt in sorted(counts.items()):
            print(f"    {rn}: {cnt}")

    print("\nDone.")


if __name__ == '__main__':
    main()
