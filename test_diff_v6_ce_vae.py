"""
Cross-Embodiment Diffusion V6 (Fourier RPE) evaluation script.

Uses RobotGraphV6CE with classifier-free guidance at inference.
Supports batch evaluation across multiple robot hands from a single config.

Usage:
    # Single hand (backward compatible)
    python test_diff_v6_ce_vae.py --config config/test_diff_v6_ce_vae.yaml

    # Multiple hands via CLI
    python test_diff_v6_ce_vae.py --config config/test_diff_v6_ce_vae.yaml \
        --hands allegro barrett shadowhand leaphand

    # Override checkpoint and GPU
    python test_diff_v6_ce_vae.py --config config/test_diff_v6_ce_vae.yaml \
        --hands allegro barrett shadowhand leaphand \
        --ckpt graph_exp/diff_v6_ce_vae/ckpt/epoch=299.ckpt --gpu 0

    # Use JAX on CPU to save GPU memory for IK:
    JAX_PLATFORM_NAME=cpu python test_diff_v6_ce_vae.py --config config/test_diff_v6_ce_vae.yaml
"""

import os
import argparse
from omegaconf import OmegaConf

# Reuse all evaluation infrastructure from V5 test script
from test_diff_v5_ce_vae import (
    test as _test_v5,
    load_checkpoint,
    eval_single_hand,
    save_hand_results,
    extract_fingertip_joints,
    print_banner,
    print_hand_results,
    print_summary_table,
    FINGERTIP_JOINTS,
    LEAPHAND_TIP_MAPPING,
    LEAPHAND_LINK_WEIGHTS,
    LEAPHAND_LOCKED_JOINTS,
)

# Import V6 model instead of V4
from model.tro_graph_v6_ce import RobotGraphV6CE

import json
import time
import tqdm
import torch
import numpy as np
import jax
import jax.numpy as jnp

from dataset.CMapDataset import create_dataloader
from utils.hand_model import create_hand_model
from utils.pyroki_ik import PyrokiRetarget
from utils.optimization import process_transform

# Import all tip/weight mappings from V5
from test_diff_v5_ce_vae import (
    LEAPHAND_GRAPH_1_TIP_MAPPING,
    LEAPHAND_GRAPH_1_LINK_WEIGHTS,
    LEAPHAND_GRAPH_2_TIP_MAPPING,
    LEAPHAND_GRAPH_2_LINK_WEIGHTS,
    LEAPHAND_MORPHO_1_TIP_MAPPING,
    LEAPHAND_MORPHO_1_LINK_WEIGHTS,
    LEAPHAND_MORPHO_2_TIP_MAPPING,
    LEAPHAND_MORPHO_2_LINK_WEIGHTS,
    LEAPHAND_MORPHO_3_TIP_MAPPING,
    LEAPHAND_MORPHO_3_LINK_WEIGHTS,
    LEAPHAND_GRAPH_MORPHO_1_TIP_MAPPING,
    LEAPHAND_GRAPH_MORPHO_1_LINK_WEIGHTS,
    BOLD, DIM, CYAN, GREEN, YELLOW, MAGENTA, RESET,
)


def test(config, hand_overrides=None, ckpt_override=None, gpu_override=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gpu = gpu_override if gpu_override is not None else config.test.gpu
    hands = hand_overrides or [config.test.embodiment]

    if ckpt_override is None:
        ckpt_list = [config.test.ckpt]
    elif isinstance(ckpt_override, str):
        ckpt_list = [ckpt_override]
    else:
        ckpt_list = ckpt_override

    base_save_dir = config.test.save_dir

    with open("data/data_urdf/robot/urdf_assets_meta.json", "r") as f:
        robot_urdf_meta = json.load(f)

    batch_size = config.dataset.batch_size
    split_batch_size = config.test.split_batch_size

    for ckpt_idx, ckpt in enumerate(ckpt_list):
        if len(ckpt_list) > 1:
            ckpt_name = os.path.splitext(os.path.basename(ckpt))[0]
            ckpt_save_dir = os.path.join(base_save_dir, ckpt_name)
            print(f"\n{BOLD}=== Checkpoint {ckpt_idx + 1}/{len(ckpt_list)}: {ckpt_name} ==={RESET}")
        else:
            ckpt_save_dir = base_save_dir

        # Load V6 model
        print(f"{DIM}Building V6 model (Fourier RPE)...{RESET}")
        model = RobotGraphV6CE(**config.model).to(device)
        load_checkpoint(model, ckpt)
        model.eval()

        all_results = {}

        for hand_idx, hand_name in enumerate(hands):
            dataset_cfg = OmegaConf.create(
                {**OmegaConf.to_container(config.dataset, resolve=True),
                 "robot_names": [hand_name]}
            )
            dataloader = create_dataloader(dataset_cfg, is_train=False)
            num_objects = len(dataloader)

            print_banner(hand_name, ckpt, gpu, num_objects, batch_size)

            hand = create_hand_model(hand_name, device)
            urdf_path = robot_urdf_meta["urdf_path"][hand_name]
            target_links = list(hand.links_pc.keys())

            ik_target_links = target_links
            link_weights = None
            locked_joints = None
            if hand_name.startswith('leaphand'):
                if hand_name == 'leaphand':
                    tip_mapping = LEAPHAND_TIP_MAPPING
                    link_weights_config = LEAPHAND_LINK_WEIGHTS
                elif hand_name == 'leaphand_graph_1':
                    tip_mapping = LEAPHAND_GRAPH_1_TIP_MAPPING
                    link_weights_config = LEAPHAND_GRAPH_1_LINK_WEIGHTS
                elif hand_name == 'leaphand_graph_2':
                    tip_mapping = LEAPHAND_GRAPH_2_TIP_MAPPING
                    link_weights_config = LEAPHAND_GRAPH_2_LINK_WEIGHTS
                elif hand_name == 'leaphand_morpho_1':
                    tip_mapping = LEAPHAND_MORPHO_1_TIP_MAPPING
                    link_weights_config = LEAPHAND_MORPHO_1_LINK_WEIGHTS
                elif hand_name == 'leaphand_morpho_2':
                    tip_mapping = LEAPHAND_MORPHO_2_TIP_MAPPING
                    link_weights_config = LEAPHAND_MORPHO_2_LINK_WEIGHTS
                elif hand_name == 'leaphand_morpho_3':
                    tip_mapping = LEAPHAND_MORPHO_3_TIP_MAPPING
                    link_weights_config = LEAPHAND_MORPHO_3_LINK_WEIGHTS
                elif hand_name == 'leaphand_graph_morpho_1':
                    tip_mapping = LEAPHAND_GRAPH_MORPHO_1_TIP_MAPPING
                    link_weights_config = LEAPHAND_GRAPH_MORPHO_1_LINK_WEIGHTS
                else:
                    tip_mapping = {}
                    link_weights_config = {}
                ik_target_links = [tip_mapping.get(link, link) for link in target_links]
                link_weights = [link_weights_config.get(link, 1.0) for link in ik_target_links]
                locked_joints = LEAPHAND_LOCKED_JOINTS

            ik_solver = PyrokiRetarget(
                urdf_path, ik_target_links,
                hand_joint_names=hand.get_joint_orders(),
                link_weights=link_weights,
                locked_joint_indices=locked_joints,
            )
            batch_retarget = jax.jit(ik_solver.solve_retarget)

            results = eval_single_hand(
                model, dataloader, hand, batch_retarget, target_links,
                batch_size, split_batch_size, gpu, device,
            )

            if len(hands) > 1:
                hand_save_dir = os.path.join(ckpt_save_dir, hand_name)
            else:
                hand_save_dir = ckpt_save_dir

            save_hand_results(results, hand_save_dir)
            print_hand_results(
                hand_name, results["success_rate"],
                results["diversity"], results["gen_time"],
            )

            all_results[hand_name] = results

        if len(hands) > 1:
            print_summary_table(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-Embodiment Diffusion V6 (Fourier RPE) evaluation (single or batch multi-hand)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/test_diff_v6_ce_vae.yaml",
        help="Base config file",
    )
    parser.add_argument(
        "--hands",
        nargs="+",
        default=None,
        help="Robot hands to evaluate (overrides config.test.embodiment)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="+",
        default=None,
        help="Override checkpoint path(s). Multiple paths iterate sequentially.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Override GPU for Isaac Gym validation",
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    test(config, hand_overrides=args.hands, ckpt_override=args.ckpt, gpu_override=args.gpu)
