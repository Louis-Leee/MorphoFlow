"""
Diffusion V3 evaluation script.

Uses RobotGraphV3 with FlashAttentionDenoiserNoEdge (no edge computation).
Supports batch evaluation across multiple robot hands from a single config.

Usage:
    # Single hand (backward compatible)
    python test_diff_v3.py --config config/test_diff_v3.yaml

    # Multiple hands via CLI
    python test_diff_v3.py --config config/test_diff_v3.yaml \
        --hands allegro barrett shadowhand

    # Override checkpoint and GPU
    python test_diff_v3.py --config config/test_diff_v3.yaml \
        --hands allegro barrett shadowhand \
        --ckpt graph_exp/diff_v3/ckpt/epoch=299.ckpt --gpu 0

    # Use JAX on CPU to save GPU memory for IK:
    JAX_PLATFORM_NAME=cpu python test_diff_v3.py --config config/test_diff_v3.yaml
"""

import os
import jax
import json
import time
import tqdm
import torch
import argparse
import numpy as np
import jax.numpy as jnp
from omegaconf import OmegaConf

from dataset.CMapDataset import create_dataloader
from model.tro_graph_v3 import RobotGraphV3
from utils.hand_model import create_hand_model
from utils.pyroki_ik import PyrokiRetarget
from utils.optimization import process_transform
from validation.validate_utils import validate_isaac


# ── Fingertip joint extraction for pure-rotation joints ─────────────────
# These joints have zero positional Jacobian, so IK leaves them unchanged.
# We extract their angles from the diffusion model's predicted SE3 rotations.

# Robotiq 3-finger has a fixed rpy offset in joint_4 origin
ROBOTIQ_JOINT4_OFFSET = -0.436332312999  # rpy="0 -0.436332312999 0" in URDF

FINGERTIP_JOINTS = {
    'leaphand': {
        # (parent_link, child_link): q_index
        ('dip', 'fingertip'): 9,           # joint 3
        ('dip_2', 'fingertip_2'): 13,      # joint 7
        ('dip_3', 'fingertip_3'): 17,      # joint 11
        ('thumb_dip', 'thumb_fingertip'): 21,  # joint 15
    },
    'leaphand_graph_1': {
        # No middle finger (joints 4-7 removed)
        ('dip', 'fingertip'): 9,           # joint 3 (unchanged)
        ('dip_3', 'fingertip_3'): 13,      # joint 11 -> q_idx 13 (was 17)
        ('thumb_dip', 'thumb_fingertip'): 17,  # joint 15 -> q_idx 17 (was 21)
    },
    'leaphand_graph_2': {
        # No index finger (joints 0-3 removed)
        ('dip_2', 'fingertip_2'): 9,       # joint 7 -> q_idx 9
        ('dip_3', 'fingertip_3'): 13,      # joint 11 -> q_idx 13
        ('thumb_dip', 'thumb_fingertip'): 17,  # joint 15 -> q_idx 17
    },
    'ezgripper': {
        # L2 joints are pure Y-axis rotation, extract from SE3 rotation
        ('left_ezgripper_finger_L1_1', 'left_ezgripper_finger_L2_1'): 7,
        ('left_ezgripper_finger_L1_2', 'left_ezgripper_finger_L2_2'): 9,
    },
    'robotiq_3finger': {
        # joint_4 are pure Y-axis rotation with fixed offset, extract from SE3 rotation
        ('gripper_fingerA_med', 'gripper_fingerA_dist'): 8,
        ('gripper_fingerB_med', 'gripper_fingerB_dist'): 12,
        ('gripper_fingerC_med', 'gripper_fingerC_dist'): 16,
    },
    'xhand': {
        # joint2 are pure rotation joints - thumb uses Y-axis, others use X-axis
        ('right_hand_thumb_rota_link1', 'right_hand_thumb_rota_link2'): 8,
        ('right_hand_index_rota_link1', 'right_hand_index_rota_link2'): 11,
        ('right_hand_mid_link1', 'right_hand_mid_link2'): 13,
        ('right_hand_ring_link1', 'right_hand_ring_link2'): 15,
        ('right_hand_pinky_link1', 'right_hand_pinky_link2'): 17,
    },
}

# ── LeapHand tip mapping (use fixed joints instead of revolute for IK) ──
# NOTE: thumb_fingertip is NOT included because extra_thumb_tip_head has Z offset
# in the opposite direction (-0.015 vs +0.015 for other fingers)
LEAPHAND_TIP_MAPPING = {
    'fingertip': 'extra_index_tip_head',
    'fingertip_2': 'extra_middle_tip_head',
    'fingertip_3': 'extra_ring_tip_head',
}

# ── LeapHand Graph 1 tip mapping (no middle finger) ──
LEAPHAND_GRAPH_1_TIP_MAPPING = {
    'fingertip': 'extra_index_tip_head',
    'fingertip_3': 'extra_ring_tip_head',
    # No fingertip_2 (middle finger removed)
}

# ── LeapHand Graph 2 tip mapping (no index finger) ──
LEAPHAND_GRAPH_2_TIP_MAPPING = {
    'fingertip_2': 'extra_middle_tip_head',
    'fingertip_3': 'extra_ring_tip_head',
    # No fingertip (index finger removed)
}

# ── LeapHand per-link IK weights ──
# 实验: 疯狂提高 palm 权重，让 IK 优先对齐手掌
LEAPHAND_LINK_WEIGHTS = {
    'palm_lower': 1.0,
    'extra_ring_tip_head': 0.5,
    'extra_middle_tip_head': 0.7,
    'extra_index_tip_head': 0.8,
    'thumb_fingertip': 0.8,
}

# ── LeapHand Graph 1 per-link IK weights (no middle finger) ──
LEAPHAND_GRAPH_1_LINK_WEIGHTS = {
    'palm_lower': 1.0,
    'extra_ring_tip_head': 0.5,
    'extra_index_tip_head': 0.8,
    'thumb_fingertip': 0.8,
    # No extra_middle_tip_head (middle finger removed)
}

# ── LeapHand Graph 2 per-link IK weights (no index finger) ──
LEAPHAND_GRAPH_2_LINK_WEIGHTS = {
    'palm_lower': 1.0,
    'extra_ring_tip_head': 0.5,
    'extra_middle_tip_head': 0.7,
    'thumb_fingertip': 0.8,
    # No extra_index_tip_head (index finger removed)
}

# ── LeapHand locked joints (keep at initial value during IK) ──
# Set to empty to disable joint locking
LEAPHAND_LOCKED_JOINTS = []

def extract_fingertip_joints(predict_q, transform_dict, robot_name):
    """
    Extract fingertip joint angles from diffusion SE3 rotation.

    For joints that are pure rotation (don't affect link position), the IK
    solver cannot optimize them. Instead, we compute the joint angle from
    the relative rotation between parent and child links in the diffusion output.

    Different robots have different joint axes:
    - Leaphand: Z-axis rotation (0, 0, -1), angle = atan2(R[1,0], R[0,0])
    - EZGripper: Y-axis rotation (0, 1, 0), angle = atan2(R[0,2], R[2,2])
    - Robotiq 3-finger: Y-axis rotation with fixed offset, angle = atan2(R[0,2], R[2,2]) - offset
    - XHand: mixed axes - thumb uses Y-axis, other fingers use X-axis
    """
    if robot_name not in FINGERTIP_JOINTS:
        return predict_q

    is_ezgripper = (robot_name == 'ezgripper')
    is_robotiq = (robot_name == 'robotiq_3finger')
    is_xhand = (robot_name == 'xhand')

    for (parent, child), q_idx in FINGERTIP_JOINTS[robot_name].items():
        if parent not in transform_dict or child not in transform_dict:
            continue
        R_parent = transform_dict[parent][:, :3, :3]  # (B, 3, 3)
        R_child = transform_dict[child][:, :3, :3]
        # Relative rotation: R_rel = R_parent^T @ R_child
        R_rel = torch.bmm(R_parent.transpose(-1, -2), R_child)

        if is_ezgripper:
            # EZGripper: Y-axis rotation, axis=(0,1,0), no offset
            # R = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
            # angle = atan2(R[0,2], R[2,2]) = atan2(sin, cos)
            angle = torch.atan2(R_rel[:, 0, 2], R_rel[:, 2, 2])
            predict_q[:, q_idx] = angle
        elif is_robotiq:
            # Robotiq 3-finger: Y-axis rotation with fixed offset
            # URDF has rpy="0 -0.436332312999 0" offset in joint_4 origin
            # R_rel = Ry(offset + joint_angle), so joint_angle = extracted - offset
            angle_with_offset = torch.atan2(R_rel[:, 0, 2], R_rel[:, 2, 2])
            predict_q[:, q_idx] = angle_with_offset - ROBOTIQ_JOINT4_OFFSET
        elif is_xhand:
            # XHand: mixed axes - thumb uses Y-axis, other fingers use X-axis
            if 'thumb' in parent:
                # Y-axis rotation: Ry = [[cos,0,sin],[0,1,0],[-sin,0,cos]]
                # angle = atan2(R[0,2], R[2,2])
                angle = torch.atan2(R_rel[:, 0, 2], R_rel[:, 2, 2])
            else:
                # X-axis rotation: Rx = [[1,0,0],[0,cos,-sin],[0,sin,cos]]
                # angle = atan2(R[2,1], R[1,1])
                angle = torch.atan2(R_rel[:, 2, 1], R_rel[:, 1, 1])
            predict_q[:, q_idx] = angle
        else:
            # Leaphand: Z-axis rotation, axis=(0, 0, -1)
            # For Rz(θ): R[1,0] = sin(θ), R[0,0] = cos(θ)
            # For thumb (q_idx=21): remove π offset in URDF origin before extracting angle
            if q_idx == 21:  # thumb_fingertip has rpy="0 0 3.14159" offset
                R_offset = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                                        dtype=R_rel.dtype, device=R_rel.device)
                R_joint_only = torch.bmm(
                    R_offset.unsqueeze(0).expand(R_rel.shape[0], -1, -1).transpose(-1, -2),
                    R_rel
                )
                angle = torch.atan2(R_joint_only[:, 1, 0], R_joint_only[:, 0, 0])
            else:
                angle = torch.atan2(R_rel[:, 1, 0], R_rel[:, 0, 0])
            # Negate because joint axis is (0, 0, -1)
            predict_q[:, q_idx] = -angle

    return predict_q


# ── Pretty terminal output ──────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

BOX_TL = "\u2554"  # ╔
BOX_TR = "\u2557"  # ╗
BOX_BL = "\u255a"  # ╚
BOX_BR = "\u255d"  # ╝
BOX_H = "\u2550"   # ═
BOX_V = "\u2551"   # ║
BOX_ML = "\u2560"  # ╠
BOX_MR = "\u2563"  # ╣
BOX_CROSS = "\u256c"  # ╬
BOX_MT = "\u2566"  # ╦
BOX_MB = "\u2569"  # ╩

LINE_TL = "\u250c"  # ┌
LINE_TR = "\u2510"  # ┐
LINE_BL = "\u2514"  # └
LINE_BR = "\u2518"  # ┘
LINE_H = "\u2500"   # ─
LINE_V = "\u2502"   # │


def _box_line(text, width, left=BOX_V, right=BOX_V):
    """Pad text to width inside box chars."""
    visible_len = len(text.replace(BOLD, "").replace(DIM, "").replace(CYAN, "")
                       .replace(GREEN, "").replace(YELLOW, "").replace(MAGENTA, "")
                       .replace(RESET, ""))
    padding = width - visible_len
    return f"{left} {text}{' ' * padding} {right}"


def print_banner(hand_name, ckpt, gpu, num_objects, batch_size):
    W = 52
    title = "EVALUATING: " + hand_name.upper()
    title_pad = (W - len(title)) // 2
    title_line = BOLD + " " * title_pad + title
    ckpt_short = ckpt if len(ckpt) <= W else "..." + ckpt[-(W - 3):]
    ckpt_line = "Checkpoint: " + DIM + ckpt_short + RESET + CYAN
    info_line = "GPU: %d  |  Objects: %d  |  Batch: %d" % (gpu, num_objects, batch_size)
    print()
    print(CYAN + BOX_TL + BOX_H * (W + 2) + BOX_TR + RESET)
    print(CYAN + _box_line(title_line, W) + RESET)
    print(CYAN + BOX_ML + BOX_H * (W + 2) + BOX_MR + RESET)
    print(CYAN + _box_line(ckpt_line, W) + RESET)
    print(CYAN + _box_line(info_line, W) + RESET)
    print(CYAN + BOX_BL + BOX_H * (W + 2) + BOX_BR + RESET)
    print()


def print_hand_results(hand_name, success_rate, diversity, gen_time):
    W = 44
    title = f"{hand_name} Results"
    title_pad = (W - len(title)) // 2
    print()
    print(f"{GREEN}{LINE_TL}{LINE_H * title_pad} {BOLD}{title}{RESET}{GREEN} {LINE_H * (W - title_pad - len(title))}{LINE_TR}{RESET}")
    print(f"{GREEN}{LINE_V}{RESET}  Success Rate: {BOLD}{success_rate:.1%}{RESET}{' ' * (W - 22)}{GREEN}{LINE_V}{RESET}")
    print(f"{GREEN}{LINE_V}{RESET}  Diversity:    {BOLD}{diversity:.4f}{RESET}{' ' * (W - 22)}{GREEN}{LINE_V}{RESET}")
    time_str = f"{gen_time:.3f} s/grasp" if gen_time > 0 else "N/A"
    print(f"{GREEN}{LINE_V}{RESET}  Gen Time:     {BOLD}{time_str}{RESET}{' ' * (W - 14 - len(time_str))}{GREEN}{LINE_V}{RESET}")
    print(f"{GREEN}{LINE_BL}{LINE_H * (W + 2)}{LINE_BR}{RESET}")


def print_summary_table(all_results):
    """Print final summary table across all hands."""
    print(f"\n{MAGENTA}{BOLD}")
    print(f"{BOX_TL}{BOX_H * 56}{BOX_TR}")
    title = "EVALUATION SUMMARY"
    pad = (56 - len(title)) // 2
    print(f"{BOX_V}{' ' * pad}{title}{' ' * (56 - pad - len(title))}{BOX_V}")
    print(f"{BOX_ML}{BOX_H * 20}{BOX_MT}{BOX_H * 11}{BOX_MT}{BOX_H * 11}{BOX_MT}{BOX_H * 11}{BOX_MR}")
    print(f"{BOX_V} {'Hand':<18} {BOX_V} {'Success':>9} {BOX_V} {'Diverse':>9} {BOX_V} {'Time':>9} {BOX_V}")
    print(f"{BOX_ML}{BOX_H * 20}{BOX_CROSS}{BOX_H * 11}{BOX_CROSS}{BOX_H * 11}{BOX_CROSS}{BOX_H * 11}{BOX_MR}")

    for hand, r in all_results.items():
        sr = f"{r['success_rate']:.1%}"
        div = f"{r['diversity']:.4f}"
        tm = f"{r['gen_time']:.3f}s" if r['gen_time'] > 0 else "N/A"
        print(f"{BOX_V} {hand:<18} {BOX_V} {sr:>9} {BOX_V} {div:>9} {BOX_V} {tm:>9} {BOX_V}")

    print(f"{BOX_BL}{BOX_H * 20}{BOX_MB}{BOX_H * 11}{BOX_MB}{BOX_H * 11}{BOX_MB}{BOX_H * 11}{BOX_BR}")
    print(f"{RESET}")


# ── Core evaluation logic ────────────────────────────────────────────────

def load_checkpoint(model, ckpt_path):
    """Load checkpoint, handling both Lightning and raw formats."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        # Lightning checkpoint: keys prefixed with "model."
        state_dict = {
            k.replace("model.", "", 1): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
    elif "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    return model


def eval_single_hand(
    model, dataloader, hand, batch_retarget, target_links,
    batch_size, split_batch_size, gpu, device,
):
    """Evaluate a single robot hand. Returns dict with metrics and vis data."""
    success_dict = {}
    diversity_dict = {}
    vis_info = []
    total_inference_time = 0
    total_grasp_num = 0
    warmed_up = False

    for batch_id, batch in tqdm.tqdm(enumerate(dataloader), desc="  Batches"):
        transform_dict = {}
        data_count = 0
        predict_q_list = []
        initial_q_list = []
        initial_se3_list = []
        object_pc_list = []

        while data_count != batch_size:
            split_num = min(batch_size - data_count, split_batch_size)
            initial_q = batch["initial_q"][data_count : data_count + split_num].to(device)
            initial_se3 = batch["initial_se3"][data_count : data_count + split_num].to(device)
            object_pc = batch["object_pc"][data_count : data_count + split_num].to(device)
            robot_links_pc = batch["robot_links_pc"][data_count : data_count + split_num]
            split_batch = {
                "robot_name": batch["robot_name"],
                "object_name": batch["object_name"],
                "initial_q": initial_q,
                "initial_se3": initial_se3,
                "object_pc": object_pc,
                "robot_links_pc": robot_links_pc,
            }
            data_count += split_num

            time_start = time.time()
            with torch.no_grad():
                all_step_poses_dict = model.inference(split_batch)

            # IK with Pyroki
            clean_robot_pose = all_step_poses_dict[0]
            optim_transform = process_transform(hand.pk_chain, clean_robot_pose)
            initial_q_jnp = jnp.array(initial_q.cpu().numpy())
            target_pos_list = [optim_transform[name] for name in target_links]
            target_pos = torch.stack(target_pos_list, dim=1)

            target_pos_jnp = jnp.array(target_pos.detach().cpu().numpy())
            predict_q_jnp = batch_retarget(
                initial_q=initial_q_jnp, target_pos=target_pos_jnp
            )
            jax.block_until_ready(predict_q_jnp)
            time_end = time.time()

            if warmed_up:
                total_inference_time += time_end - time_start
                total_grasp_num += split_num
            else:
                warmed_up = True

            predict_q = torch.from_numpy(np.array(predict_q_jnp)).to(
                device=device, dtype=initial_q.dtype
            )

            # Extract fingertip joint angles from diffusion rotation
            predict_q = extract_fingertip_joints(
                predict_q, clean_robot_pose, batch["robot_name"]
            )

            initial_q_list.append(initial_q)
            initial_se3_list.append(initial_se3)
            predict_q_list.append(predict_q)
            object_pc_list.append(object_pc)
            for diffuse_step, pred_robot_pose in all_step_poses_dict.items():
                if diffuse_step not in transform_dict:
                    transform_dict[diffuse_step] = []
                transform_dict[diffuse_step].append(pred_robot_pose)

        # Isaac simulation validation
        all_predict_q = torch.cat(predict_q_list, dim=0)

        success, isaac_q = validate_isaac(
            batch["robot_name"],
            batch["object_name"],
            all_predict_q,
            gpu=gpu,
        )
        success_dict[batch["object_name"]] = success

        # Diversity
        success_q = all_predict_q[success]
        diversity_dict[batch["object_name"]] = success_q

        for diffuse_step, transform_list in transform_dict.items():
            transform_batch = {}
            for transform in transform_list:
                for k, v in transform.items():
                    transform_batch[k] = (
                        v
                        if k not in transform_batch
                        else torch.cat((transform_batch[k], v), dim=0)
                    )
            transform_dict[diffuse_step] = transform_batch

        vis_info.append(
            {
                "robot_name": batch["robot_name"],
                "object_name": batch["object_name"],
                "initial_q": torch.cat(initial_q_list, dim=0),
                "initial_se3": torch.cat(initial_se3_list, dim=0),
                "predict_q": torch.cat(predict_q_list, dim=0),
                "object_pc": torch.cat(object_pc_list, dim=0),
                "predict_transform": transform_dict,
                "success": success,
                "isaac_q": isaac_q,
            }
        )

    # Compute aggregate metrics
    total_success = sum(s.sum() for s in success_dict.values())
    total_sum = sum(len(s) for s in success_dict.values())
    success_rate = (total_success / total_sum).item() if total_sum > 0 else 0.0

    all_success_q_list = [v for v in diversity_dict.values() if len(v) > 0]
    if all_success_q_list:
        all_success_q = torch.cat(all_success_q_list, dim=0)
        diversity = torch.std(all_success_q, dim=0).mean().item()
    else:
        diversity = 0.0

    gen_time = total_inference_time / total_grasp_num if total_grasp_num > 0 else 0.0

    return {
        "success_rate": success_rate,
        "diversity": diversity,
        "gen_time": gen_time,
        "success_dict": success_dict,
        "diversity_dict": diversity_dict,
        "vis_info": vis_info,
        "total_success": total_success,
        "total_sum": total_sum,
    }


def save_hand_results(results, save_dir):
    """Save vis.pt and res.txt for a single hand."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(results["vis_info"], os.path.join(save_dir, "vis.pt"))

    output_path = os.path.join(save_dir, "res.txt")
    with open(output_path, "w") as f:
        for obj, obj_res in results["success_dict"].items():
            line = f"{obj}: {obj_res.sum() / len(obj_res)}\n"
            print(f"    {line}", end="")
            f.write(line)

        line = f"Total success rate: {results['success_rate']:.4f}\n"
        print(f"    {BOLD}{line}{RESET}", end="")
        f.write(line)

        line = f"Total diversity: {results['diversity']:.4f}\n"
        print(f"    {line}", end="")
        f.write(line)

        if results["gen_time"] > 0:
            line = f"Grasp generation time: {results['gen_time']:.4f} s.\n"
        else:
            line = "Grasp generation time: N/A (no warmed-up batches).\n"
        print(f"    {line}", end="")
        f.write(line)


# ── Main entry point ─────────────────────────────────────────────────────

def test(config, hand_overrides=None, ckpt_override=None, gpu_override=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply CLI overrides
    ckpt = ckpt_override or config.test.ckpt
    gpu = gpu_override if gpu_override is not None else config.test.gpu
    hands = hand_overrides or [config.test.embodiment]

    # Load model ONCE
    print(f"{DIM}Building model...{RESET}")
    model = RobotGraphV3(**config.model).to(device)
    load_checkpoint(model, ckpt)
    model.eval()

    with open("data/data_urdf/robot/urdf_assets_meta.json", "r") as f:
        robot_urdf_meta = json.load(f)

    batch_size = config.dataset.batch_size
    split_batch_size = config.test.split_batch_size

    all_results = {}

    for hand_idx, hand_name in enumerate(hands):
        # Override dataset config for this hand
        dataset_cfg = OmegaConf.create(
            {**OmegaConf.to_container(config.dataset, resolve=True),
             "robot_names": [hand_name]}
        )
        dataloader = create_dataloader(dataset_cfg, is_train=False)
        num_objects = len(dataloader)

        print_banner(hand_name, ckpt, gpu, num_objects, batch_size)

        # Hand model + IK
        hand = create_hand_model(hand_name, device)
        urdf_path = robot_urdf_meta["urdf_path"][hand_name]
        target_links = list(hand.links_pc.keys())

        # Apply LeapHand tip mapping, weights, and locked joints
        ik_target_links = target_links
        link_weights = None
        locked_joints = None
        if hand_name.startswith('leaphand'):
            # Select config based on specific variant
            if hand_name == 'leaphand':
                tip_mapping = LEAPHAND_TIP_MAPPING
                link_weights_config = LEAPHAND_LINK_WEIGHTS
            elif hand_name == 'leaphand_graph_1':
                tip_mapping = LEAPHAND_GRAPH_1_TIP_MAPPING
                link_weights_config = LEAPHAND_GRAPH_1_LINK_WEIGHTS
            elif hand_name == 'leaphand_graph_2':
                tip_mapping = LEAPHAND_GRAPH_2_TIP_MAPPING
                link_weights_config = LEAPHAND_GRAPH_2_LINK_WEIGHTS
            else:
                # Unknown variant, use empty config
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

        # Determine save directory
        if len(hands) > 1:
            hand_save_dir = os.path.join(config.test.save_dir, hand_name)
        else:
            hand_save_dir = config.test.save_dir

        save_hand_results(results, hand_save_dir)
        print_hand_results(
            hand_name, results["success_rate"],
            results["diversity"], results["gen_time"],
        )

        all_results[hand_name] = results

    # Final summary (only when evaluating multiple hands)
    if len(hands) > 1:
        print_summary_table(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diffusion V3 evaluation (single or batch multi-hand)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/test_diff_v3.yaml",
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
        default=None,
        help="Override checkpoint path",
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
