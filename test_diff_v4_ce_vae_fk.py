"""
Cross-Embodiment Diffusion V4 (VAE) evaluation with FK Steering.

Adds Feynman-Kac Diffusion Steering (particle-based SMC) to the grasp
inference pipeline. When fk_steering.enabled=false (default), behavior
is identical to test_diff_v4_ce_vae.py.

Usage:
    # Without FK Steering (identical to original)
    python test_diff_v4_ce_vae_fk.py --config config/test_diff_v4_ce_vae_fk.yaml \
        --hands allegro

    # With FK Steering enabled
    python test_diff_v4_ce_vae_fk.py --config config/test_diff_v4_ce_vae_fk.yaml \
        --hands allegro fk_steering.enabled=true fk_steering.num_particles=4

    # Use JAX on CPU to save GPU memory for IK:
    JAX_PLATFORM_NAME=cpu python test_diff_v4_ce_vae_fk.py ...
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

from pytorch3d.ops import knn_points

from dataset.CMapDataset import create_dataloader
from model.tro_graph_v4_ce import RobotGraphV4CE
from utils.hand_model import create_hand_model
from utils.pyroki_ik import PyrokiRetarget
from utils.optimization import process_transform
from utils.rotation import vector_to_matrix
from utils.func_utils import farthest_point_sampling
from validation.validate_utils import validate_isaac


# ── Fingertip joint extraction for pure-rotation joints ─────────────────
# These joints have zero positional Jacobian, so IK leaves them unchanged.
# We extract their angles from the diffusion model's predicted SE3 rotations.

# Robotiq 3-finger has a fixed rpy offset in joint_4 origin
ROBOTIQ_JOINT4_OFFSET = -0.436332312999  # rpy="0 -0.436332312999 0" in URDF

FINGERTIP_JOINTS = {
    'leaphand': {
        ('dip', 'fingertip'): 9,
        ('dip_2', 'fingertip_2'): 13,
        ('dip_3', 'fingertip_3'): 17,
        ('thumb_dip', 'thumb_fingertip'): 21,
    },
    'leaphand_graph_1': {
        ('dip', 'fingertip'): 9,
        ('dip_3', 'fingertip_3'): 13,
        ('thumb_dip', 'thumb_fingertip'): 17,
    },
    'leaphand_graph_2': {
        ('dip_2', 'fingertip_2'): 9,
        ('dip_3', 'fingertip_3'): 13,
        ('thumb_dip', 'thumb_fingertip'): 17,
    },
    'leaphand_morpho_1': {
        ('pip', 'fingertip'): 8,
        ('pip_2', 'fingertip_2'): 11,
        ('pip_3', 'fingertip_3'): 14,
        ('thumb_pip', 'thumb_fingertip'): 17,
    },
    'leaphand_morpho_2': {
        ('dip_1', 'fingertip'): 10,
        ('dip_2_1', 'fingertip_2'): 15,
        ('dip_3_1', 'fingertip_3'): 20,
        ('thumb_dip_1', 'thumb_fingertip'): 25,
    },
    'leaphand_morpho_3': {
        ('pip', 'fingertip'): 8,
        ('pip_2', 'fingertip_2'): 11,
        ('pip_3', 'fingertip_3'): 14,
        ('thumb_dip', 'thumb_fingertip'): 18,
    },
    'leaphand_graph_morpho_1': {
        ('dip_2', 'fingertip_2'): 9,
        ('dip_3', 'fingertip_3'): 13,
        ('thumb_dip_1', 'thumb_fingertip'): 18,
    },
    'ezgripper': {
        ('left_ezgripper_finger_L1_1', 'left_ezgripper_finger_L2_1'): 7,
        ('left_ezgripper_finger_L1_2', 'left_ezgripper_finger_L2_2'): 9,
    },
    'robotiq_3finger': {
        ('gripper_fingerA_med', 'gripper_fingerA_dist'): 8,
        ('gripper_fingerB_med', 'gripper_fingerB_dist'): 12,
        ('gripper_fingerC_med', 'gripper_fingerC_dist'): 16,
    },
    'xhand': {
        ('right_hand_thumb_rota_link1', 'right_hand_thumb_rota_link2'): 8,
        ('right_hand_index_rota_link1', 'right_hand_index_rota_link2'): 11,
        ('right_hand_mid_link1', 'right_hand_mid_link2'): 13,
        ('right_hand_ring_link1', 'right_hand_ring_link2'): 15,
        ('right_hand_pinky_link1', 'right_hand_pinky_link2'): 17,
    },
}


def extract_fingertip_joints(predict_q, transform_dict, robot_name):
    """
    Extract fingertip joint angles from diffusion SE3 rotation.
    Identical to test_diff_v4_ce_vae.py version.
    """
    if robot_name not in FINGERTIP_JOINTS:
        return predict_q

    is_ezgripper = (robot_name == 'ezgripper')
    is_robotiq = (robot_name == 'robotiq_3finger')
    is_xhand = (robot_name == 'xhand')
    is_morpho_1 = (robot_name == 'leaphand_morpho_1')
    is_morpho_3 = (robot_name == 'leaphand_morpho_3')

    for (parent, child), q_idx in FINGERTIP_JOINTS[robot_name].items():
        if parent not in transform_dict or child not in transform_dict:
            continue
        R_parent = transform_dict[parent][:, :3, :3]
        R_child = transform_dict[child][:, :3, :3]
        R_rel = torch.bmm(R_parent.transpose(-1, -2), R_child)

        if is_ezgripper:
            angle = torch.atan2(R_rel[:, 0, 2], R_rel[:, 2, 2])
            predict_q[:, q_idx] = angle
        elif is_robotiq:
            angle_with_offset = torch.atan2(R_rel[:, 0, 2], R_rel[:, 2, 2])
            predict_q[:, q_idx] = angle_with_offset - ROBOTIQ_JOINT4_OFFSET
        elif is_xhand:
            if 'thumb' in parent:
                angle = torch.atan2(R_rel[:, 0, 2], R_rel[:, 2, 2])
            else:
                angle = torch.atan2(R_rel[:, 2, 1], R_rel[:, 1, 1])
            predict_q[:, q_idx] = angle
        elif is_morpho_1:
            is_thumb = (child == 'thumb_fingertip')
            if is_thumb:
                R_origin = torch.tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
                                        dtype=R_rel.dtype, device=R_rel.device)
            else:
                R_origin = torch.tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]],
                                        dtype=R_rel.dtype, device=R_rel.device)
            R_joint = torch.bmm(
                R_origin.T.unsqueeze(0).expand(R_rel.shape[0], -1, -1),
                R_rel
            )
            angle = torch.atan2(R_joint[:, 1, 0], R_joint[:, 0, 0])
            predict_q[:, q_idx] = -angle
        elif is_morpho_3:
            is_thumb = (child == 'thumb_fingertip')
            if is_thumb:
                R_offset = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                                        dtype=R_rel.dtype, device=R_rel.device)
                R_joint_only = torch.bmm(
                    R_offset.unsqueeze(0).expand(R_rel.shape[0], -1, -1).transpose(-1, -2),
                    R_rel
                )
                angle = torch.atan2(R_joint_only[:, 1, 0], R_joint_only[:, 0, 0])
            else:
                R_origin = torch.tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]],
                                        dtype=R_rel.dtype, device=R_rel.device)
                R_joint = torch.bmm(
                    R_origin.T.unsqueeze(0).expand(R_rel.shape[0], -1, -1),
                    R_rel
                )
                angle = torch.atan2(R_joint[:, 1, 0], R_joint[:, 0, 0])
            predict_q[:, q_idx] = -angle
        else:
            if child == 'thumb_fingertip':
                R_offset = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                                        dtype=R_rel.dtype, device=R_rel.device)
                R_joint_only = torch.bmm(
                    R_offset.unsqueeze(0).expand(R_rel.shape[0], -1, -1).transpose(-1, -2),
                    R_rel
                )
                angle = torch.atan2(R_joint_only[:, 1, 0], R_joint_only[:, 0, 0])
            else:
                angle = torch.atan2(R_rel[:, 1, 0], R_rel[:, 0, 0])
            predict_q[:, q_idx] = -angle

    return predict_q


# ── LeapHand tip mapping (use fixed joints instead of revolute for IK) ──
LEAPHAND_TIP_MAPPING = {
    'fingertip': 'extra_index_tip_head',
    'fingertip_2': 'extra_middle_tip_head',
    'fingertip_3': 'extra_ring_tip_head',
}
LEAPHAND_GRAPH_1_TIP_MAPPING = {
    'fingertip': 'extra_index_tip_head',
    'fingertip_3': 'extra_ring_tip_head',
}
LEAPHAND_GRAPH_2_TIP_MAPPING = {
    'fingertip_2': 'extra_middle_tip_head',
    'fingertip_3': 'extra_ring_tip_head',
}
LEAPHAND_LINK_WEIGHTS = {
    'palm_lower': 1.0,
    'extra_ring_tip_head': 0.5,
    'extra_middle_tip_head': 0.7,
    'extra_index_tip_head': 0.8,
    'thumb_fingertip': 0.8,
}
LEAPHAND_GRAPH_1_LINK_WEIGHTS = {
    'palm_lower': 1.0,
    'extra_ring_tip_head': 0.5,
    'extra_index_tip_head': 0.8,
    'thumb_fingertip': 0.8,
}
LEAPHAND_GRAPH_2_LINK_WEIGHTS = {
    'palm_lower': 1.0,
    'extra_ring_tip_head': 0.5,
    'extra_middle_tip_head': 0.7,
    'thumb_fingertip': 0.8,
}
LEAPHAND_MORPHO_1_TIP_MAPPING = {
    'fingertip': 'extra_index_tip_head',
    'fingertip_2': 'extra_middle_tip_head',
    'fingertip_3': 'extra_ring_tip_head',
}
LEAPHAND_MORPHO_1_LINK_WEIGHTS = {
    'palm_lower': 1.0,
    'extra_ring_tip_head': 0.5,
    'extra_middle_tip_head': 0.7,
    'extra_index_tip_head': 0.8,
    'thumb_fingertip': 0.8,
}
LEAPHAND_MORPHO_2_TIP_MAPPING = {
    'fingertip': 'extra_index_tip_head',
    'fingertip_2': 'extra_middle_tip_head',
    'fingertip_3': 'extra_ring_tip_head',
}
LEAPHAND_MORPHO_2_LINK_WEIGHTS = {
    'palm_lower': 1.0,
    'extra_ring_tip_head': 0.5,
    'extra_middle_tip_head': 0.7,
    'extra_index_tip_head': 0.8,
    'thumb_fingertip': 0.8,
}
LEAPHAND_MORPHO_3_TIP_MAPPING = {
    'fingertip': 'extra_index_tip_head',
    'fingertip_2': 'extra_middle_tip_head',
    'fingertip_3': 'extra_ring_tip_head',
}
LEAPHAND_MORPHO_3_LINK_WEIGHTS = {
    'palm_lower': 1.0,
    'extra_ring_tip_head': 0.5,
    'extra_middle_tip_head': 0.7,
    'extra_index_tip_head': 0.8,
    'thumb_fingertip': 0.8,
}
LEAPHAND_GRAPH_MORPHO_1_TIP_MAPPING = {
    'fingertip_2': 'extra_middle_tip_head',
    'fingertip_3': 'extra_ring_tip_head',
}
LEAPHAND_GRAPH_MORPHO_1_LINK_WEIGHTS = {
    'palm_lower': 1.0,
    'extra_ring_tip_head': 0.5,
    'extra_middle_tip_head': 0.7,
    'thumb_fingertip': 0.8,
}
LEAPHAND_LOCKED_JOINTS = []


# ── Pretty terminal output ──────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

BOX_TL = "\u2554"
BOX_TR = "\u2557"
BOX_BL = "\u255a"
BOX_BR = "\u255d"
BOX_H = "\u2550"
BOX_V = "\u2551"
BOX_ML = "\u2560"
BOX_MR = "\u2563"
BOX_CROSS = "\u256c"
BOX_MT = "\u2566"
BOX_MB = "\u2569"

LINE_TL = "\u250c"
LINE_TR = "\u2510"
LINE_BL = "\u2514"
LINE_BR = "\u2518"
LINE_H = "\u2500"
LINE_V = "\u2502"


def _box_line(text, width, left=BOX_V, right=BOX_V):
    visible_len = len(text.replace(BOLD, "").replace(DIM, "").replace(CYAN, "")
                       .replace(GREEN, "").replace(YELLOW, "").replace(MAGENTA, "")
                       .replace(RESET, ""))
    padding = width - visible_len
    return f"{left} {text}{' ' * padding} {right}"


def print_banner(hand_name, ckpt, gpu, num_objects, batch_size,
                  fk_enabled=False, fk_cfg=None):
    W = 52
    color = YELLOW if fk_enabled else CYAN
    title = "EVALUATING: " + hand_name.upper()
    title_pad = (W - len(title)) // 2
    title_line = BOLD + " " * title_pad + title
    ckpt_short = ckpt if len(ckpt) <= W else "..." + ckpt[-(W - 3):]
    ckpt_line = "Checkpoint: " + DIM + ckpt_short + RESET + color
    info_line = "GPU: %d  |  Objects: %d  |  Batch: %d" % (gpu, num_objects, batch_size)
    print()
    print(color + BOX_TL + BOX_H * (W + 2) + BOX_TR + RESET)
    print(color + _box_line(title_line, W) + RESET)
    if fk_enabled:
        fk_title = BOLD + ">>> FK STEERING ENABLED <<<"
        fk_pad = (W - 26) // 2
        fk_line = " " * fk_pad + fk_title
        print(color + BOX_ML + BOX_H * (W + 2) + BOX_MR + RESET)
        print(color + _box_line(fk_line, W) + RESET)
        if fk_cfg is not None:
            k = fk_cfg.num_particles
            lmbda = fk_cfg.lmbda
            pot = fk_cfg.potential_type
            fk_detail = f"k={k}  lambda={lmbda}  potential={pot}"
            fk_detail_pad = (W - len(fk_detail)) // 2
            print(color + _box_line(" " * fk_detail_pad + fk_detail, W) + RESET)
            rng = f"resample: [{fk_cfg.resample_start}, {fk_cfg.resample_end}] every {fk_cfg.resample_frequency}"
            rng_pad = (W - len(rng)) // 2
            print(color + _box_line(" " * rng_pad + rng, W) + RESET)
    print(color + BOX_ML + BOX_H * (W + 2) + BOX_MR + RESET)
    print(color + _box_line(ckpt_line, W) + RESET)
    print(color + _box_line(info_line, W) + RESET)
    print(color + BOX_BL + BOX_H * (W + 2) + BOX_BR + RESET)
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


# ── FK Steering: Object normal loading ───────────────────────────────────

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_object_normals(object_name, num_points=512):
    """
    Load object point cloud normals from the pre-computed .pt file.
    The .pt files store [512, 6] tensors: [x,y,z, nx,ny,nz].
    """
    name = object_name.split('+')
    object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
    data = torch.load(object_path, map_location='cpu')
    normals = data[:, 3:6]  # [512, 3]
    return normals


# ── FK Steering: Surface point cloud from FK(q) ─────────────────────────

def get_surface_pc(hand, q=None):
    """
    Get subsampled surface point cloud from FK(q) for reward computation.

    Palm/base links kept at full resolution; finger links FPS-downsampled
    to 128 (64 for shadowhand) to reduce KNN cost.

    :param hand: HandModel instance
    :param q: (DOF,) joint values (euler representation)
    :return: (N_total, 3) concatenated surface point cloud
    """
    if q is None:
        q = hand.get_canonical_q()
    all_pc_se3, _ = hand.get_transformed_links_pc(q)
    pc_list = []
    for link_name, pts in all_pc_se3.items():
        if hand.robot_name == "barrett" and link_name == "bh_base_link":
            pc_list.append(pts)
            continue
        if hand.robot_name == "allegro" and "base_link" in link_name:
            pc_list.append(pts)
            continue
        if "palm" in link_name:
            pc_list.append(pts)
            continue
        target_num = 64 if hand.robot_name == "shadowhand" else 128
        if pts.shape[0] > target_num:
            sampled_pts, _ = farthest_point_sampling(pts, target_num)
            pc_list.append(sampled_pts)
        else:
            pc_list.append(pts)
    return torch.cat(pc_list, dim=0)


# ── FK Steering: Per-sample reward functions (single-sample wrappers) ────

def ERF_loss_single(obj_pcd_nor, hand_pcd):
    """Penetration penalty for a single sample. [1,N,6]+[1,N,3] → scalar."""
    obj_pcd = obj_pcd_nor[:, :, :3]
    obj_nor = obj_pcd_nor[:, :, 3:6]
    knn_result = knn_points(hand_pcd, obj_pcd, K=1, return_nn=True)
    distances = knn_result.dists.sqrt()
    indices = knn_result.idx
    hand_obj_points = torch.gather(obj_pcd, 1, indices.expand(-1, -1, 3))
    hand_obj_normals = torch.gather(obj_nor, 1, indices.expand(-1, -1, 3))
    hand_obj_signs = ((hand_obj_points - hand_pcd) * hand_obj_normals).sum(dim=2)
    hand_obj_signs = (hand_obj_signs > 0.).float()
    collision_value = (hand_obj_signs * distances.squeeze(2)).max(dim=1).values
    return collision_value.item()


def SPF_loss_single(dis_points, obj_pcd, thres_dis=0.02):
    """Fingertip proximity for a single sample. [1,N,3]+[1,N,3] → scalar."""
    dis_pred = knn_points(dis_points.float(), obj_pcd.float()).dists[:, :, 0]
    small_mask = dis_pred < thres_dis ** 2
    qualified_dist = dis_pred.sqrt() * small_mask.float()
    count = small_mask.float().sum(dim=1).clamp(min=1.0)
    return (qualified_dist.sum(dim=1) / count).item()


def SRF_loss_single(points):
    """Self-collision penalty for a single sample. [1,N,3] → scalar."""
    dis = (points.unsqueeze(2) - points.unsqueeze(1) + 1e-13).square().sum(3).sqrt()
    dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)
    penalty = (0.02 - dis).clamp(min=0)
    return penalty.sum().item()


# ── FK Steering: IK-based grasp reward computation ───────────────────────

def compute_grasp_reward(
    x0_pred_trans, x0_pred_rot, centroids, scale,
    link_names, valid_links, hand,
    obj_pcd, obj_pcd_normal, reward_weights,
    batch_retarget, target_links, initial_q,
):
    """
    Compute per-sample grasp quality reward using IK-based FK poses.

    Pipeline: x0_pred → denormalize → SE3 → process_transform → IK → FK(q)
              → get_surface_pc, get_keypoints, get_dis_keypoints → rewards

    :param x0_pred_trans: [B, max_link, 3] normalized translation
    :param x0_pred_rot: [B, max_link, 3] axis-angle rotation
    :param centroids: [B, 1, 3] object centroid
    :param scale: [B, 1, 1] object scale
    :param link_names: list of link names
    :param valid_links: number of valid links
    :param hand: HandModel instance
    :param obj_pcd: [B, N_obj, 3] object point cloud
    :param obj_pcd_normal: [B, N_obj, 3] object normals
    :param reward_weights: dict with ERF, SPF, SRF weights
    :param batch_retarget: JIT-compiled JAX IK solver
    :param target_links: list of link names for IK targets
    :param initial_q: [B, DOF] initial joint angles (IK seed)
    :return: [B] reward values (higher = better)
    """
    device = x0_pred_trans.device
    B = x0_pred_trans.shape[0]

    # ---- Step 1: Denormalize and build SE3 ----
    pred_trans_world = x0_pred_trans[:, :valid_links] * scale + centroids
    pred_rot = x0_pred_rot[:, :valid_links]
    pred_pose = torch.cat([pred_trans_world, pred_rot], dim=-1)  # [B, L, 6]
    link_se3 = vector_to_matrix(pred_pose)  # [B, L, 4, 4]

    # ---- Step 2: Build pose dict and process_transform ----
    predict_link_pose_dict = {}
    for link_id, link_name in enumerate(link_names):
        predict_link_pose_dict[link_name] = link_se3[:, link_id]
    optim_transform = process_transform(hand.pk_chain, predict_link_pose_dict)
    target_pos_list = [optim_transform[name] for name in target_links]
    target_pos = torch.stack(target_pos_list, dim=1)  # [B, num_targets, 3]

    # ---- Step 3: IK (JAX) ----
    initial_q_jnp = jnp.array(initial_q.cpu().numpy())
    target_pos_jnp = jnp.array(target_pos.detach().cpu().numpy())
    predict_q_jnp = batch_retarget(
        initial_q=initial_q_jnp, target_pos=target_pos_jnp
    )
    jax.block_until_ready(predict_q_jnp)
    predict_q = torch.from_numpy(np.array(predict_q_jnp)).to(
        device=device, dtype=x0_pred_trans.dtype
    )

    # ---- Step 4: Extract fingertip joints from diffusion rotation ----
    predict_q = extract_fingertip_joints(
        predict_q, predict_link_pose_dict, hand.robot_name
    )

    # ---- Step 5: Per-sample FK → surface PC, keypoints ----
    w_erf = reward_weights.get('ERF', 1.0)
    w_spf = reward_weights.get('SPF', 1.0)
    w_srf = reward_weights.get('SRF', 0.5)
    obj_pcd_nor = torch.cat([obj_pcd, obj_pcd_normal], dim=-1).float()

    reward = torch.zeros(B, device=device)
    for i in range(B):
        hand_pcd_i = get_surface_pc(hand, q=predict_q[i]).unsqueeze(0).float()
        obj_i = obj_pcd_nor[i:i + 1]
        obj_pcd_i = obj_pcd[i:i + 1]

        erf = ERF_loss_single(obj_i, hand_pcd_i)

        dis_kp_parts = hand.get_dis_keypoints(q=predict_q[i])
        if dis_kp_parts:
            dis_kp_i = torch.cat(dis_kp_parts, dim=0).unsqueeze(0)
            spf = SPF_loss_single(dis_kp_i, obj_pcd_i)
        else:
            spf = 0.0

        kp_parts = hand.get_keypoints(q=predict_q[i])
        if kp_parts:
            kp_i = torch.cat(kp_parts, dim=0).unsqueeze(0)
            srf = SRF_loss_single(kp_i)
        else:
            srf = 0.0

        reward[i] = -(w_erf * erf + w_spf * spf + w_srf * srf)

    return reward


# ── FK Steering: DDIM loop with particle resampling ─────────────────────

@torch.no_grad()
def inference_with_fk_steering(model, batch, hand, fk_cfg, device,
                               batch_retarget, target_links):
    """
    Run DDIM denoising with FK Steering (particle-based SMC resampling).

    Replicates model._inference_unconditioned() DDIM loop with resampling
    hooks injected at intermediate steps. Rewards are computed via IK → FK(q).

    :param model: RobotGraphV4CE instance
    :param batch: dict with object_pc, robot_name, robot_links_pc, etc.
    :param hand: HandModel instance with keypoints
    :param fk_cfg: OmegaConf config for fk_steering section
    :param device: torch device
    :return: all_diffuse_step_poses_dict (same format as model.inference())
    """
    # ── Setup (replicate model.inference()) ──
    object_pc = batch["object_pc"]
    S = object_pc.shape[0]  # original batch size
    dtype = object_pc.dtype

    robot_name = batch["robot_name"]
    link_names = list(batch["robot_links_pc"][0].keys())
    valid_links = model.robot_links[robot_name]

    with torch.no_grad():
        normal_pc, centroids, scale = model._normalize_pc_(object_pc)

    if model.inference_mode == "no_object":
        object_nodes = torch.zeros(
            [S, model.object_patch, sum(model.V_object_dims)],
            device=device, dtype=dtype,
        )
    elif getattr(model, 'object_encoder_type', 'vqvae') == "triposg_vae":
        object_pc_normal = batch.get("object_pc_normal", torch.zeros_like(object_pc))
        if object_pc_normal.device != device:
            object_pc_normal = object_pc_normal.to(device)
        object_nodes = model._encode_object_vae(normal_pc, object_pc_normal, scale)
    else:
        with torch.no_grad():
            object_tokens = model.vqvae.encode(normal_pc)
        object_nodes = torch.cat(
            [
                object_tokens["xyz"],
                scale.expand(-1, model.object_patch, -1),
                object_tokens["z_q"],
            ],
            dim=-1,
        )

    # ── FK Steering config ──
    k = fk_cfg.num_particles
    potential_type = fk_cfg.potential_type
    lmbda = fk_cfg.lmbda
    resample_start = fk_cfg.resample_start
    resample_end = fk_cfg.resample_end
    resample_frequency = fk_cfg.resample_frequency
    adaptive = fk_cfg.adaptive_resampling
    reward_weights = OmegaConf.to_container(fk_cfg.reward_weights, resolve=True)

    # Build resampling schedule (DDIM step indices)
    resample_schedule = set(range(resample_start, resample_end + 1, resample_frequency))

    # ── Expand batch: S samples → S*k particles ──
    B = S * k

    # Repeat conditioning tensors: each original sample repeated k times contiguously
    object_nodes_exp = object_nodes.repeat_interleave(k, dim=0)  # [S*k, P, D]
    dummy_object_nodes = torch.zeros_like(object_nodes_exp)
    centroids_exp = centroids.repeat_interleave(k, dim=0)  # [S*k, 1, 3]
    scale_exp = scale.repeat_interleave(k, dim=0)  # [S*k, 1, 1]

    # Object PC + normals for reward computation
    obj_pcd = batch["object_pc"].repeat_interleave(k, dim=0).to(device)  # [S*k, N, 3]
    # Load normals from .pt file (validation batch doesn't include normals)
    obj_normals_single = load_object_normals(batch["object_name"]).to(device)  # [512, 3]
    obj_pcd_normal = obj_normals_single.unsqueeze(0).expand(S, -1, -1)  # [S, 512, 3]
    obj_pcd_normal = obj_pcd_normal.repeat_interleave(k, dim=0)  # [S*k, 512, 3]

    # Expand initial_q for IK seeding inside reward computation
    initial_q_exp = batch["initial_q"].to(device).repeat_interleave(k, dim=0)  # [S*k, DOF]

    # Link embeds — need to prepare for expanded batch
    # Create a modified batch dict with expanded size
    link_robot_embeds = model._prepare_link_embeds(batch, B, device, dtype)

    # Initialize noisy latents (independent noise per particle!)
    noisy_V_R_trans = torch.randn(
        [B, model.max_link_node, 3], device=device, dtype=dtype
    )
    noisy_V_R_rot = torch.randn(
        [B, model.max_link_node, 3], device=device, dtype=dtype
    )
    noisy_V_R = torch.cat(
        [noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds], dim=-1
    )

    # ── FK Steering state ──
    prev_rewards = torch.zeros(B, device=device)  # [S*k]
    total_resample_count = 0

    # ── DDIM Loop ──
    model.noise_scheduler.set_timesteps(model.ddim_steps, device=device)
    ddim_t = model.noise_scheduler.timesteps
    all_diffuse_step_poses_dict = {}

    for i, diffuse_step in enumerate(ddim_t):
        diffuse_step = diffuse_step.item()
        t_batch = torch.full((B,), diffuse_step, dtype=torch.long, device=device)

        # ── CFG: two passes ──
        if model.guidance_scale != 1.0:
            pred_uncond = model.denoiser(
                dummy_object_nodes, noisy_V_R, t_batch, skip_or=True,
            )
            pred_cond = model.denoiser(
                object_nodes_exp, noisy_V_R, t_batch, skip_or=False,
            )
            pred_link_pose_noise = (
                pred_uncond
                + model.guidance_scale * (pred_cond - pred_uncond)
            )
        else:
            pred_link_pose_noise = model.denoiser(
                object_nodes_exp, noisy_V_R, t_batch,
            )

        pred_link_trans_noise = pred_link_pose_noise[:, :, :3]
        pred_link_rot_noise = pred_link_pose_noise[:, :, 3:]

        # ── Compute x0_pred ──
        a_bar_t = model.noise_scheduler.alphas_cumprod[diffuse_step]
        if i == len(ddim_t) - 1:
            a_bar_prev = torch.tensor(1.0, device=device, dtype=dtype)
        else:
            a_bar_prev = model.noise_scheduler.alphas_cumprod[ddim_t[i + 1]]

        x_0_trans = (
            noisy_V_R_trans - (1 - a_bar_t).sqrt() * pred_link_trans_noise
        ) / a_bar_t.sqrt()
        x_0_rot = (
            noisy_V_R_rot - (1 - a_bar_t).sqrt() * pred_link_rot_noise
        ) / a_bar_t.sqrt()

        # ── DDIM update ──
        sigma_t = model.eta * torch.sqrt(
            ((1 - a_bar_prev) / (1 - a_bar_t)) * (1 - a_bar_t / a_bar_prev)
        )
        ddim_coeff = torch.sqrt(1 - a_bar_prev - sigma_t**2)

        if i == len(ddim_t) - 1:
            z_trans = torch.zeros_like(x_0_trans)
            z_rot = torch.zeros_like(x_0_rot)
        else:
            z_trans = torch.randn_like(x_0_trans)
            z_rot = torch.randn_like(x_0_rot)

        x_prev_trans = (
            a_bar_prev.sqrt() * x_0_trans
            + ddim_coeff * pred_link_trans_noise
            + sigma_t * z_trans * model.noise_lambda
        )
        x_prev_rot = (
            a_bar_prev.sqrt() * x_0_rot
            + ddim_coeff * pred_link_rot_noise
            + sigma_t * z_rot * model.noise_lambda
        )

        noisy_V_R_trans = x_prev_trans
        noisy_V_R_rot = x_prev_rot

        # ── FK Steering: Resample particles ──
        if i in resample_schedule or i == len(ddim_t) - 1:
            # Compute rewards from x0_pred
            rewards = compute_grasp_reward(
                x_0_trans, x_0_rot, centroids_exp, scale_exp,
                link_names, valid_links, hand,
                obj_pcd, obj_pcd_normal, reward_weights,
                batch_retarget, target_links, initial_q_exp,
            )  # [S*k]

            # Reshape to [S, k] for grouped resampling
            rewards_grouped = rewards.view(S, k)
            prev_rewards_grouped = prev_rewards.view(S, k)

            # Compute importance weights per group
            if potential_type == 'diff':
                diffs = rewards_grouped - prev_rewards_grouped
                weights = torch.exp(lmbda * diffs)
            elif potential_type == 'max':
                rewards_grouped = torch.max(rewards_grouped, prev_rewards_grouped)
                weights = torch.exp(lmbda * rewards_grouped)
            elif potential_type == 'add':
                rewards_grouped = rewards_grouped + prev_rewards_grouped
                weights = torch.exp(lmbda * rewards_grouped)
            elif potential_type == 'rt':
                weights = torch.exp(lmbda * rewards_grouped)
            else:
                raise ValueError(f"Unknown potential_type: {potential_type}")

            weights = torch.clamp(weights, 0, 1e10)
            weights[torch.isnan(weights)] = 0.0

            # Determine whether to resample (per group, using ESS)
            do_resample = torch.ones(S, dtype=torch.bool, device=device)
            if adaptive and i != len(ddim_t) - 1:
                # Compute ESS per group
                norm_w = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-10)
                ess = 1.0 / (norm_w.pow(2).sum(dim=1))  # [S]
                do_resample = ess < 0.5 * k

            if do_resample.any():
                # Multinomial resampling per group
                for s_idx in range(S):
                    if not do_resample[s_idx]:
                        continue
                    w = weights[s_idx]  # [k]
                    if w.sum() < 1e-10:
                        continue  # degenerate weights, skip
                    indices = torch.multinomial(w, k, replacement=True)  # [k]

                    # Reindex particles for this group
                    base = s_idx * k
                    src_indices = base + indices
                    noisy_V_R_trans[base:base + k] = noisy_V_R_trans[src_indices]
                    noisy_V_R_rot[base:base + k] = noisy_V_R_rot[src_indices]

                    # Update previous rewards for resampled particles
                    rewards[base:base + k] = rewards[src_indices]

                resampled_count = do_resample.sum().item()
                total_resample_count += resampled_count
                if resampled_count > 0:
                    avg_r = rewards.view(S, k).max(dim=1).values.mean().item()
                    print(f"  {YELLOW}[FK]{RESET} step {i:3d}: "
                          f"resampled {resampled_count}/{S} groups, "
                          f"best_reward={avg_r:.4f}")

            prev_rewards = rewards

        # Rebuild noisy_V_R after potential resampling
        noisy_V_R = torch.cat(
            [noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds], dim=-1
        )

        # ── Record poses (same as original) ──
        pred_trans = noisy_V_R_trans * scale_exp + centroids_exp
        pred_rot = noisy_V_R_rot
        pred_pose = torch.cat([pred_trans, pred_rot], dim=-1)

        denoised_step = ddim_t[i + 1].item() if i < len(ddim_t) - 1 else 0
        predict_link_pose = vector_to_matrix(pred_pose[:, :valid_links])

        predict_link_pose_dict_full = {}
        for link_id, link_name in enumerate(link_names):
            predict_link_pose_dict_full[link_name] = predict_link_pose[:, link_id]
        all_diffuse_step_poses_dict[denoised_step] = predict_link_pose_dict_full

    # ── Select best particle per group ──
    final_rewards = prev_rewards.view(S, k)  # [S, k]
    best_indices = final_rewards.argmax(dim=1)  # [S]
    select_indices = torch.arange(S, device=device) * k + best_indices  # [S]

    avg_reward = final_rewards.max(dim=1).values.mean().item()
    print()
    print(f"  {YELLOW}{BOLD}[FK Steering Summary]{RESET}")
    print(f"  {YELLOW}{LINE_H * 40}{RESET}")
    print(f"  {YELLOW}Particles per sample: {k}{RESET}")
    print(f"  {YELLOW}Total group resamples: {total_resample_count} across {len(ddim_t)} steps{RESET}")
    print(f"  {YELLOW}Avg best-particle reward: {avg_reward:.6f}{RESET}")
    print(f"  {YELLOW}{LINE_H * 40}{RESET}")
    print()

    # Select best poses from the full dict
    best_poses_dict = {}
    for step, pose_dict in all_diffuse_step_poses_dict.items():
        best_pose_dict = {}
        for link_name, poses in pose_dict.items():
            best_pose_dict[link_name] = poses[select_indices]  # [S, 4, 4]
        best_poses_dict[step] = best_pose_dict

    return best_poses_dict


# ── Core evaluation logic ────────────────────────────────────────────────

def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
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
    batch_size, split_batch_size, gpu, device, fk_cfg=None,
):
    """Evaluate a single robot hand. Returns dict with metrics and vis data."""
    fk_enabled = (
        fk_cfg is not None
        and fk_cfg.get('enabled', False)
        and hand.keypoints is not None
    )
    if fk_enabled:
        print(f"  {YELLOW}{BOLD}[FK]{RESET} {YELLOW}Steering active for this hand{RESET}")

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
            if "object_pc_normal" in batch:
                split_batch["object_pc_normal"] = batch["object_pc_normal"][
                    data_count : data_count + split_num
                ].to(device)
            data_count += split_num

            time_start = time.time()
            with torch.no_grad():
                if fk_enabled:
                    all_step_poses_dict = inference_with_fk_steering(
                        model, split_batch, hand, fk_cfg, device,
                        batch_retarget, target_links,
                    )
                else:
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
                for kk, v in transform.items():
                    transform_batch[kk] = (
                        v
                        if kk not in transform_batch
                        else torch.cat((transform_batch[kk], v), dim=0)
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

    gpu = gpu_override if gpu_override is not None else config.test.gpu
    hands = hand_overrides or [config.test.embodiment]

    if ckpt_override is None:
        ckpt_list = [config.test.ckpt]
    elif isinstance(ckpt_override, str):
        ckpt_list = [ckpt_override]
    else:
        ckpt_list = ckpt_override

    base_save_dir = config.test.save_dir

    # FK Steering config (with defaults for backward compatibility)
    fk_cfg = config.get('fk_steering', None)
    if fk_cfg is None:
        fk_cfg = OmegaConf.create({'enabled': False})

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

        print(f"{DIM}Building model...{RESET}")
        model = RobotGraphV4CE(**config.model).to(device)
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

            fk_active = (
                fk_cfg.get('enabled', False)
                and hand_name in ('allegro', 'barrett', 'shadowhand')
            )
            print_banner(hand_name, ckpt, gpu, num_objects, batch_size,
                         fk_enabled=fk_active,
                         fk_cfg=fk_cfg if fk_active else None)

            hand = create_hand_model(hand_name, device)
            urdf_path = robot_urdf_meta["urdf_path"][hand_name]
            target_links = list(hand.links_pc.keys())

            # Apply LeapHand tip mapping, weights, and locked joints
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
                fk_cfg=fk_cfg if fk_active else None,
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
        description="Cross-Embodiment Diffusion V4 (VAE) evaluation with FK Steering"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/test_diff_v4_ce_vae_fk.yaml",
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
    args, unknown = parser.parse_known_args()
    config = OmegaConf.load(args.config)

    # Collect OmegaConf dotlist overrides from both --hands list and trailing args
    dotlist = [a for a in unknown if "=" in a]
    hands = args.hands
    if hands:
        dotlist += [h for h in hands if "=" in h]
        hands = [h for h in hands if "=" not in h]
        if not hands:
            hands = None
    if dotlist:
        overrides = OmegaConf.from_dotlist(dotlist)
        config = OmegaConf.merge(config, overrides)

    test(config, hand_overrides=hands, ckpt_override=args.ckpt, gpu_override=args.gpu)
