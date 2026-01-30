"""
Palm-centric coordinate utilities for two-stage flow matching training.

Provides functions to:
- Identify the palm link for each robot
- Decompose world-frame link poses into palm-centric representations
- Compute per-robot canonical scales for normalization
"""

import torch
from typing import Dict, Tuple


# Palm link names for each supported robot
PALM_LINK_NAMES: Dict[str, str] = {
    "allegro": "base_link",
    "barrett": "bh_base_link",
    "shadowhand": "palm",
    "leaphand": "palm_lower",
    "ezgripper": "left_ezgripper_palm_link",
    "robotiq_3finger": "gripper_palm",
}


def get_palm_link_name(robot_name: str) -> str:
    """Return the palm link name for a given robot."""
    if robot_name not in PALM_LINK_NAMES:
        raise ValueError(
            f"Unknown robot '{robot_name}'. Supported: {list(PALM_LINK_NAMES.keys())}"
        )
    return PALM_LINK_NAMES[robot_name]


def get_palm_index(link_names: list, robot_name: str) -> int:
    """Find the index of the palm link in the link_names list."""
    palm_name = get_palm_link_name(robot_name)
    if palm_name not in link_names:
        raise ValueError(
            f"Palm link '{palm_name}' not found in link_names: {link_names}"
        )
    return link_names.index(palm_name)


def compute_palm_centric_se3(
    all_link_se3: torch.Tensor,
    palm_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose world-frame link SE3 transforms into palm-centric representation.

    Args:
        all_link_se3: [L, 4, 4] SE3 transforms for all links in world frame
        palm_index: index of the palm link

    Returns:
        palm_se3: [4, 4] palm transform in world frame
        palm_centric_se3: [L, 4, 4] link transforms relative to palm
            palm_centric_se3[palm_index] = identity
    """
    palm_se3 = all_link_se3[palm_index]  # [4, 4]
    palm_inv = torch.inverse(palm_se3)  # [4, 4]
    palm_centric_se3 = palm_inv.unsqueeze(0) @ all_link_se3  # [L, 4, 4]
    return palm_se3, palm_centric_se3


def compute_palm_centric_se3_batch(
    all_link_se3: torch.Tensor,
    palm_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched version of compute_palm_centric_se3.

    Args:
        all_link_se3: [B, L, 4, 4] SE3 transforms for all links in world frame
        palm_index: index of the palm link

    Returns:
        palm_se3: [B, 4, 4] palm transforms in world frame
        palm_centric_se3: [B, L, 4, 4] link transforms relative to palm
    """
    palm_se3 = all_link_se3[:, palm_index]  # [B, 4, 4]
    palm_inv = torch.inverse(palm_se3)  # [B, 4, 4]
    palm_centric_se3 = palm_inv.unsqueeze(1) @ all_link_se3  # [B, L, 4, 4]
    return palm_se3, palm_centric_se3


def se3_to_pose_vec(se3: torch.Tensor) -> torch.Tensor:
    """
    Convert SE3 matrix to 6D pose vector [trans_x, trans_y, trans_z, ax, ay, az].

    Args:
        se3: [..., 4, 4] SE3 matrices

    Returns:
        pose_vec: [..., 6] pose vectors (translation + axis-angle)
    """
    from scipy.spatial.transform import Rotation as R
    import numpy as np

    original_shape = se3.shape[:-2]
    se3_flat = se3.reshape(-1, 4, 4).cpu().numpy()

    poses = []
    for T in se3_flat:
        trans = T[:3, 3]
        rot = R.from_matrix(T[:3, :3]).as_rotvec()
        poses.append(np.concatenate([trans, rot]))

    result = torch.tensor(np.array(poses), dtype=se3.dtype, device=se3.device)
    return result.reshape(*original_shape, 6)


def se3_to_pose_vec_torch(se3: torch.Tensor) -> torch.Tensor:
    """
    Convert SE3 matrix to 6D pose vector using PyTorch (no scipy, GPU-friendly).

    Args:
        se3: [..., 4, 4] SE3 matrices

    Returns:
        pose_vec: [..., 6] pose vectors (translation + axis-angle)
    """
    from pytorch3d.transforms import matrix_to_axis_angle

    original_shape = se3.shape[:-2]
    se3_flat = se3.reshape(-1, 4, 4)

    trans = se3_flat[:, :3, 3]  # [N, 3]
    rot_mat = se3_flat[:, :3, :3]  # [N, 3, 3]
    axis_angle = matrix_to_axis_angle(rot_mat)  # [N, 3]

    pose_vec = torch.cat([trans, axis_angle], dim=-1)  # [N, 6]
    return pose_vec.reshape(*original_shape, 6)


def compute_canonical_scale(hand_model, robot_name: str) -> float:
    """
    Compute the canonical scale for a robot hand.

    The canonical scale is the maximum distance from the palm link to any
    other link at the rest (zero joint angles) configuration. This provides
    an intrinsic scale for the hand that doesn't depend on objects.

    Args:
        hand_model: HandModel instance
        robot_name: robot name string

    Returns:
        canonical_scale: float, max distance from palm to any link
    """
    palm_name = get_palm_link_name(robot_name)
    link_names = list(hand_model.links_pc.keys())
    palm_index = link_names.index(palm_name)

    # FK at zero joint angles (rest pose)
    q_zero = torch.zeros(hand_model.dof, dtype=torch.float32, device=hand_model.device)
    _, all_link_se3 = hand_model.get_transformed_links_pc(q_zero)

    # Decompose into palm-centric
    palm_se3, palm_centric_se3 = compute_palm_centric_se3(all_link_se3, palm_index)

    # Max distance from palm to any link center
    link_positions = palm_centric_se3[:, :3, 3]  # [L, 3]
    distances = torch.norm(link_positions, dim=-1)  # [L]

    # Use max distance, with a minimum floor to avoid division by zero
    canonical_scale = max(distances.max().item(), 0.01)
    return canonical_scale


def compute_all_canonical_scales(
    hand_dict: dict, robot_names: list
) -> Dict[str, float]:
    """
    Compute canonical scales for all robots.

    Args:
        hand_dict: dict mapping robot_name -> HandModel
        robot_names: list of robot names

    Returns:
        scales: dict mapping robot_name -> canonical_scale
    """
    scales = {}
    for robot_name in robot_names:
        hand_model = hand_dict[robot_name]
        scales[robot_name] = compute_canonical_scale(hand_model, robot_name)
        print(f"  {robot_name}: canonical_scale = {scales[robot_name]:.4f}")
    return scales


def normalize_palm_centric_pose(
    pose_vec: torch.Tensor, canonical_scale: float
) -> torch.Tensor:
    """
    Normalize a palm-centric pose vector by dividing translation by canonical_scale.
    Rotation (axis-angle) is not normalized.

    Args:
        pose_vec: [..., 6] pose vector (trans + axis-angle)
        canonical_scale: float, per-robot canonical scale

    Returns:
        normalized: [..., 6] normalized pose vector
    """
    normalized = pose_vec.clone()
    normalized[..., :3] = normalized[..., :3] / canonical_scale
    return normalized


def denormalize_palm_centric_pose(
    pose_vec: torch.Tensor, canonical_scale: float
) -> torch.Tensor:
    """
    Denormalize a palm-centric pose vector by multiplying translation by canonical_scale.

    Args:
        pose_vec: [..., 6] normalized pose vector (trans + axis-angle)
        canonical_scale: float, per-robot canonical scale

    Returns:
        denormalized: [..., 6] denormalized pose vector
    """
    denormalized = pose_vec.clone()
    denormalized[..., :3] = denormalized[..., :3] * canonical_scale
    return denormalized
