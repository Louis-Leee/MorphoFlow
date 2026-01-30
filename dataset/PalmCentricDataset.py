"""
Palm-centric datasets for two-stage flow matching training.

Stage 1: Hand configuration only (no objects). Palm-centric link poses.
Stage 2: Object-grasp pairs decomposed into palm-centric fingers + palm-to-object pose.
"""

import os
import sys
import json
import math
import random
import trimesh
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model
from utils.palm_centric import (
    get_palm_link_name,
    get_palm_index,
    compute_palm_centric_se3,
    compute_canonical_scale,
    compute_all_canonical_scales,
)


def _matrix_to_pose_(rotation_matrix):
    """Convert list of [4, 4] SE3 tensors to [L, 6] pose vectors."""
    num_link = len(rotation_matrix) if isinstance(rotation_matrix, list) else rotation_matrix.shape[0]
    poses = torch.zeros((num_link, 6), dtype=torch.float32)
    if isinstance(rotation_matrix, list):
        for link_id, T in enumerate(rotation_matrix):
            if torch.is_tensor(T):
                T = T.cpu().numpy()
            rot = T[:3, :3]
            trans = T[:3, 3]
            axis_angle = R.from_matrix(rot).as_rotvec()
            pose = np.concatenate((trans, axis_angle), axis=0)
            poses[link_id] = torch.from_numpy(pose)
    else:
        for link_id in range(num_link):
            T = rotation_matrix[link_id]
            if torch.is_tensor(T):
                T = T.cpu().numpy()
            rot = T[:3, :3]
            trans = T[:3, 3]
            axis_angle = R.from_matrix(rot).as_rotvec()
            pose = np.concatenate((trans, axis_angle), axis=0)
            poses[link_id] = torch.from_numpy(pose)
    return poses


class PalmCentricStage1Dataset(Dataset):
    """
    Stage 1 dataset: hand configurations only, no objects.

    Returns palm-centric link poses normalized by per-robot canonical scale.
    Supports augmentation with random joint angle sampling.
    """

    def __init__(
        self,
        batch_size: int,
        robot_names: list = None,
        augment_ratio: float = 0.2,
        canonical_scales: dict = None,
    ):
        self.batch_size = batch_size
        self.robot_names = robot_names or ["allegro", "barrett", "shadowhand"]
        self.augment_ratio = augment_ratio

        # Create hand models
        self.hands = {}
        for robot_name in self.robot_names:
            self.hands[robot_name] = create_hand_model(robot_name, torch.device("cpu"))

        # Compute palm indices
        self.palm_indices = {}
        for robot_name in self.robot_names:
            link_names = list(self.hands[robot_name].links_pc.keys())
            self.palm_indices[robot_name] = get_palm_index(link_names, robot_name)

        # Compute or load canonical scales
        if canonical_scales is not None:
            self.canonical_scales = canonical_scales
        else:
            print("Computing canonical scales...")
            self.canonical_scales = compute_all_canonical_scales(
                self.hands, self.robot_names
            )

        # Load grasp dataset (joint angles only)
        dataset_path = os.path.join(
            ROOT_DIR, "data/CMapDataset_filtered/cmap_full_dataset.pt"
        )
        metadata = torch.load(dataset_path)["metadata"]
        self.metadata = [
            m for m in metadata if m[2] in self.robot_names
        ]
        # Group by robot for efficient random sampling
        self.metadata_by_robot = {}
        for robot_name in self.robot_names:
            self.metadata_by_robot[robot_name] = [
                (m[0], m[1]) for m in self.metadata if m[2] == robot_name
            ]

    def _get_palm_centric_pose(self, hand, robot_name, q):
        """Compute palm-centric pose vector from joint angles."""
        _, all_link_se3 = hand.get_transformed_links_pc(q)
        palm_index = self.palm_indices[robot_name]
        _, palm_centric_se3 = compute_palm_centric_se3(all_link_se3, palm_index)
        pose_vec = _matrix_to_pose_(palm_centric_se3)

        # Normalize translation by canonical scale
        canonical_scale = self.canonical_scales[robot_name]
        pose_vec[:, :3] = pose_vec[:, :3] / canonical_scale

        return pose_vec, palm_centric_se3

    def _sample_random_q(self, hand):
        """Sample random joint angles within joint limits."""
        lower, upper = hand.pk_chain.get_joint_limits()
        lower = torch.tensor(lower, dtype=torch.float32)
        upper = torch.tensor(upper, dtype=torch.float32)

        q = torch.zeros(hand.dof, dtype=torch.float32)
        # Global pose: random rotation, zero translation
        q[:3] = 0.0  # translation at origin
        q[3:6] = (torch.rand(3) * 2 - 1) * torch.pi  # random rotation
        q[5] /= 2  # reduce range for roll

        # Joint angles: random within limits
        portion = random.uniform(0.1, 0.9)
        q[6:] = lower[6:] * (1 - portion) + upper[6:] * portion

        return q

    def __getitem__(self, index):
        robot_name_batch = []
        palm_centric_vec_batch = []
        palm_centric_se3_batch = []
        robot_links_pc_batch = []

        for idx in range(self.batch_size):
            robot_name = random.choice(self.robot_names)
            robot_name_batch.append(robot_name)
            hand = self.hands[robot_name]
            robot_links_pc_batch.append(hand.links_pc)

            # Decide: use existing grasp data or random augmentation
            use_random = random.random() < self.augment_ratio

            if use_random:
                q = self._sample_random_q(hand)
            else:
                target_q, _ = random.choice(self.metadata_by_robot[robot_name])
                q = target_q

            pose_vec, pc_se3 = self._get_palm_centric_pose(hand, robot_name, q)
            palm_centric_vec_batch.append(pose_vec)
            palm_centric_se3_batch.append(pc_se3)

        return {
            "robot_name": robot_name_batch,
            "palm_centric_vec": palm_centric_vec_batch,  # list[B] of [L, 6]
            "palm_centric_se3": palm_centric_se3_batch,  # list[B] of [L, 4, 4]
            "robot_links_pc": robot_links_pc_batch,
        }

    def __len__(self):
        return math.ceil(len(self.metadata) / self.batch_size)


class PalmCentricStage2Dataset(Dataset):
    """
    Stage 2 dataset: object-grasp pairs with palm-centric decomposition.

    Returns:
    - palm_centric_vec: finger poses relative to palm (normalized by canonical_scale)
    - palm_pose_in_object: palm SE3 in object-normalized frame
    - object_pc: object point cloud
    """

    def __init__(
        self,
        batch_size: int,
        robot_names: list = None,
        is_train: bool = True,
        debug_object_names: list = None,
        num_points: int = 512,
        object_pc_type: str = "random",
        canonical_scales: dict = None,
    ):
        self.batch_size = batch_size
        self.robot_names = robot_names or ["allegro", "barrett", "shadowhand"]
        self.is_train = is_train
        self.num_points = num_points
        self.object_pc_type = object_pc_type

        # Create hand models
        self.hands = {}
        for robot_name in self.robot_names:
            self.hands[robot_name] = create_hand_model(
                robot_name, torch.device("cpu")
            )

        # Palm indices
        self.palm_indices = {}
        for robot_name in self.robot_names:
            link_names = list(self.hands[robot_name].links_pc.keys())
            self.palm_indices[robot_name] = get_palm_index(link_names, robot_name)

        # Canonical scales
        if canonical_scales is not None:
            self.canonical_scales = canonical_scales
        else:
            print("Computing canonical scales...")
            self.canonical_scales = compute_all_canonical_scales(
                self.hands, self.robot_names
            )

        # Load object split
        split_json_path = os.path.join(
            ROOT_DIR, "data/CMapDataset_filtered/split_train_validate_objects.json"
        )
        dataset_split = json.load(open(split_json_path))
        self.object_names = (
            dataset_split["train"] if is_train else dataset_split["validate"]
        )
        if debug_object_names is not None:
            self.object_names = debug_object_names

        # Load grasp metadata
        dataset_path = os.path.join(
            ROOT_DIR, "data/CMapDataset_filtered/cmap_full_dataset.pt"
        )
        metadata = torch.load(dataset_path)["metadata"]
        self.metadata = [
            m
            for m in metadata
            if m[1] in self.object_names and m[2] in self.robot_names
        ]

        if not self.is_train:
            self.combination = []
            for robot_name in self.robot_names:
                for object_name in self.object_names:
                    self.combination.append((robot_name, object_name))
            self.combination = sorted(self.combination)

        # Load object point clouds
        self.object_pcs = {}
        self.object_normals = {}
        if self.object_pc_type != "fixed":
            for object_name in self.object_names:
                name = object_name.split("+")
                mesh_path = os.path.join(
                    ROOT_DIR,
                    f"data/data_urdf/object/{name[0]}/{name[1]}/{name[1]}.stl",
                )
                mesh = trimesh.load_mesh(mesh_path)
                object_pc, indices = mesh.sample(65536, return_index=True)
                pc_normals = mesh.face_normals[indices]
                self.object_pcs[object_name] = torch.tensor(
                    object_pc, dtype=torch.float32
                )
                self.object_normals[object_name] = torch.tensor(
                    pc_normals, dtype=torch.float32
                )

    def _compute_object_normalization(self, object_pc):
        """Compute object centroid and scale for normalization."""
        centroid = object_pc.mean(dim=0, keepdim=True)  # [1, 3]
        pc_centered = object_pc - centroid
        scale = pc_centered.abs().max()  # scalar
        return centroid.squeeze(0), scale  # [3], scalar

    def _decompose_grasp(self, hand, robot_name, q, object_pc):
        """
        Decompose a grasp into palm-centric fingers + palm-in-object-frame.

        Returns:
            palm_centric_vec: [L, 6] palm-centric finger poses (normalized by canonical_scale)
            palm_pose_in_object: [6] palm pose in object-normalized frame
            palm_centric_se3: [L, 4, 4] palm-centric SE3 transforms
        """
        _, all_link_se3 = hand.get_transformed_links_pc(q)
        palm_index = self.palm_indices[robot_name]

        # Palm-centric decomposition
        palm_se3, palm_centric_se3 = compute_palm_centric_se3(
            all_link_se3, palm_index
        )

        # Palm-centric pose vector (normalized by canonical scale)
        palm_centric_vec = _matrix_to_pose_(palm_centric_se3)
        canonical_scale = self.canonical_scales[robot_name]
        palm_centric_vec[:, :3] = palm_centric_vec[:, :3] / canonical_scale

        # Palm pose in object-normalized frame
        obj_centroid, obj_scale = self._compute_object_normalization(object_pc)
        palm_trans_world = palm_se3[:3, 3]
        palm_rot_world = palm_se3[:3, :3]

        palm_trans_obj = (palm_trans_world - obj_centroid) / obj_scale
        palm_rot_aa = torch.from_numpy(
            R.from_matrix(palm_rot_world.cpu().numpy()).as_rotvec()
        ).float()
        palm_pose_in_object = torch.cat([palm_trans_obj, palm_rot_aa], dim=0)  # [6]

        return palm_centric_vec, palm_pose_in_object, palm_centric_se3

    def __getitem__(self, index):
        if self.is_train:
            return self._get_train_batch()
        else:
            return self._get_val_batch(index)

    def _get_train_batch(self):
        robot_name_batch = []
        object_name_batch = []
        palm_centric_vec_batch = []
        palm_centric_se3_batch = []
        palm_pose_in_object_batch = []
        object_pc_batch = []
        object_normal_batch = []
        robot_links_pc_batch = []
        target_q_batch = []
        initial_q_batch = []
        initial_se3_batch = []
        initial_palm_centric_vec_batch = []
        initial_palm_pose_in_object_batch = []

        for idx in range(self.batch_size):
            robot_name = random.choice(self.robot_names)
            robot_name_batch.append(robot_name)
            hand = self.hands[robot_name]
            robot_links_pc_batch.append(hand.links_pc)

            # Sample a grasp
            metadata_robot = [
                (m[0], m[1]) for m in self.metadata if m[2] == robot_name
            ]
            target_q, object_name = random.choice(metadata_robot)
            target_q_batch.append(target_q)
            object_name_batch.append(object_name)

            # Sample object point cloud
            if self.object_pc_type == "random":
                indices = torch.randperm(65536)[: self.num_points]
                object_pc = self.object_pcs[object_name][indices]
                object_normal = self.object_normals[object_name][indices]
                object_pc += torch.randn(object_pc.shape) * 0.002
            elif self.object_pc_type == "partial":
                indices = torch.randperm(65536)[: self.num_points * 2]
                object_pc = self.object_pcs[object_name][indices]
                direction = torch.randn(3)
                direction = direction / torch.norm(direction)
                proj = object_pc @ direction
                _, sort_indices = torch.sort(proj)
                sort_indices = sort_indices[self.num_points :]
                object_pc = object_pc[sort_indices]
                object_normal = self.object_normals[object_name][sort_indices]
            else:
                name = object_name.split("+")
                object_path = os.path.join(
                    ROOT_DIR, f"data/PointCloud/object/{name[0]}/{name[1]}.pt"
                )
                object_pc = torch.load(object_path)[:, :3]
                object_normal = torch.zeros_like(object_pc)

            object_pc_batch.append(object_pc)
            object_normal_batch.append(object_normal)

            # Decompose target grasp
            palm_centric_vec, palm_pose_obj, palm_centric_se3 = (
                self._decompose_grasp(hand, robot_name, target_q, object_pc)
            )
            palm_centric_vec_batch.append(palm_centric_vec)
            palm_pose_in_object_batch.append(palm_pose_obj)
            palm_centric_se3_batch.append(palm_centric_se3)

            # Initial q (for palm-conditioned inference)
            initial_q = hand.get_initial_q(target_q)
            initial_q_batch.append(initial_q)
            _, initial_se3 = hand.get_transformed_links_pc(initial_q)
            initial_se3_batch.append(initial_se3)

            init_pc_vec, init_palm_obj, _ = self._decompose_grasp(
                hand, robot_name, initial_q, object_pc
            )
            initial_palm_centric_vec_batch.append(init_pc_vec)
            initial_palm_pose_in_object_batch.append(init_palm_obj)

        return {
            "robot_name": robot_name_batch,
            "object_name": object_name_batch,
            "palm_centric_vec": palm_centric_vec_batch,  # list[B] of [L, 6]
            "palm_centric_se3": palm_centric_se3_batch,  # list[B] of [L, 4, 4]
            "palm_pose_in_object": palm_pose_in_object_batch,  # list[B] of [6]
            "object_pc": torch.stack(object_pc_batch),  # [B, N, 3]
            "object_pc_normal": torch.stack(object_normal_batch),
            "robot_links_pc": robot_links_pc_batch,
            "target_q": target_q_batch,
            "initial_q": initial_q_batch,
            "initial_se3": initial_se3_batch,
            "initial_palm_centric_vec": initial_palm_centric_vec_batch,
            "initial_palm_pose_in_object": initial_palm_pose_in_object_batch,
        }

    def _get_val_batch(self, index):
        robot_name, object_name = self.combination[index]
        hand = self.hands[robot_name]

        initial_q_batch = torch.zeros(
            [self.batch_size, hand.dof], dtype=torch.float32
        )
        object_pc_batch = torch.zeros(
            [self.batch_size, self.num_points, 3], dtype=torch.float32
        )
        robot_links_pc_batch = []
        initial_se3_batch = []
        initial_palm_centric_vec_batch = []
        initial_palm_pose_in_object_batch = []

        for batch_idx in range(self.batch_size):
            initial_q = hand.get_initial_q()
            _, all_link_se3_initial = hand.get_transformed_links_pc(initial_q)
            initial_se3_batch.append(all_link_se3_initial)
            robot_links_pc_batch.append(hand.links_pc)

            if self.object_pc_type == "partial":
                indices = torch.randperm(65536)[: self.num_points * 2]
                object_pc = self.object_pcs[object_name][indices]
                direction = torch.randn(3)
                direction = direction / torch.norm(direction)
                proj = object_pc @ direction
                _, sort_indices = torch.sort(proj)
                sort_indices = sort_indices[self.num_points :]
                object_pc = object_pc[sort_indices]
            else:
                name = object_name.split("+")
                object_path = os.path.join(
                    ROOT_DIR, f"data/PointCloud/object/{name[0]}/{name[1]}.pt"
                )
                object_pc = torch.load(object_path)[:, :3]

            initial_q_batch[batch_idx] = initial_q
            object_pc_batch[batch_idx] = object_pc

            # Decompose initial pose
            init_pc_vec, init_palm_obj, _ = self._decompose_grasp(
                hand, robot_name, initial_q, object_pc
            )
            initial_palm_centric_vec_batch.append(init_pc_vec)
            initial_palm_pose_in_object_batch.append(init_palm_obj)

        return {
            "robot_name": robot_name,
            "object_name": object_name,
            "initial_q": initial_q_batch,
            "initial_se3": torch.stack(initial_se3_batch, dim=0),
            "object_pc": object_pc_batch,
            "robot_links_pc": robot_links_pc_batch,
            "initial_palm_centric_vec": initial_palm_centric_vec_batch,
            "initial_palm_pose_in_object": initial_palm_pose_in_object_batch,
        }

    def __len__(self):
        if self.is_train:
            return math.ceil(len(self.metadata) / self.batch_size)
        else:
            return len(self.combination)


def palm_centric_collate_fn(batch):
    """Custom collate for single-sample batches (dataset handles batching internally)."""
    return batch[0]
