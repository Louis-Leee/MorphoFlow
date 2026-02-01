"""
CrossEmbodimentDataset: Mixed GT + no-GT dataset for cross-embodiment training.

GT robots (e.g., allegro, barrett, shadowhand): sample (object, grasp) pairs
from CMapDataset_filtered, same as CMapDataset.

No-GT robots (e.g., leaphand): generate random valid FK poses within URDF
joint limits. No object conditioning — used for learning hand configuration
priors via self-reconstruction.

Each __getitem__ returns a full batch (batch_size samples) for one robot.
Robot selection is uniform across all robots (GT + no-GT).
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


class CrossEmbodimentDataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        gt_robot_names: list,
        no_gt_robot_names: list = None,
        is_train: bool = True,
        debug_object_names: list = None,
        num_points: int = 512,
        object_pc_type: str = 'random',
        no_gt_joint_range: list = None,
        no_gt_q_noise_std: float = 0.0,
    ):
        self.batch_size = batch_size
        self.gt_robot_names = list(gt_robot_names)
        self.no_gt_robot_names = list(no_gt_robot_names) if no_gt_robot_names else []
        self.all_robot_names = self.gt_robot_names + self.no_gt_robot_names
        self.gt_robot_set = set(self.gt_robot_names)
        self.is_train = is_train
        self.num_points = num_points
        self.object_pc_type = object_pc_type
        self.no_gt_joint_range = tuple(no_gt_joint_range) if no_gt_joint_range else (0.1, 0.9)
        self.no_gt_q_noise_std = no_gt_q_noise_std

        # Load hand models for ALL robots
        self.hands = {}
        self.dofs = []
        for robot_name in self.all_robot_names:
            self.hands[robot_name] = create_hand_model(robot_name, torch.device('cpu'))
            self.dofs.append(math.sqrt(self.hands[robot_name].dof))

        # Object splits
        split_json_path = os.path.join(ROOT_DIR, 'data/CMapDataset_filtered/split_train_validate_objects.json')
        dataset_split = json.load(open(split_json_path))
        self.object_names = dataset_split['train'] if is_train else dataset_split['validate']
        if debug_object_names is not None:
            print("!!! Using debug objects !!!")
            self.object_names = debug_object_names

        # GT metadata (only for GT robots)
        dataset_path = os.path.join(ROOT_DIR, 'data/CMapDataset_filtered/cmap_full_dataset.pt')
        metadata = torch.load(dataset_path)['metadata']
        self.metadata = [m for m in metadata if m[1] in self.object_names and m[2] in self.gt_robot_set]

        # Group metadata by robot for efficient sampling
        self.metadata_by_robot = {}
        for m in self.metadata:
            robot = m[2]
            if robot not in self.metadata_by_robot:
                self.metadata_by_robot[robot] = []
            self.metadata_by_robot[robot].append((m[0], m[1]))  # (target_q, object_name)

        # No-GT metadata: load real grasps for no-GT robots from the same dataset
        self.nogt_metadata_by_robot = {}
        no_gt_robot_set = set(self.no_gt_robot_names)
        for m in metadata:
            if m[2] in no_gt_robot_set and m[1] in self.object_names:
                robot = m[2]
                if robot not in self.nogt_metadata_by_robot:
                    self.nogt_metadata_by_robot[robot] = []
                self.nogt_metadata_by_robot[robot].append((m[0], m[1]))
        for robot, entries in self.nogt_metadata_by_robot.items():
            print(f"No-GT robot '{robot}': {len(entries)} real grasps loaded")

        # Object point clouds (for GT batches)
        self.object_pcs = {}
        self.object_normals = {}
        if self.object_pc_type != 'fixed':
            for object_name in self.object_names:
                name = object_name.split('+')
                mesh_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{name[1]}.stl')
                mesh = trimesh.load_mesh(mesh_path)
                object_pc, indices = mesh.sample(65536, return_index=True)
                pc_normals = mesh.face_normals[indices]
                self.object_pcs[object_name] = torch.tensor(object_pc, dtype=torch.float32)
                self.object_normals[object_name] = torch.tensor(pc_normals, dtype=torch.float32)

        if not is_train:
            # Validation: create combinations for all robots (GT + no-GT) × objects
            self.combination = []
            for robot_name in self.all_robot_names:
                for object_name in self.object_names:
                    self.combination.append((robot_name, object_name))
            self.combination = sorted(self.combination)

    def _matrix_to_pose_(self, rotation_matrix):
        """Convert list of [4, 4] SE3 matrices to [L, 6] pose vectors."""
        num_link = len(rotation_matrix)
        poses = torch.zeros((num_link, 6), dtype=torch.float32)
        for link_id, T in enumerate(rotation_matrix):
            rot = T[:3, :3].numpy()
            trans = T[:3, 3].numpy()
            axis_angle = R.from_matrix(rot).as_rotvec()
            pose = np.concatenate((trans, axis_angle), axis=0)
            poses[link_id] = torch.from_numpy(pose)
        return poses

    def _sample_random_q(self, hand):
        """Sample random valid joint config within URDF limits."""
        lower, upper = hand.pk_chain.get_joint_limits()
        q = torch.zeros(hand.dof, dtype=torch.float32)

        # Root pose: random rotation, zero translation
        q[:3] = 0.0
        q[3:6] = (torch.rand(3) * 2 - 1) * torch.pi
        q[5] /= 2  # reduce roll range

        # Joint angles: random within limits
        lo, hi = self.no_gt_joint_range
        portion = random.uniform(lo, hi)
        lower_t = torch.tensor(lower[6:], dtype=torch.float32)
        upper_t = torch.tensor(upper[6:], dtype=torch.float32)
        q[6:] = lower_t * (1 - portion) + upper_t * portion

        return q

    def _sample_object_pc(self, object_name):
        """Sample object point cloud (same logic as CMapDataset)."""
        if self.object_pc_type == 'fixed':
            name = object_name.split('+')
            object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
            object_pc = torch.load(object_path)[:, :3]
            object_normal = torch.zeros_like(object_pc)
        elif self.object_pc_type == 'random':
            indices = torch.randperm(65536)[:self.num_points]
            object_pc = self.object_pcs[object_name][indices]
            object_normal = self.object_normals[object_name][indices]
            object_pc = object_pc + torch.randn(object_pc.shape) * 0.002
        else:  # 'partial'
            indices = torch.randperm(65536)[:self.num_points * 2]
            object_pc = self.object_pcs[object_name][indices]
            direction = torch.randn(3)
            direction = direction / torch.norm(direction)
            proj = object_pc @ direction
            _, sort_idx = torch.sort(proj)
            sort_idx = sort_idx[self.num_points:]
            object_pc = object_pc[sort_idx]
            object_normal = self.object_normals[object_name][indices][sort_idx]
        return object_pc, object_normal

    def _get_gt_batch(self, robot_name):
        """Sample a batch of GT (object, grasp) pairs for a GT robot."""
        hand = self.hands[robot_name]
        metadata_robot = self.metadata_by_robot[robot_name]

        robot_name_batch = []
        object_name_batch = []
        robot_pc_initial_batch = []
        robot_pc_target_batch = []
        robot_links_pc_batch = []
        object_pc_batch = []
        object_normal_batch = []
        initial_q_batch = []
        target_q_batch = []
        all_link_se3_target_batch = []
        all_link_se3_initial_batch = []
        all_link_vec_target_batch = []
        all_link_vec_initial_batch = []

        for idx in range(self.batch_size):
            robot_name_batch.append(robot_name)

            target_q, object_name = random.choice(metadata_robot)
            target_q_batch.append(target_q)
            object_name_batch.append(object_name)
            robot_links_pc_batch.append(hand.links_pc)

            object_pc, object_normal = self._sample_object_pc(object_name)
            object_pc_batch.append(object_pc)
            object_normal_batch.append(object_normal)

            robot_pc_target, all_link_se3_target = hand.get_transformed_links_pc(target_q)
            robot_pc_target_batch.append(robot_pc_target)
            all_link_se3_target_batch.append(all_link_se3_target)
            all_link_vec_target_batch.append(self._matrix_to_pose_(all_link_se3_target))

            initial_q = hand.get_initial_q(target_q)
            initial_q_batch.append(initial_q)

            robot_pc_initial, all_link_se3_initial = hand.get_transformed_links_pc(initial_q)
            robot_pc_initial_batch.append(robot_pc_initial)
            all_link_se3_initial_batch.append(all_link_se3_initial)
            all_link_vec_initial_batch.append(self._matrix_to_pose_(all_link_se3_initial))

        return {
            'has_gt': True,
            'robot_name': robot_name_batch,
            'object_name': object_name_batch,
            'robot_pc_initial': robot_pc_initial_batch,
            'robot_pc_target': robot_pc_target_batch,
            'robot_links_pc': robot_links_pc_batch,
            'object_pc': torch.stack(object_pc_batch),
            'object_pc_normal': torch.stack(object_normal_batch),
            'initial_se3': all_link_se3_initial_batch,
            'target_se3': all_link_se3_target_batch,
            'initial_vec': all_link_vec_initial_batch,
            'target_vec': all_link_vec_target_batch,
            'initial_q': initial_q_batch,
            'target_q': target_q_batch,
        }

    def _augment_q(self, q, hand, noise_std):
        """Add Gaussian noise to finger joints, clamped to URDF limits."""
        q_aug = q.clone()
        lower, upper = hand.pk_chain.get_joint_limits()
        lower_t = torch.tensor(lower[6:], dtype=torch.float32)
        upper_t = torch.tensor(upper[6:], dtype=torch.float32)
        noise = torch.randn_like(q_aug[6:]) * noise_std
        q_aug[6:] = (q_aug[6:] + noise).clamp(min=lower_t, max=upper_t)
        return q_aug

    def _get_nogt_batch(self, robot_name):
        """Generate a no-GT batch using real grasp data (without object features).

        If real grasp data exists for this robot, sample from it and compute
        object centroid/scale for normalization (V_O remains zeros).
        Falls back to random FK if no grasp data is available.
        """
        nogt_metadata = self.nogt_metadata_by_robot.get(robot_name)
        if not nogt_metadata:
            return self._get_nogt_batch_random(robot_name)

        hand = self.hands[robot_name]

        robot_name_batch = []
        robot_links_pc_batch = []
        initial_q_batch = []
        target_q_batch = []
        all_link_se3_target_batch = []
        all_link_se3_initial_batch = []
        all_link_vec_target_batch = []
        all_link_vec_initial_batch = []
        object_centroid_batch = []
        object_scale_batch = []

        for idx in range(self.batch_size):
            robot_name_batch.append(robot_name)
            robot_links_pc_batch.append(hand.links_pc)

            # Sample a real grasp
            target_q, object_name = random.choice(nogt_metadata)

            # Optional q-space augmentation
            if self.no_gt_q_noise_std > 0:
                target_q = self._augment_q(target_q, hand, self.no_gt_q_noise_std)

            target_q_batch.append(target_q)

            # Compute object centroid/scale for normalization
            # (same computation as _normalize_pc_ in the model)
            object_pc, _ = self._sample_object_pc(object_name)
            centroid = object_pc.mean(dim=0, keepdim=True)   # [1, 3]
            centered = object_pc - centroid
            scale = centered.abs().max(dim=0, keepdim=True)[0]  # [1, 3]
            scale = scale.max(dim=1, keepdim=True)[0]           # [1, 1]
            object_centroid_batch.append(centroid)
            object_scale_batch.append(scale)

            # FK for target
            _, all_link_se3_target = hand.get_transformed_links_pc(target_q)
            all_link_se3_target_batch.append(all_link_se3_target)
            all_link_vec_target_batch.append(self._matrix_to_pose_(all_link_se3_target))

            # Initial q: perturbation around target (same as GT path)
            initial_q = hand.get_initial_q(target_q)
            initial_q_batch.append(initial_q)

            _, all_link_se3_initial = hand.get_transformed_links_pc(initial_q)
            all_link_se3_initial_batch.append(all_link_se3_initial)
            all_link_vec_initial_batch.append(self._matrix_to_pose_(all_link_se3_initial))

        # Dummy object data (V_O will be zeros in the model)
        dummy_object_pc = torch.zeros(self.batch_size, self.num_points, 3)
        dummy_object_normal = torch.zeros(self.batch_size, self.num_points, 3)

        return {
            'has_gt': False,
            'robot_name': robot_name_batch,
            'object_name': ['none'] * self.batch_size,
            'robot_pc_initial': [None] * self.batch_size,
            'robot_pc_target': [None] * self.batch_size,
            'robot_links_pc': robot_links_pc_batch,
            'object_pc': dummy_object_pc,
            'object_pc_normal': dummy_object_normal,
            'initial_se3': all_link_se3_initial_batch,
            'target_se3': all_link_se3_target_batch,
            'initial_vec': all_link_vec_initial_batch,
            'target_vec': all_link_vec_target_batch,
            'initial_q': initial_q_batch,
            'target_q': target_q_batch,
            'object_centroids': torch.cat(object_centroid_batch, dim=0),   # [B, 3]
            'object_scales': torch.cat(object_scale_batch, dim=0),         # [B, 1]
        }

    def _get_nogt_batch_random(self, robot_name):
        """Fallback: random FK poses for robots with no grasp data."""
        hand = self.hands[robot_name]

        robot_name_batch = []
        robot_links_pc_batch = []
        initial_q_batch = []
        target_q_batch = []
        all_link_se3_target_batch = []
        all_link_se3_initial_batch = []
        all_link_vec_target_batch = []
        all_link_vec_initial_batch = []

        for idx in range(self.batch_size):
            robot_name_batch.append(robot_name)
            robot_links_pc_batch.append(hand.links_pc)

            target_q = self._sample_random_q(hand)
            target_q_batch.append(target_q)

            _, all_link_se3_target = hand.get_transformed_links_pc(target_q)
            all_link_se3_target_batch.append(all_link_se3_target)
            all_link_vec_target_batch.append(self._matrix_to_pose_(all_link_se3_target))

            initial_q = hand.get_initial_q(q=None)
            initial_q_batch.append(initial_q)

            _, all_link_se3_initial = hand.get_transformed_links_pc(initial_q)
            all_link_se3_initial_batch.append(all_link_se3_initial)
            all_link_vec_initial_batch.append(self._matrix_to_pose_(all_link_se3_initial))

        dummy_object_pc = torch.zeros(self.batch_size, self.num_points, 3)
        dummy_object_normal = torch.zeros(self.batch_size, self.num_points, 3)

        return {
            'has_gt': False,
            'robot_name': robot_name_batch,
            'object_name': ['none'] * self.batch_size,
            'robot_pc_initial': [None] * self.batch_size,
            'robot_pc_target': [None] * self.batch_size,
            'robot_links_pc': robot_links_pc_batch,
            'object_pc': dummy_object_pc,
            'object_pc_normal': dummy_object_normal,
            'initial_se3': all_link_se3_initial_batch,
            'target_se3': all_link_se3_target_batch,
            'initial_vec': all_link_vec_initial_batch,
            'target_vec': all_link_vec_target_batch,
            'initial_q': initial_q_batch,
            'target_q': target_q_batch,
        }

    def _get_validate_batch(self, robot_name, object_name):
        """Validation batch: initial configs + object PC for any robot."""
        hand = self.hands[robot_name]
        initial_q_batch = torch.zeros([self.batch_size, hand.dof], dtype=torch.float32)
        object_pc_batch = torch.zeros([self.batch_size, self.num_points, 3], dtype=torch.float32)
        robot_links_pc_batch = []
        initial_se3_batch = []

        for batch_idx in range(self.batch_size):
            initial_q = hand.get_initial_q()
            _, all_link_se3_initial = hand.get_transformed_links_pc(initial_q)
            initial_se3_batch.append(all_link_se3_initial)
            robot_links_pc_batch.append(hand.links_pc)

            if self.object_pc_type == 'partial':
                indices = torch.randperm(65536)[:self.num_points * 2]
                object_pc = self.object_pcs[object_name][indices]
                direction = torch.randn(3)
                direction = direction / torch.norm(direction)
                proj = object_pc @ direction
                _, indices = torch.sort(proj)
                indices = indices[self.num_points:]
                object_pc = object_pc[indices]
            else:
                name = object_name.split('+')
                object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
                object_pc = torch.load(object_path)[:, :3]

            initial_q_batch[batch_idx] = initial_q
            object_pc_batch[batch_idx] = object_pc

        return {
            'robot_name': robot_name,
            'object_name': object_name,
            'initial_q': initial_q_batch,
            'initial_se3': torch.stack(initial_se3_batch, dim=0),
            'object_pc': object_pc_batch,
            'robot_links_pc': robot_links_pc_batch,
        }

    def __getitem__(self, index):
        if self.is_train:
            robot_name = random.choice(self.all_robot_names)
            if robot_name in self.gt_robot_set:
                return self._get_gt_batch(robot_name)
            else:
                return self._get_nogt_batch(robot_name)
        else:
            robot_name, object_name = self.combination[index]
            return self._get_validate_batch(robot_name, object_name)

    def __len__(self):
        if self.is_train:
            return math.ceil(len(self.metadata) / self.batch_size)
        else:
            return len(self.combination)


def ce_custom_collate_fn(batch):
    return batch[0]


def create_ce_dataloader(cfg, is_train):
    dataset = CrossEmbodimentDataset(
        batch_size=cfg.batch_size,
        gt_robot_names=list(cfg.gt_robot_names),
        no_gt_robot_names=list(cfg.get('no_gt_robot_names', [])),
        is_train=is_train,
        debug_object_names=cfg.get('debug_object_names'),
        object_pc_type=cfg.get('object_pc_type', 'random'),
        no_gt_joint_range=cfg.get('no_gt_joint_range', [0.1, 0.9]),
        no_gt_q_noise_std=cfg.get('no_gt_q_noise_std', 0.0),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=ce_custom_collate_fn,
        num_workers=cfg.get('num_workers', 0),
        shuffle=is_train,
    )
    return dataloader
