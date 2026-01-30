"""
TwoStageFlowMatchingGraph: Two-stage SE(3) flow matching for zero-shot cross-embodiment grasping.

Stage 1: Palm-centric hand configuration prior (RR self-attention only, no objects)
Stage 2: Object-conditioned grasp generation (full OR + RR graph, palm velocity head)

Key design:
- Robot nodes always use palm-centric poses → weight transfer from Stage 1 to Stage 2
- RR edges are frame-invariant → perfect transfer
- OR edges computed by converting palm-centric to object-frame
- Palm velocity head predicts palm SE(3) in object frame (Stage 2 only)
"""

import math
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from theseus.geometry.so3 import SO3
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix

from bps_torch.bps import bps_torch
from model.vqvae.vq_vae import VQVAE
from utils.rotation import (
    vector_to_matrix,
    matrix_to_vector,
    compute_relative_se3,
    compute_batch_relative_se3,
)
from model.denoiser import GraphDenoiser
from utils.hand_model import create_hand_model
from utils.palm_centric import (
    get_palm_link_name,
    get_palm_index,
    compute_canonical_scale,
    compute_all_canonical_scales,
)


class PalmVelocityHead(nn.Module):
    """Predicts palm SE(3) velocity from pooled robot + object features."""

    def __init__(self, robot_feat_dim=384, object_feat_dim=384, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(robot_feat_dim + object_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),  # 3D trans velocity + 3D rot velocity
        )

    def forward(self, robot_features, object_features):
        """
        Args:
            robot_features: [B, L, D] robot node features from denoiser
            object_features: [B, P, D] object node features from denoiser

        Returns:
            palm_velocity: [B, 6] (3D trans + 3D rot in so(3))
        """
        robot_pool = robot_features.max(dim=1).values  # [B, D]
        object_pool = object_features.max(dim=1).values  # [B, D]
        combined = torch.cat([robot_pool, object_pool], dim=-1)
        return self.net(combined)


class TwoStageFlowMatchingGraph(nn.Module):

    def __init__(
        self,
        stage,  # 1 or 2
        vqvae_cfg,
        vqvae_pretrain,
        object_patch,
        max_link_node,
        robot_links,
        inference_config,
        bps_config,
        N_t_training,
        flow_matching_config,
        denoiser_config,
        embodiment,
        loss_config,
        palm_config=None,
        mode="train",
    ):
        super(TwoStageFlowMatchingGraph, self).__init__()

        self.stage = stage
        assert stage in [1, 2], f"stage must be 1 or 2, got {stage}"

        # ---- VQ-VAE encoder (frozen, needed for Stage 2) ----
        if isinstance(vqvae_cfg, dict):
            vqvae_cfg = OmegaConf.create(vqvae_cfg)
        self.vqvae = VQVAE(vqvae_cfg)
        if vqvae_pretrain is not None:
            state_dict = torch.load(vqvae_pretrain, map_location="cpu")
            self.vqvae.load_state_dict(state_dict)
        for param in self.vqvae.parameters():
            param.requires_grad = False

        # ---- Meta ----
        self.embodiment = embodiment
        self.hand_dict = {}
        for hand_name in self.embodiment:
            self.hand_dict[hand_name] = create_hand_model(hand_name)

        self.object_patch = object_patch
        self.max_link_node = max_link_node
        self.robot_links = robot_links

        # ---- Palm config ----
        if palm_config is None:
            palm_config = {}
        if isinstance(palm_config, dict) and "palm_link_names" not in palm_config:
            # Use defaults
            from utils.palm_centric import PALM_LINK_NAMES
            palm_config["palm_link_names"] = {
                k: v for k, v in PALM_LINK_NAMES.items() if k in self.embodiment
            }

        self.palm_link_names = palm_config.get("palm_link_names", {})
        self.palm_indices = {}
        for robot_name in self.embodiment:
            if robot_name in self.hand_dict:
                link_names = list(self.hand_dict[robot_name].links_pc.keys())
                palm_name = self.palm_link_names.get(robot_name)
                if palm_name and palm_name in link_names:
                    self.palm_indices[robot_name] = link_names.index(palm_name)

        # Compute canonical scales
        print("Computing canonical scales for palm-centric normalization...")
        self.canonical_scales = {}
        for robot_name in self.embodiment:
            if robot_name in self.hand_dict:
                self.canonical_scales[robot_name] = compute_canonical_scale(
                    self.hand_dict[robot_name], robot_name
                )
                print(f"  {robot_name}: {self.canonical_scales[robot_name]:.4f}")

        # ---- Link embedding (BPS) ----
        if isinstance(bps_config, dict):
            bps_config = OmegaConf.create(bps_config)
        self.link_embed_dim = bps_config.n_bps_points + 4
        self.bps = bps_torch(**bps_config)
        self.link_token_encoder = nn.Sequential(
            nn.Linear(self.link_embed_dim, self.link_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.link_embed_dim, self.link_embed_dim),
        )

        # ---- Flow matching schedule ----
        self.N_t_training = N_t_training
        if isinstance(flow_matching_config, dict):
            flow_matching_config = OmegaConf.create(flow_matching_config)
        self.init_flow_matching(flow_matching_config)

        # ---- Denoiser ----
        denoiser_dict = (
            OmegaConf.to_container(denoiser_config, resolve=True)
            if not isinstance(denoiser_config, dict)
            else denoiser_config
        )
        self.denoiser = GraphDenoiser(
            M=flow_matching_config["M"],
            object_patch=self.object_patch,
            max_link_node=self.max_link_node,
            **denoiser_dict,
        )

        # ---- Palm velocity head (Stage 2 only) ----
        if self.stage == 2:
            v_conv_dim = denoiser_dict.get("v_conv_dim", 384)
            self.palm_velocity_head = PalmVelocityHead(
                robot_feat_dim=v_conv_dim,
                object_feat_dim=v_conv_dim,
            )

        self.mode = mode
        self.link_embeddings = self.construct_bps()

        if self.mode == "train":
            self.loss_config = loss_config
        elif self.mode == "test":
            if not isinstance(inference_config, dict):
                inference_config = OmegaConf.to_container(inference_config, resolve=True)
            self.inference_mode = inference_config["inference_mode"]
            if self.inference_mode == "palm_conditioned":
                self.t_start = inference_config.get("t_start", 0.15)
                self.interpolation_rate = inference_config["interpolation_rate"]
                self.interpolation_clip = (
                    inference_config["interpolation_clip"] * math.pi / 180
                )

    # ------------------------------------------------------------------
    # Flow matching init
    # ------------------------------------------------------------------
    def init_flow_matching(self, cfg):
        self.M = cfg["M"]
        self.ode_steps = cfg["ode_steps"]
        self.solver = cfg.get("solver", "euler")

    # ------------------------------------------------------------------
    # BPS construction (same as original)
    # ------------------------------------------------------------------
    def construct_bps(self):
        link_embedding_dict = {}
        for embodiment, hand_model in self.hand_dict.items():
            links_pc = hand_model.links_pc
            embodiment_bps = []
            for link_name, link_pc in links_pc.items():
                centroid, scale = self._unit_ball_(link_pc)
                link_pc_norm = (link_pc - centroid) / scale
                link_bps = self.bps.encode(
                    link_pc_norm,
                    feature_type=["dists"],
                    x_features=None,
                    custom_basis=None,
                )["dists"]
                link_bps = torch.cat([link_bps, centroid, scale.view(1, 1)], dim=-1)
                embodiment_bps.append(link_bps)
            link_embedding_dict[embodiment] = torch.cat(embodiment_bps, dim=0)
        return link_embedding_dict

    def _unit_ball_(self, pc):
        centroid = torch.mean(pc, dim=0, keepdim=True)
        pc = pc - centroid
        max_radius = pc.norm(dim=-1).max()
        return centroid, max_radius

    def _normalize_pc_(self, pc):
        B, N, _ = pc.shape
        centroids = torch.mean(pc, dim=1, keepdim=True)
        pc = pc - centroids
        scale, _ = torch.max(torch.abs(pc), dim=1, keepdim=True)
        scale, _ = torch.max(scale, dim=2, keepdim=True)
        pc = pc / scale
        return pc, centroids, scale

    def _expand_and_reshape_(self, x, name):
        shape = x.shape
        B = x.shape[0]
        if len(shape) == 3:
            return (
                x[:, None, :, :]
                .expand(-1, self.N_t_training, -1, -1)
                .reshape(B * self.N_t_training, shape[1], shape[2])
            )
        elif len(shape) == 4:
            return (
                x[:, None, :, :, :]
                .expand(-1, self.N_t_training, -1, -1, -1)
                .reshape(B * self.N_t_training, shape[1], shape[2], shape[3])
            )
        elif len(shape) == 2:
            return (
                x[:, None, :]
                .expand(-1, self.N_t_training, -1)
                .reshape(B * self.N_t_training, shape[1])
            )
        elif len(shape) == 1:
            return (
                x[:, None]
                .expand(-1, self.N_t_training)
                .reshape(B * self.N_t_training)
            )
        else:
            raise ValueError(f"Unsupported shape for {name}: {shape}")

    # ------------------------------------------------------------------
    # Graph construction helpers
    # ------------------------------------------------------------------
    def _build_object_nodes(self, object_pc):
        """Encode object point cloud via frozen VQ-VAE."""
        with torch.no_grad():
            normal_pc, centroids, scale = self._normalize_pc_(object_pc)
            object_tokens = self.vqvae.encode(normal_pc)
        object_nodes = torch.cat(
            [
                object_tokens["xyz"],
                scale.expand(-1, self.object_patch, -1),
                object_tokens["z_q"],
            ],
            dim=-1,
        )
        return object_nodes, centroids, scale

    def _build_robot_nodes_palm_centric(self, batch, B, device, dtype):
        """Build robot nodes using palm-centric pose vectors (both stages)."""
        palm_centric_vec = batch["palm_centric_vec"]
        link_target_poses = torch.zeros(
            [B, self.max_link_node, 6], device=device, dtype=dtype
        )
        link_robot_embeds = torch.zeros(
            [B, self.max_link_node, self.link_embed_dim], device=device, dtype=dtype
        )
        link_node_masks = torch.zeros(
            [B, self.max_link_node], device=device, dtype=torch.bool
        )

        for b in range(B):
            robot_name = batch["robot_name"][b]
            num_link = self.robot_links[robot_name]
            pose_vec = palm_centric_vec[b].to(device)  # [L, 6] already normalized

            link_target_poses[b, :num_link, :] = pose_vec[:num_link]

            link_bps = self.link_embeddings[robot_name].to(device)
            link_embed = self.link_token_encoder(link_bps)
            link_robot_embeds[b, :num_link, :] = link_embed
            link_node_masks[b, :num_link] = True

        robot_nodes = torch.cat([link_target_poses, link_robot_embeds], dim=-1)
        return robot_nodes, link_node_masks

    def _build_rr_edges(self, robot_nodes, batch, B, device, dtype):
        """Build robot-robot relative SE(3) edges."""
        norm_robot_vec = robot_nodes[:, :, :6]
        link_rel_poses = torch.zeros(
            [B, self.max_link_node, self.max_link_node, 6], device=device, dtype=dtype
        )
        for b in range(B):
            robot_name = batch["robot_name"][b]
            num_link = self.robot_links[robot_name]
            T = vector_to_matrix(norm_robot_vec[b])[:num_link]
            rel_pose = compute_relative_se3(T, T)
            link_rel_poses[b, :num_link, :num_link, :] = matrix_to_vector(rel_pose)
        return link_rel_poses

    def _build_or_edges_from_object_frame(
        self, palm_centric_poses, palm_poses_in_object, object_nodes, batch, B, device, dtype
    ):
        """
        Build object-robot edges by converting palm-centric poses to object frame.

        T_link_in_object = T_palm_in_object @ T_link_in_palm
        """
        link_object_rel_poses = torch.zeros(
            [B, self.max_link_node, self.object_patch, 6], device=device, dtype=dtype
        )
        object_positions = object_nodes[:, :, :3]
        object_se3 = (
            torch.eye(4, device=device, dtype=dtype)
            .expand(B, self.object_patch, -1, -1)
            .clone()
        )
        object_se3[:, :, :3, 3] = object_positions

        for b in range(B):
            robot_name = batch["robot_name"][b]
            num_link = self.robot_links[robot_name]

            # Palm-centric link poses
            T_link_palm = vector_to_matrix(palm_centric_poses[b, :num_link, :6])  # [L, 4, 4]

            # Palm pose in object frame
            T_palm_obj = vector_to_matrix(palm_poses_in_object[b].unsqueeze(0))  # [1, 4, 4]

            # Convert to object frame
            T_link_obj = T_palm_obj @ T_link_palm  # [L, 4, 4]

            # Compute relative SE3 to object patches
            T_o = object_se3[b]  # [P, 4, 4]
            T_rel = compute_relative_se3(T_link_obj, T_o)  # [L, P, 4, 4]
            link_object_rel_poses[b, :num_link, : self.object_patch] = (
                matrix_to_vector(T_rel)
            )

        return link_object_rel_poses

    def _build_noisy_rr_edges(self, noisy_V_R, device, dtype):
        """Build RR edges from noisy/interpolated robot poses."""
        noisy_V_R_se3 = vector_to_matrix(noisy_V_R[:, :, :6])
        noisy_E_RR = matrix_to_vector(
            compute_batch_relative_se3(noisy_V_R_se3, noisy_V_R_se3)
        )
        return noisy_E_RR

    def _build_noisy_or_edges(self, noisy_V_R, noisy_palm_poses, V_O, device, dtype):
        """Build OR edges from noisy palm-centric poses + noisy palm pose."""
        BT = noisy_V_R.shape[0]
        P = V_O.shape[1]

        object_positions = V_O[:, :, :3]
        object_se3 = (
            torch.eye(4, device=device, dtype=dtype).expand(BT, P, -1, -1).clone()
        )
        object_se3[:, :, :3, 3] = object_positions

        # Convert palm-centric to object frame
        T_link_palm = vector_to_matrix(noisy_V_R[:, :, :6])  # [BT, L, 4, 4]
        T_palm_obj = vector_to_matrix(noisy_palm_poses.unsqueeze(1))  # [BT, 1, 4, 4]
        T_link_obj = T_palm_obj @ T_link_palm  # [BT, L, 4, 4]

        noisy_E_OR = matrix_to_vector(
            compute_batch_relative_se3(T_link_obj, object_se3)
        )
        return noisy_E_OR

    # ------------------------------------------------------------------
    # Forward: Stage 1 (RR-only, palm-centric)
    # ------------------------------------------------------------------
    def _forward_stage1(self, batch, eps=1e-8):
        B = len(batch["robot_name"])
        device = next(self.parameters()).device
        dtype = torch.float32

        # Build robot nodes (palm-centric)
        robot_nodes, link_node_masks = self._build_robot_nodes_palm_centric(
            batch, B, device, dtype
        )

        # Build dummy object nodes and OR edges (needed for denoiser shape)
        dummy_V_O = torch.zeros(
            [B, self.object_patch, 68], device=device, dtype=dtype
        )
        dummy_E_OR = torch.zeros(
            [B, self.max_link_node, self.object_patch, 6], device=device, dtype=dtype
        )

        # Expand for N_t_training timesteps
        BT = B * self.N_t_training
        V_O = self._expand_and_reshape_(dummy_V_O, "V_O")
        V_R = self._expand_and_reshape_(robot_nodes, "V_R")
        V_R_trans = V_R[:, :, :3]
        V_R_rot = V_R[:, :, 3:6]
        V_R_embed = V_R[:, :, 6:]

        # Sample continuous time t ~ U(0,1)
        t_continuous = torch.rand(BT, device=device, dtype=dtype)
        t_idx = (t_continuous * (self.M - 1)).long()

        # Translation: R³ linear interpolation
        x0_trans = torch.randn_like(V_R_trans)
        t_bc = t_continuous[:, None, None]
        xt_trans = (1.0 - t_bc) * x0_trans + t_bc * V_R_trans
        v_target_trans = V_R_trans - x0_trans

        # Rotation: SO(3) geodesic interpolation
        eps_rot = torch.randn_like(V_R_rot)
        flat_eps = eps_rot.reshape(-1, 3)
        R0_flat = SO3.exp_map(flat_eps).to_matrix()

        flat_x1_rot = V_R_rot.reshape(-1, 3)
        R1_flat = axis_angle_to_matrix(flat_x1_rot)

        R0T_R1 = R0_flat.transpose(-2, -1) @ R1_flat
        log_R0T_R1 = SO3(tensor=R0T_R1).log_map()

        t_rot_bc = (
            t_continuous[:, None]
            .expand(-1, self.max_link_node)
            .reshape(-1, 1)
        )
        scaled_log = t_rot_bc * log_R0T_R1
        Rt_delta = SO3.exp_map(scaled_log).to_matrix()
        Rt_flat = R0_flat @ Rt_delta

        xt_rot = matrix_to_axis_angle(Rt_flat).reshape(BT, self.max_link_node, 3)
        v_target_rot = log_R0T_R1.reshape(BT, self.max_link_node, 3)

        # Assemble interpolated node features
        noisy_V_R = torch.cat([xt_trans, xt_rot, V_R_embed], dim=-1)

        # Build edges from interpolated poses (RR only)
        noisy_E_RR = self._build_noisy_rr_edges(noisy_V_R, device, dtype)
        noisy_E_OR = self._expand_and_reshape_(dummy_E_OR, "E_OR")

        # Predict velocity (skip OR attention)
        pred_velocity = self.denoiser(
            V_O, noisy_V_R, noisy_E_OR, noisy_E_RR, t_idx, skip_or=True
        )

        # Loss computation
        M_V_R = self._expand_and_reshape_(link_node_masks, "M_V_R").float()
        pred_v_trans = pred_velocity[:, :, :3]
        pred_v_rot = pred_velocity[:, :, 3:]

        error_trans = (v_target_trans - pred_v_trans) ** 2
        error_trans = error_trans.mean(dim=-1)
        loss_trans = (error_trans * M_V_R).sum() / (M_V_R.sum() + eps)

        error_rot = (v_target_rot - pred_v_rot) ** 2
        error_rot = error_rot.mean(dim=-1)
        loss_rot = (error_rot * M_V_R).sum() / (M_V_R.sum() + eps)

        total_loss = (
            self.loss_config["trans_weight"] * loss_trans
            + self.loss_config["rot_weight"] * loss_rot
        )

        return {
            "loss_rot": loss_rot,
            "loss_trans": loss_trans,
            "loss_total": total_loss,
        }

    # ------------------------------------------------------------------
    # Forward: Stage 2 (full graph + palm velocity)
    # ------------------------------------------------------------------
    def _forward_stage2(self, batch, eps=1e-8):
        object_pc = batch["object_pc"]
        B = object_pc.shape[0]
        device = object_pc.device
        dtype = object_pc.dtype

        # Object encoding
        object_nodes, centroids, scale = self._build_object_nodes(object_pc)

        # Robot nodes (palm-centric, same normalization as Stage 1)
        robot_nodes, link_node_masks = self._build_robot_nodes_palm_centric(
            batch, B, device, dtype
        )

        # Palm poses in object-normalized frame
        palm_poses = torch.zeros([B, 6], device=device, dtype=dtype)
        for b in range(B):
            palm_poses[b] = batch["palm_pose_in_object"][b].to(device)

        # Expand for N_t_training timesteps
        BT = B * self.N_t_training
        V_O = self._expand_and_reshape_(object_nodes, "V_O")
        V_R = self._expand_and_reshape_(robot_nodes, "V_R")
        V_R_trans = V_R[:, :, :3]
        V_R_rot = V_R[:, :, 3:6]
        V_R_embed = V_R[:, :, 6:]

        palm_poses_exp = self._expand_and_reshape_(palm_poses, "palm_poses")  # [BT, 6]

        # Sample continuous time t ~ U(0,1)
        t_continuous = torch.rand(BT, device=device, dtype=dtype)
        t_idx = (t_continuous * (self.M - 1)).long()

        # ---- Finger flow matching (same as Stage 1) ----
        x0_trans = torch.randn_like(V_R_trans)
        t_bc = t_continuous[:, None, None]
        xt_trans = (1.0 - t_bc) * x0_trans + t_bc * V_R_trans
        v_target_trans = V_R_trans - x0_trans

        eps_rot = torch.randn_like(V_R_rot)
        flat_eps = eps_rot.reshape(-1, 3)
        R0_flat = SO3.exp_map(flat_eps).to_matrix()

        flat_x1_rot = V_R_rot.reshape(-1, 3)
        R1_flat = axis_angle_to_matrix(flat_x1_rot)

        R0T_R1 = R0_flat.transpose(-2, -1) @ R1_flat
        log_R0T_R1 = SO3(tensor=R0T_R1).log_map()

        t_rot_bc = (
            t_continuous[:, None]
            .expand(-1, self.max_link_node)
            .reshape(-1, 1)
        )
        scaled_log = t_rot_bc * log_R0T_R1
        Rt_delta = SO3.exp_map(scaled_log).to_matrix()
        Rt_flat = R0_flat @ Rt_delta

        xt_rot = matrix_to_axis_angle(Rt_flat).reshape(BT, self.max_link_node, 3)
        v_target_rot = log_R0T_R1.reshape(BT, self.max_link_node, 3)

        # ---- Palm flow matching ----
        palm_x1_trans = palm_poses_exp[:, :3]  # target palm translation
        palm_x1_rot = palm_poses_exp[:, 3:]  # target palm rotation

        palm_x0_trans = torch.randn_like(palm_x1_trans)
        palm_t_bc = t_continuous[:, None]
        palm_xt_trans = (1.0 - palm_t_bc) * palm_x0_trans + palm_t_bc * palm_x1_trans
        palm_v_target_trans = palm_x1_trans - palm_x0_trans

        palm_eps_rot = torch.randn_like(palm_x1_rot)
        palm_R0 = SO3.exp_map(palm_eps_rot).to_matrix()
        palm_R1 = axis_angle_to_matrix(palm_x1_rot)
        palm_R0T_R1 = palm_R0.transpose(-2, -1) @ palm_R1
        palm_log = SO3(tensor=palm_R0T_R1).log_map()
        palm_scaled_log = t_continuous[:, None] * palm_log
        palm_Rt = palm_R0 @ SO3.exp_map(palm_scaled_log).to_matrix()
        palm_xt_rot = matrix_to_axis_angle(palm_Rt)
        palm_v_target_rot = palm_log

        noisy_palm_poses = torch.cat([palm_xt_trans, palm_xt_rot], dim=-1)  # [BT, 6]

        # ---- Assemble noisy features ----
        noisy_V_R = torch.cat([xt_trans, xt_rot, V_R_embed], dim=-1)

        # Build edges
        noisy_E_RR = self._build_noisy_rr_edges(noisy_V_R, device, dtype)
        noisy_E_OR = self._build_noisy_or_edges(
            noisy_V_R, noisy_palm_poses, V_O, device, dtype
        )

        # Predict velocity (full graph with OR + RR)
        pred_velocity, robot_feats, object_feats = self.denoiser(
            V_O, noisy_V_R, noisy_E_OR, noisy_E_RR, t_idx,
            skip_or=False, return_features=True,
        )

        # Palm velocity prediction
        pred_palm_velocity = self.palm_velocity_head(robot_feats, object_feats)  # [BT, 6]

        # ---- Loss: fingers ----
        M_V_R = self._expand_and_reshape_(link_node_masks, "M_V_R").float()
        pred_v_trans = pred_velocity[:, :, :3]
        pred_v_rot = pred_velocity[:, :, 3:]

        error_trans = (v_target_trans - pred_v_trans) ** 2
        error_trans = error_trans.mean(dim=-1)
        loss_finger_trans = (error_trans * M_V_R).sum() / (M_V_R.sum() + eps)

        error_rot = (v_target_rot - pred_v_rot) ** 2
        error_rot = error_rot.mean(dim=-1)
        loss_finger_rot = (error_rot * M_V_R).sum() / (M_V_R.sum() + eps)

        # ---- Loss: palm ----
        pred_palm_v_trans = pred_palm_velocity[:, :3]
        pred_palm_v_rot = pred_palm_velocity[:, 3:]

        palm_v_target = torch.cat([palm_v_target_trans, palm_v_target_rot], dim=-1)
        pred_palm_v = torch.cat([pred_palm_v_trans, pred_palm_v_rot], dim=-1)
        loss_palm = ((palm_v_target - pred_palm_v) ** 2).mean()

        # ---- Total loss ----
        palm_weight = self.loss_config.get("palm_weight", 1.0)
        total_loss = (
            self.loss_config["trans_weight"] * loss_finger_trans
            + self.loss_config["rot_weight"] * loss_finger_rot
            + palm_weight * loss_palm
        )

        return {
            "loss_finger_rot": loss_finger_rot,
            "loss_finger_trans": loss_finger_trans,
            "loss_palm": loss_palm,
            "loss_total": total_loss,
        }

    def forward(self, batch, eps=1e-8):
        if self.stage == 1:
            return self._forward_stage1(batch, eps)
        else:
            return self._forward_stage2(batch, eps)

    # ------------------------------------------------------------------
    # Inference: Stage 2 ODE integration
    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference(self, batch, eps=1e-8):
        assert self.stage == 2, "Inference only supported for Stage 2"

        object_pc = batch["object_pc"]
        B = object_pc.shape[0]
        device = object_pc.device
        dtype = object_pc.dtype

        robot_name = batch["robot_name"]
        link_names = list(batch["robot_links_pc"][0].keys())
        valid_links = self.robot_links[robot_name]

        # Object encoding
        object_nodes, centroids, scale = self._build_object_nodes(object_pc)

        N_steps = self.ode_steps
        dt = 1.0 / N_steps

        # Initialize noise for fingers (palm-centric)
        V_R_trans = torch.randn(
            [B, self.max_link_node, 3], device=device, dtype=dtype
        )
        eps_rot = torch.randn(
            [B, self.max_link_node, 3], device=device, dtype=dtype
        )
        R0 = SO3.exp_map(eps_rot.reshape(-1, 3)).to_matrix()
        V_R_rot = matrix_to_axis_angle(R0).reshape(B, self.max_link_node, 3)

        # Initialize noise for palm pose
        palm_trans = torch.randn([B, 3], device=device, dtype=dtype)
        palm_eps_rot = torch.randn([B, 3], device=device, dtype=dtype)
        palm_R0 = SO3.exp_map(palm_eps_rot).to_matrix()
        palm_rot = matrix_to_axis_angle(palm_R0)  # [B, 3]

        # Link embeddings
        link_robot_embeds = torch.zeros(
            [B, self.max_link_node, self.link_embed_dim], device=device, dtype=dtype
        )
        num_link = self.robot_links[robot_name]
        link_bps = self.link_embeddings[robot_name].to(device)
        link_embed = self.link_token_encoder(link_bps)
        for b in range(B):
            link_robot_embeds[b, :num_link, :] = link_embed

        for step in range(N_steps):
            t_current = step * dt
            t_idx = int(round(t_current * (self.M - 1)))
            t_idx = min(t_idx, self.M - 1)
            t_tensor = torch.full((B,), t_idx, dtype=torch.long, device=device)

            # Build graph
            noisy_V_R = torch.cat(
                [V_R_trans, V_R_rot, link_robot_embeds], dim=-1
            )
            noisy_palm_poses = torch.cat([palm_trans, palm_rot], dim=-1)  # [B, 6]

            noisy_E_RR = self._build_noisy_rr_edges(noisy_V_R, device, dtype)
            noisy_E_OR = self._build_noisy_or_edges(
                noisy_V_R, noisy_palm_poses, object_nodes, device, dtype
            )

            # Predict velocity
            pred_velocity, robot_feats, object_feats = self.denoiser(
                object_nodes, noisy_V_R, noisy_E_OR, noisy_E_RR, t_tensor,
                skip_or=False, return_features=True,
            )
            pred_v_trans = pred_velocity[:, :, :3]
            pred_v_rot = pred_velocity[:, :, 3:]

            # Palm velocity
            pred_palm_v = self.palm_velocity_head(robot_feats, object_feats)
            pred_palm_v_trans = pred_palm_v[:, :3]
            pred_palm_v_rot = pred_palm_v[:, 3:]

            # Euler step: fingers
            V_R_trans = V_R_trans + dt * pred_v_trans
            R_current = axis_angle_to_matrix(V_R_rot.reshape(-1, 3))
            delta_R = SO3.exp_map(dt * pred_v_rot.reshape(-1, 3)).to_matrix()
            R_new = R_current @ delta_R
            V_R_rot = matrix_to_axis_angle(R_new).reshape(B, self.max_link_node, 3)

            # Euler step: palm
            palm_trans = palm_trans + dt * pred_palm_v_trans
            palm_R_cur = axis_angle_to_matrix(palm_rot)
            palm_delta_R = SO3.exp_map(dt * pred_palm_v_rot).to_matrix()
            palm_R_new = palm_R_cur @ palm_delta_R
            palm_rot = matrix_to_axis_angle(palm_R_new)

        # Convert palm-centric to world frame
        # First denormalize palm-centric translations by canonical scale
        canonical_scale = self.canonical_scales[robot_name]
        finger_trans_denorm = V_R_trans * canonical_scale  # [B, L, 3]
        finger_rot = V_R_rot  # [B, L, 3]
        finger_pose = torch.cat([finger_trans_denorm, finger_rot], dim=-1)  # [B, L, 6]

        # Palm pose: denormalize by object scale
        palm_trans_world = palm_trans * scale.squeeze(-1) + centroids.squeeze(1)  # [B, 3]
        palm_pose_world = torch.cat([palm_trans_world, palm_rot], dim=-1)  # [B, 6]

        # Compose: T_link_world = T_palm_world @ T_link_palm
        T_link_palm = vector_to_matrix(finger_pose[:, :valid_links])  # [B, L, 4, 4]
        T_palm_world = vector_to_matrix(palm_pose_world.unsqueeze(1))  # [B, 1, 4, 4]
        T_link_world = T_palm_world @ T_link_palm  # [B, L, 4, 4]

        predict_link_pose_dict = {}
        for link_id, link_name in enumerate(link_names):
            predict_link_pose_dict[link_name] = T_link_world[:, link_id]

        return {0: predict_link_pose_dict}
