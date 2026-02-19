from typing import Dict

import pytorch3d
from pytorch3d.ops import knn_points
import torch


def ERF_loss(obj_pcd_nor: torch.Tensor, hand_pcd: torch.Tensor):
    """
    Calculate the penalty loss based on point cloud and normal.

    :param obj_pcd_nor: B x N_obj x 6 (object point cloud with normals)
    :param hand_pcd: B x N_hand x 3 (hand point cloud)
    :return: ERF_loss (scalar)
    """
    b = hand_pcd.shape[0]
    n_obj = obj_pcd_nor.shape[1]
    n_hand = hand_pcd.shape[1]

    # Separate object point cloud and normals
    obj_pcd = obj_pcd_nor[:, :, :3]
    obj_nor = obj_pcd_nor[:, :, 3:6]

    # Compute K-nearest neighbors
    knn_result = knn_points(hand_pcd, obj_pcd, K=1, return_nn=True)
    distances = knn_result.dists
    indices = knn_result.idx
    knn = knn_result.knn
    distances = distances.sqrt()
    # Extract the closest object points and normals
    hand_obj_points = torch.gather(obj_pcd, 1, indices.expand(-1, -1, 3))
    hand_obj_normals = torch.gather(obj_nor, 1, indices.expand(-1, -1, 3))
    # Compute the signs
    hand_obj_signs = ((hand_obj_points - hand_pcd) * hand_obj_normals).sum(dim=2)
    hand_obj_signs = (hand_obj_signs > 0.).float()
    # Compute collision value
    # collision_value = (hand_obj_signs * hand_obj_dist).mean(dim=1)
    collision_value = (hand_obj_signs * distances.squeeze(2)).max(dim=1).values
    ERF_loss = collision_value.mean()
    return ERF_loss

def SPF_loss(dis_points, obj_pcd: torch.Tensor, thres_dis = 0.02 ):
    dis_points = dis_points.to(dtype=torch.float32)
    obj_pcd = obj_pcd.to(dtype=torch.float32)
    dis_pred = pytorch3d.ops.knn_points(dis_points, obj_pcd).dists[:, :, 0] # 64*140  # squared chamfer distance from object_pc to contact_candidates_pred
    small_dis_pred = dis_pred < thres_dis ** 2# 64*140
    SPF_loss = dis_pred[small_dis_pred].sqrt().sum() / (small_dis_pred.sum().item() + 1e-5)#1
    return SPF_loss

def SRF_loss(points):
    B, *points_shape = points.shape
    dis_spen = (points.unsqueeze(1) - points.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
    dis_spen = torch.where(dis_spen < 1e-6, 1e6 * torch.ones_like(dis_spen), dis_spen)
    dis_spen = 0.02 - dis_spen
    dis_spen[dis_spen < 0] = 0
    SRF_loss = dis_spen.sum() / B
    return SRF_loss