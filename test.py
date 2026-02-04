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
from model.tro_graph import RobotGraph
from utils.hand_model import create_hand_model
from utils.pyroki_ik import PyrokiRetarget
from utils.optimization import *
from validation.validate_utils import validate_isaac


# ── Fingertip joint extraction for pure-rotation joints ─────────────────
# These joints have zero positional Jacobian, so IK leaves them unchanged.
# We extract their angles from the diffusion model's predicted SE3 rotations.

# LeapHand: 使用 fixed joint 的 extra tip 链接替代 revolute joint 的 fingertip 链接
# 这与 Allegro 的 link_*_tip 处理方式一致
# NOTE: thumb_fingertip 不包含，因为 extra_thumb_tip_head 的 Z 偏移方向相反
# (-0.015 vs +0.015)，会导致大拇指更收拢
LEAPHAND_TIP_MAPPING = {
    'fingertip': 'extra_index_tip_head',
    'fingertip_2': 'extra_middle_tip_head',
    'fingertip_3': 'extra_ring_tip_head',
    # thumb_fingertip 保持原样，不使用 extra_thumb_tip_head
}

# LeapHand per-link IK 权重配置
LEAPHAND_LINK_WEIGHTS = {
    'palm_lower': 2.0,       # 手掌权重增大，优先对齐手掌
    'thumb_fingertip': 0.8,  # 大拇指权重略降
}

FINGERTIP_JOINTS = {
    'leaphand': {
        # (parent_link, child_link): q_index
        ('dip', 'fingertip'): 9,           # joint 3
        ('dip_2', 'fingertip_2'): 13,      # joint 7
        ('dip_3', 'fingertip_3'): 17,      # joint 11
        ('thumb_dip', 'thumb_fingertip'): 21,  # joint 15
    },
}


def extract_fingertip_joints(predict_q, transform_dict, robot_name):
    """
    Extract fingertip joint angles from diffusion SE3 rotation.

    For joints that are pure rotation (don't affect link position), the IK
    solver cannot optimize them. Instead, we compute the joint angle from
    the relative rotation between parent and child links in the diffusion output.
    """
    if robot_name not in FINGERTIP_JOINTS:
        return predict_q

    for (parent, child), q_idx in FINGERTIP_JOINTS[robot_name].items():
        if parent not in transform_dict or child not in transform_dict:
            continue
        R_parent = transform_dict[parent][:, :3, :3]  # (B, 3, 3)
        R_child = transform_dict[child][:, :3, :3]
        # Relative rotation: R_rel = R_parent^T @ R_child
        R_rel = torch.bmm(R_parent.transpose(-1, -2), R_child)
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


def prepare_input(batch, device):

    batch['object_pc'] = batch['object_pc'].to(device)
    batch['initial_q'] = [x.to(device) for x in batch['initial_q']]
    
    return batch

def test(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building dataloader...")
    dataloader = create_dataloader(config.dataset, is_train=False)
    print("Building model...")
    model = RobotGraph(**config.model).to(device)
    
    state_dict = torch.load(config.test.ckpt)["model_state"]
    model.load_state_dict(state_dict, strict=False)

    with open("data/data_urdf/robot/urdf_assets_meta.json", 'r') as f:
        robot_urdf_meta = json.load(f)
    
    #### Compile Jax for Pyroki ####
    robot_name = config.test.embodiment
    hand = create_hand_model(robot_name, device)
    urdf_path = robot_urdf_meta["urdf_path"][robot_name]
    target_links = list(hand.links_pc.keys())

    # LeapHand: 用 fixed joint 的 extra tip 链接替代 revolute joint 的 fingertip 链接
    # 这与 Allegro 使用 link_*_tip 的方式一致，避免 revolute 指尖关节的零雅可比问题
    # NOTE: thumb 不在映射中，因为 extra_thumb_tip_head Z 偏移方向相反
    target_links_ik = target_links.copy()
    link_weights = None
    if robot_name == 'leaphand':
        for old_tip, new_tip in LEAPHAND_TIP_MAPPING.items():
            if old_tip in target_links_ik:
                idx = target_links_ik.index(old_tip)
                target_links_ik[idx] = new_tip
        # 构建 per-link 权重数组
        link_weights = [LEAPHAND_LINK_WEIGHTS.get(name, 1.0) for name in target_links_ik]
        print(f"LeapHand: Using extra_*_tip_head for fingers, thumb_fingertip unchanged")
        print(f"LeapHand: Link weights - palm_lower={LEAPHAND_LINK_WEIGHTS.get('palm_lower', 1.0)}")

    ik_solver = PyrokiRetarget(
        urdf_path, target_links_ik,
        hand_joint_names=hand.get_joint_orders(),
        link_weights=link_weights,
    )
    batch_retarget = jax.jit(ik_solver.solve_retarget)
    ################################

    success_dict = {}
    diversity_dict = {}
    vis_info = []
    batch_size = config.dataset.batch_size
    model.eval()
    total_inference_time = 0
    total_grasp_num = 0
    warmed_up = False

    for batch_id, batch in tqdm.tqdm(enumerate(dataloader)):

        transform_dict = {}
        data_count = 0
        predict_q_list = []
        initial_q_list = []      # Unconditioned
        initial_se3_list = []    # Conditioned
        object_pc_list = []
        transform_list = []
        while data_count != batch_size:
            split_num = min(batch_size - data_count, config.test.split_batch_size)
            initial_q = batch["initial_q"][data_count : data_count + split_num].to(device)
            initial_se3 = batch['initial_se3'][data_count : data_count + split_num].to(device)
            object_pc = batch['object_pc'][data_count : data_count + split_num].to(device)
            robot_links_pc = batch['robot_links_pc'][data_count : data_count + split_num]
            split_batch = {
                'robot_name': batch['robot_name'],
                'object_name': batch['object_name'],
                'initial_q': initial_q,
                'initial_se3': initial_se3,
                'object_pc': object_pc,
                'robot_links_pc': robot_links_pc
            }
            data_count += split_num

            time_start = time.time()
            with torch.no_grad():
                all_diffuse_step_poses_dict = model.inference(split_batch)
  
            ## IK process with pyroki
            clean_robot_pose = all_diffuse_step_poses_dict[0]
            optim_transform = process_transform(hand.pk_chain, clean_robot_pose)
            initial_q_jnp = jnp.array(initial_q.cpu().numpy())
            target_pos_list = [optim_transform[name] for name in target_links_ik]
            target_pos = torch.stack(target_pos_list, dim=1)
    
            target_pos_jnp = jnp.array(target_pos.detach().cpu().numpy())
            predict_q_jnp = batch_retarget(
                initial_q=initial_q_jnp,
                target_pos=target_pos_jnp
            )
            jax.block_until_ready(predict_q_jnp)
            time_end = time.time()
            
            if warmed_up:
                total_inference_time += (time_end - time_start)
                total_grasp_num += split_num
            else:
                warmed_up = True

            predict_q = torch.from_numpy(np.array(predict_q_jnp)).to(device=device, dtype=initial_q.dtype)

            # Extract fingertip joint angles from diffusion rotation
            predict_q = extract_fingertip_joints(predict_q, clean_robot_pose, batch['robot_name'])

            initial_q_list.append(initial_q)
            initial_se3_list.append(initial_se3)
            predict_q_list.append(predict_q)
            object_pc_list.append(object_pc)
            for diffuse_step, pred_robot_pose in all_diffuse_step_poses_dict.items():
                if diffuse_step not in transform_dict:
                    transform_dict[diffuse_step] = []
                transform_dict[diffuse_step].append(pred_robot_pose)
        
        # Simulation, isaac subprocess
        all_predict_q = torch.cat(predict_q_list, dim=0)
        
        success, isaac_q = validate_isaac(
            batch['robot_name'], 
            batch["object_name"], 
            all_predict_q, 
            gpu=config.test.gpu
        )
        success_dict[batch["object_name"]] = success

        # Diversity
        success_q = all_predict_q[success]
        diversity_dict[batch["object_name"]] = success_q

        for diffuse_step, transform_list in transform_dict.items():
            transform_batch = {}
            for transform in transform_list:
                for k, v in transform.items():
                    transform_batch[k] = v if k not in transform_batch else torch.cat((transform_batch[k], v), dim=0)
            transform_dict[diffuse_step] = transform_batch

        vis_info.append({
            'robot_name': batch['robot_name'],
            'object_name': batch['object_name'],
            'initial_q': torch.cat(initial_q_list, dim=0),
            'initial_se3': torch.cat(initial_se3_list, dim=0),
            'predict_q': torch.cat(predict_q_list, dim=0),
            'object_pc': torch.cat(object_pc_list, dim=0),
            'predict_transform': transform_dict,
            'success': success,
            'isaac_q': isaac_q
        })

    os.makedirs(config.test.save_dir, exist_ok=True)
    torch.save(
        vis_info,
        os.path.join(config.test.save_dir, "vis.pt")
    )

    output_path = os.path.join(config.test.save_dir, "res.txt")
    with open(output_path, "w") as f:
        total_success = 0
        total_sum = 0
        for obj, obj_res in success_dict.items():
            line = f"{obj}: {obj_res.sum() / len(obj_res)}\n"
            print(line, end="")
            f.write(line)
            total_success += obj_res.sum()
            total_sum += len(obj_res)

        line = f"Total success rate: {total_success / total_sum}.\n"
        print(line, end="")
        f.write(line)

        all_success_q = torch.cat(list(diversity_dict.values()), dim=0)
        diversity_std = torch.std(all_success_q, dim=0).mean()
        line = f"Total diversity: {diversity_std}\n"
        print(line, end="")
        f.write(line)

        line = f"Grasp generation time: {total_inference_time / total_grasp_num} s.\n"
        print(line, end="")
        f.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/test_palm_unconditioned_ezgripper.yaml", help="config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    test(config)
