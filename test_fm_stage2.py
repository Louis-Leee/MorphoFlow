"""
Stage 2 test: Object-conditioned grasp generation via two-stage flow matching.

Supports both standard evaluation (training embodiments) and zero-shot evaluation
(embodiments seen in Stage 1 but not Stage 2).

Usage:
    python test_fm_stage2.py --config config/test_fm_stage2.yaml
    python test_fm_stage2.py --config config/test_fm_stage2_zeroshot_leaphand.yaml
    export JAX_PLATFORM_NAME=cpu  # reduce GPU memory for IK
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
from model.flow_matching_twostage import TwoStageFlowMatchingGraph
from utils.hand_model import create_hand_model
from utils.pyroki_ik import PyrokiRetarget
from utils.optimization import process_transform
from validation.validate_utils import validate_isaac


def load_stage2_checkpoint(model, ckpt_path):
    """Load Stage 2 model weights from a Lightning checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    # Remove Lightning module prefix
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model.model."):
            cleaned[k[len("model."):]] = v
        elif k.startswith("model."):
            cleaned[k[len("model."):]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"  Loaded: {len(cleaned) - len(unexpected)} parameters")
    if missing:
        print(f"  Missing: {len(missing)}")
        for m in missing[:5]:
            print(f"    - {m}")
    if unexpected:
        print(f"  Unexpected: {len(unexpected)}")
        for u in unexpected[:5]:
            print(f"    - {u}")


def test(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building dataloader...")
    dataloader = create_dataloader(config.dataset, is_train=False)

    print("Building model...")
    model_cfg = OmegaConf.to_container(config.model, resolve=True)
    model = TwoStageFlowMatchingGraph(**model_cfg).to(device)

    load_stage2_checkpoint(model, config.test.ckpt)

    with open("data/data_urdf/robot/urdf_assets_meta.json", "r") as f:
        robot_urdf_meta = json.load(f)

    #### Compile Jax for Pyroki ####
    robot_name = config.test.embodiment
    hand = create_hand_model(robot_name, device)
    urdf_path = robot_urdf_meta["urdf_path"][robot_name]
    target_links = list(hand.links_pc.keys())
    ik_solver = PyrokiRetarget(urdf_path, target_links)
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

    print(f"\nTesting {robot_name} (Stage 2 flow matching)")
    print(f"  Batch size: {batch_size}")
    print(f"  Split batch: {config.test.split_batch_size}")
    print(f"  ODE steps: {config.model.flow_matching_config.ode_steps}")

    for batch_id, batch in tqdm.tqdm(enumerate(dataloader)):
        transform_dict = {}
        data_count = 0
        predict_q_list = []
        initial_q_list = []
        object_pc_list = []

        while data_count != batch_size:
            split_num = min(batch_size - data_count, config.test.split_batch_size)
            initial_q = batch["initial_q"][data_count : data_count + split_num].to(device)
            object_pc = batch["object_pc"][data_count : data_count + split_num].to(device)
            robot_links_pc = batch["robot_links_pc"][data_count : data_count + split_num]

            split_batch = {
                "robot_name": batch["robot_name"],
                "object_name": batch["object_name"],
                "initial_q": initial_q,
                "object_pc": object_pc,
                "robot_links_pc": robot_links_pc,
            }
            data_count += split_num

            time_start = time.time()
            with torch.no_grad():
                all_step_poses_dict = model.inference(split_batch)

            ## IK process with pyroki
            clean_robot_pose = all_step_poses_dict[0]
            optim_transform = process_transform(hand.pk_chain, clean_robot_pose)
            initial_q_jnp = jnp.array(initial_q.cpu().numpy())
            target_pos_list = [optim_transform[name] for name in target_links]
            target_pos = torch.stack(target_pos_list, dim=1)

            target_pos_jnp = jnp.array(target_pos.detach().cpu().numpy())
            predict_q_jnp = batch_retarget(
                initial_q=initial_q_jnp,
                target_pos=target_pos_jnp,
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
            initial_q_list.append(initial_q)
            predict_q_list.append(predict_q)
            object_pc_list.append(object_pc)

            for diffuse_step, pred_robot_pose in all_step_poses_dict.items():
                if diffuse_step not in transform_dict:
                    transform_dict[diffuse_step] = []
                transform_dict[diffuse_step].append(pred_robot_pose)

        # Simulation, isaac subprocess
        all_predict_q = torch.cat(predict_q_list, dim=0)

        success, isaac_q = validate_isaac(
            batch["robot_name"],
            batch["object_name"],
            all_predict_q,
            gpu=config.test.gpu,
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
                        v if k not in transform_batch
                        else torch.cat((transform_batch[k], v), dim=0)
                    )
            transform_dict[diffuse_step] = transform_batch

        vis_info.append({
            "robot_name": batch["robot_name"],
            "object_name": batch["object_name"],
            "initial_q": torch.cat(initial_q_list, dim=0),
            "predict_q": torch.cat(predict_q_list, dim=0),
            "object_pc": torch.cat(object_pc_list, dim=0),
            "predict_transform": transform_dict,
            "success": success,
            "isaac_q": isaac_q,
        })

    os.makedirs(config.test.save_dir, exist_ok=True)
    torch.save(vis_info, os.path.join(config.test.save_dir, "vis.pt"))

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

        if total_grasp_num > 0:
            line = f"Grasp generation time: {total_inference_time / total_grasp_num} s.\n"
            print(line, end="")
            f.write(line)

    print(f"\nResults saved to {config.test.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/test_fm_stage2.yaml",
        help="config file",
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    test(config)
