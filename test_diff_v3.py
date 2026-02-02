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
        ik_solver = PyrokiRetarget(urdf_path, target_links, hand_joint_names=hand.get_joint_orders())
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
