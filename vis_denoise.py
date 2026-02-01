"""
Interactive denoising trajectory visualization.

Runs model inference (skipping Isaac evaluation) and visualizes per-link
point cloud pose changes across denoising steps using viser.

Supports all model variants via model_type config field:
  - original:  RobotGraph (diffusion + GraphDenoiser)
  - fm:        FlowMatchingRobotGraph
  - diff_v2:   RobotGraphV2 (diffusion + FlashAttention)
  - diff_v3:   RobotGraphV3 (diffusion + FlashAttention, no edge)
  - fm_v2:     FlowMatchingV2 (FM + FlashAttention)
  - diff_v3_ce: RobotGraphV3CE (cross-embodiment, supports no_object mode)

Usage:
    python vis_denoise.py --config config/vis_denoise.yaml
    python vis_denoise.py --load vis_cache.pt          # skip inference, load cached data
"""

import os
import sys
import time
import inspect
import colorsys
import argparse
import importlib

import torch
import tqdm
import trimesh
import numpy as np
import viser
from omegaconf import OmegaConf

from dataset.CMapDataset import create_dataloader
from utils.hand_model import create_hand_model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# ---- Model registry: model_type → (module_path, class_name) ----
MODEL_REGISTRY = {
    "original": ("model.tro_graph", "RobotGraph"),
    "fm": ("model.flow_matching_graph", "FlowMatchingRobotGraph"),
    "diff_v2": ("model.tro_graph_v2", "RobotGraphV2"),
    "diff_v3": ("model.tro_graph_v3", "RobotGraphV3"),
    "fm_v2": ("model.flow_matching_v2", "FlowMatchingV2"),
    "fm_v3": ("model.flow_matching_v3", "FlowMatchingV3"),
    "diff_v3_ce": ("model.tro_graph_v3_ce", "RobotGraphV3CE"),
}


def load_checkpoint(model, ckpt_path):
    """Load checkpoint, handling Lightning, model_state, and raw formats."""
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
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
    return model


def build_model(config, device):
    """Dynamically import and instantiate the model from config.model_type."""
    model_type = config.model_type
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    module_path, class_name = MODEL_REGISTRY[model_type]
    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)

    model_cfg = OmegaConf.to_container(config.model, resolve=True)

    # Filter config to only include params accepted by this model class,
    # so a single config can contain keys for multiple model types
    # (e.g. both flow_matching_config and diffusion_config).
    sig = inspect.signature(ModelClass.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if not has_var_keyword:
        model_cfg = {k: v for k, v in model_cfg.items() if k in valid_params}

    model = ModelClass(**model_cfg).to(device)
    load_checkpoint(model, config.vis.ckpt)
    model.eval()
    print(f"Loaded {class_name} from {config.vis.ckpt}")
    return model


def _generate_no_object_batches(config):
    """Generate batches for no-object inference (no CMapDataset needed).

    Returns a list of batch dicts with dummy object_pc (zeros).
    The model's _inference_no_object() will skip VQ-VAE encoding.
    """
    robot_name = config.vis.embodiment
    hand = create_hand_model(robot_name, device=torch.device("cpu"))
    batch_size = config.dataset.batch_size

    initial_q_list = []
    initial_se3_list = []
    robot_links_pc_list = []
    for _ in range(batch_size):
        q = hand.get_initial_q()
        _, se3 = hand.get_transformed_links_pc(q)
        initial_q_list.append(q)
        initial_se3_list.append(se3)
        robot_links_pc_list.append(hand.links_pc)

    batch = {
        "robot_name": robot_name,
        "object_name": "none",
        "initial_q": torch.stack(initial_q_list),
        "initial_se3": torch.stack(initial_se3_list),
        "object_pc": torch.zeros(batch_size, 512, 3),
        "robot_links_pc": robot_links_pc_list,
    }
    return [batch]


def _get_batch_grasp_count(info):
    """Get grasp count from vis_data entry, handling both object/no-object."""
    if info.get("object_pc") is not None:
        return info["object_pc"].shape[0]
    # No object: get B from transform_dict
    sample_step = next(iter(info["transform_dict"]))
    sample_link = next(iter(info["transform_dict"][sample_step]))
    return info["transform_dict"][sample_step][sample_link].shape[0]


def run_inference(config, model, device):
    """Run inference on all test batches, collect per-step transforms."""
    inference_mode = OmegaConf.select(
        config, "model.inference_config.inference_mode", default="unconditioned"
    )
    no_object = inference_mode == "no_object"

    if no_object:
        batches = _generate_no_object_batches(config)
    else:
        batches = create_dataloader(config.dataset, is_train=False)

    batch_size = config.dataset.batch_size
    split_batch_size = config.vis.split_batch_size

    vis_data = []

    for batch_id, batch in tqdm.tqdm(enumerate(batches), desc="Inference"):
        transform_dict = {}
        data_count = 0
        object_pc_list = []

        while data_count != batch_size:
            split_num = min(batch_size - data_count, split_batch_size)
            object_pc = batch["object_pc"][data_count : data_count + split_num].to(
                device
            )
            robot_links_pc = batch["robot_links_pc"][
                data_count : data_count + split_num
            ]
            initial_q = batch["initial_q"][data_count : data_count + split_num].to(
                device
            )
            initial_se3 = batch["initial_se3"][data_count : data_count + split_num].to(
                device
            )

            split_batch = {
                "robot_name": batch["robot_name"],
                "object_name": batch["object_name"],
                "initial_q": initial_q,
                "initial_se3": initial_se3,
                "object_pc": object_pc,
                "robot_links_pc": robot_links_pc,
            }
            data_count += split_num

            with torch.no_grad():
                all_step_poses_dict = model.inference(split_batch)

            # Merge transforms from this split
            for step, pred_robot_pose in all_step_poses_dict.items():
                if step not in transform_dict:
                    transform_dict[step] = []
                transform_dict[step].append(pred_robot_pose)

            if not no_object:
                object_pc_list.append(object_pc.cpu())

        # Concatenate splits for each step
        for step, transform_list in transform_dict.items():
            merged = {}
            for transform in transform_list:
                for k, v in transform.items():
                    merged[k] = (
                        v.cpu()
                        if k not in merged
                        else torch.cat((merged[k], v.cpu()), dim=0)
                    )
            transform_dict[step] = merged

        vis_data.append(
            {
                "robot_name": batch["robot_name"],
                "object_name": batch["object_name"],
                "object_pc": torch.cat(object_pc_list, dim=0) if object_pc_list else None,
                "transform_dict": transform_dict,
            }
        )

    return vis_data


def generate_link_colors(link_names):
    """Generate distinct HSV-spaced colors for each link."""
    colors = {}
    n = len(link_names)
    for i, name in enumerate(link_names):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors[name] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


def launch_viewer(vis_data, config):
    """Launch interactive viser viewer."""
    port = config.vis.get("port", 8080)
    server = viser.ViserServer(host="0.0.0.0", port=port)
    print(f"Viser server running at http://localhost:{port}")

    # Compute total grasp count and per-batch offsets
    grasp_offsets = []
    total_grasps = 0
    for info in vis_data:
        grasp_offsets.append(total_grasps)
        total_grasps += _get_batch_grasp_count(info)

    # Get sorted denoising steps (noise → clean)
    sample_steps = sorted(vis_data[0]["transform_dict"].keys(), reverse=True)
    num_steps = len(sample_steps)

    # Cache hand models
    hand_cache = {}

    def get_hand(robot_name):
        if robot_name not in hand_cache:
            hand_cache[robot_name] = create_hand_model(
                robot_name, device=torch.device("cpu")
            )
        return hand_cache[robot_name]

    # ---- GUI controls ----
    grasp_slider = server.gui.add_slider(
        label="grasp_idx",
        min=0,
        max=max(total_grasps - 1, 0),
        step=1,
        initial_value=0,
    )
    step_slider = server.gui.add_slider(
        label="denoise_step",
        min=0,
        max=max(num_steps - 1, 0),
        step=1,
        initial_value=num_steps - 1,
    )
    show_mesh_cb = server.gui.add_checkbox(
        label="show_object_mesh",
        initial_value=True,
    )
    show_pc_cb = server.gui.add_checkbox(
        label="show_object_pc",
        initial_value=False,
    )

    def on_update(_=None):
        grasp_idx = int(grasp_slider.value)
        step_idx = int(step_slider.value)

        # Find which batch and sample index
        info = None
        local_idx = grasp_idx
        for i, data in enumerate(vis_data):
            n = _get_batch_grasp_count(data)
            if local_idx < n:
                info = data
                break
            local_idx -= n

        if info is None:
            return

        robot_name = info["robot_name"]
        object_name = info["object_name"]
        step_key = sample_steps[step_idx]
        has_object = info.get("object_pc") is not None

        print(f"{robot_name} | {object_name} | sample {local_idx} | step {step_key}")

        # ---- Object mesh ----
        if has_object:
            obj_parts = object_name.split("+")
            obj_path = os.path.join(
                ROOT_DIR,
                f"data/data_urdf/object/{obj_parts[0]}/{obj_parts[1]}/{obj_parts[1]}.stl",
            )
            if os.path.exists(obj_path) and show_mesh_cb.value:
                obj_mesh = trimesh.load_mesh(obj_path)
                server.scene.add_mesh_simple(
                    "object/mesh",
                    obj_mesh.vertices,
                    obj_mesh.faces,
                    color=(239, 132, 167),
                    opacity=0.6,
                )
            else:
                server.scene.add_mesh_simple(
                    "object/mesh",
                    np.zeros((3, 3), dtype=np.float32),
                    np.zeros((1, 3), dtype=np.int32),
                    visible=False,
                )
        else:
            # No object: clear mesh and PC placeholders
            server.scene.add_mesh_simple(
                "object/mesh",
                np.zeros((3, 3), dtype=np.float32),
                np.zeros((1, 3), dtype=np.int32),
                visible=False,
            )

        # ---- Object point cloud ----
        if has_object:
            obj_pc = info["object_pc"][local_idx].numpy()
            server.scene.add_point_cloud(
                "object/pc",
                obj_pc,
                point_size=0.002,
                point_shape="circle",
                colors=(239, 132, 167),
                visible=show_pc_cb.value,
            )
        else:
            server.scene.add_point_cloud(
                "object/pc",
                np.zeros((1, 3), dtype=np.float32),
                point_size=0.001,
                point_shape="circle",
                colors=(0, 0, 0),
                visible=False,
            )

        # ---- Robot per-link point clouds ----
        hand = get_hand(robot_name)
        transform = info["transform_dict"][step_key]
        link_names = list(hand.links_pc.keys())
        link_colors = generate_link_colors(link_names)

        for link_name, link_pc in hand.links_pc.items():
            if link_name not in transform:
                continue
            se3 = transform[link_name][local_idx]  # (4, 4)
            n_pts = link_pc.shape[0]
            homo = torch.cat(
                [link_pc, torch.ones(n_pts, 1, device=link_pc.device)], dim=1
            )
            positioned = (homo @ se3.T)[:, :3]

            server.scene.add_point_cloud(
                f"robot/{link_name}",
                positioned.numpy(),
                point_size=0.003,
                point_shape="circle",
                colors=link_colors[link_name],
            )

    # Register callbacks
    grasp_slider.on_update(on_update)
    step_slider.on_update(on_update)
    show_mesh_cb.on_update(on_update)
    show_pc_cb.on_update(on_update)

    # Initial render
    on_update()

    print("Viewer ready. Use sliders to browse grasps and denoising steps.")
    while True:
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Denoising trajectory visualization")
    parser.add_argument(
        "--config",
        type=str,
        default="config/vis_denoise.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Load cached vis_data from .pt file (skip inference)",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if args.load and os.path.exists(args.load):
        print(f"Loading cached data from {args.load}...")
        vis_data = torch.load(args.load, weights_only=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Building model...")
        model = build_model(config, device)

        print("Running inference...")
        vis_data = run_inference(config, model, device)

        # Optionally save for reuse
        save_path = config.vis.get("save_path")
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(vis_data, save_path)
            print(f"Saved vis_data to {save_path}")

    print(f"Total objects: {len(vis_data)}")
    total = sum(_get_batch_grasp_count(d) for d in vis_data)
    print(f"Total grasps: {total}")
    steps = sorted(vis_data[0]["transform_dict"].keys())
    print(f"Denoising steps: {len(steps)} ({max(steps)} -> {min(steps)})")

    launch_viewer(vis_data, config)


if __name__ == "__main__":
    main()
