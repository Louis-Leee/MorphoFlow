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
    """Generate batches for no-object inference.

    Uses a random reference object's point cloud for denormalization
    (centroid/scale mapping), but the model does NOT see object features
    (V_O=zeros, skip_or=True). Each sample gets a different random
    reference object for spatial diversity.
    """
    robot_name = config.vis.embodiment
    hand = create_hand_model(robot_name, device=torch.device("cpu"))
    batch_size = config.dataset.batch_size

    # Load available object meshes for reference normalization
    import json
    split_path = os.path.join(ROOT_DIR, "data/CMapDataset_filtered/split_train_validate_objects.json")
    with open(split_path) as f:
        dataset_split = json.load(f)
    ref_objects = dataset_split.get("validate", dataset_split.get("train", []))

    initial_q_list = []
    initial_se3_list = []
    robot_links_pc_list = []
    object_pc_list = []
    for _ in range(batch_size):
        q = hand.get_initial_q()
        _, se3 = hand.get_transformed_links_pc(q)
        initial_q_list.append(q)
        initial_se3_list.append(se3)
        robot_links_pc_list.append(hand.links_pc)

        # Sample a random reference object for normalization frame
        import random as _random
        ref_obj = _random.choice(ref_objects)
        parts = ref_obj.split("+")
        mesh_path = os.path.join(
            ROOT_DIR, f"data/data_urdf/object/{parts[0]}/{parts[1]}/{parts[1]}.stl"
        )
        mesh = trimesh.load_mesh(mesh_path)
        pts, _ = mesh.sample(65536, return_index=True)
        indices = np.random.permutation(65536)[:512]
        ref_pc = torch.tensor(pts[indices], dtype=torch.float32)
        ref_pc = ref_pc + torch.randn_like(ref_pc) * 0.002
        object_pc_list.append(ref_pc)

    batch = {
        "robot_name": robot_name,
        "object_name": "none",
        "initial_q": torch.stack(initial_q_list),
        "initial_se3": torch.stack(initial_se3_list),
        "object_pc": torch.stack(object_pc_list),
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
        initial_q_list = []

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

            object_pc_list.append(object_pc.cpu())
            initial_q_list.append(initial_q.cpu())

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
                "object_pc": torch.cat(object_pc_list, dim=0),
                "initial_q": torch.cat(initial_q_list, dim=0),
                "transform_dict": transform_dict,
            }
        )

    return vis_data


# ── LeapHand tip mapping (use fixed joints instead of revolute for IK) ──
# NOTE: thumb_fingertip is NOT included because extra_thumb_tip_head has Z offset
# in the opposite direction (-0.015 vs +0.015 for other fingers), which causes
# the thumb to curl inward even more. Keep thumb_fingertip as IK target.
LEAPHAND_TIP_MAPPING = {
    'fingertip': 'extra_index_tip_head',
    'fingertip_2': 'extra_middle_tip_head',
    'fingertip_3': 'extra_ring_tip_head',
    # thumb_fingertip stays as-is (don't use extra_thumb_tip_head)
}

# ── LeapHand per-link IK weights ──
# 实验: 疯狂提高 palm 权重，让 IK 优先对齐手掌
LEAPHAND_LINK_WEIGHTS = {
    'palm_lower': 3.0,            # 极高 palm 权重
    'extra_ring_tip_head': 0.5,    # 大幅降低无名指
    'extra_middle_tip_head': 0.7,  # 大幅降低中指
    'extra_index_tip_head': 0.8,   # 大幅降低食指
    'thumb_fingertip': 3.0,        # 降低拇指
}

# ── LeapHand locked joints (keep at initial value during IK) ──
# Set to empty to disable joint locking
LEAPHAND_LOCKED_JOINTS = []

# ── IK experiment modes (select via config: vis.ik.mode) ──
IK_MODE_DEFAULTS = {
    "baseline": {"thumb_extra": False, "palm_anchor": False},
    "thumb_extra": {"thumb_extra": True, "palm_anchor": False},
    "palm_anchor": {"thumb_extra": False, "palm_anchor": True},
    "combo": {"thumb_extra": True, "palm_anchor": True},
}

# ── Fingertip joint extraction for pure-rotation joints ─────────────────
FINGERTIP_JOINTS = {
    'leaphand': {
        ('dip', 'fingertip'): 9,
        ('dip_2', 'fingertip_2'): 13,
        ('dip_3', 'fingertip_3'): 17,
        ('thumb_dip', 'thumb_fingertip'): 21,
    },
}


def extract_fingertip_joints(predict_q, transform_dict, robot_name):
    """Extract fingertip joint angles from diffusion SE3 rotation."""
    if robot_name not in FINGERTIP_JOINTS:
        return predict_q
    for (parent, child), q_idx in FINGERTIP_JOINTS[robot_name].items():
        if parent not in transform_dict or child not in transform_dict:
            continue
        R_parent = transform_dict[parent][:, :3, :3]
        R_child = transform_dict[child][:, :3, :3]
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


def _resolve_ik_cfg(config):
    ik_cfg = OmegaConf.select(config, "vis.ik", default=None)
    if ik_cfg is None:
        ik_cfg = {}
    elif not isinstance(ik_cfg, dict):
        ik_cfg = OmegaConf.to_container(ik_cfg, resolve=True)

    mode = ik_cfg.get("mode", "baseline")
    defaults = IK_MODE_DEFAULTS.get(mode, IK_MODE_DEFAULTS["baseline"])

    return {
        "mode": mode,
        "thumb_use_extra": ik_cfg.get("thumb_tip_use_extra", defaults["thumb_extra"]),
        "add_base_anchor": ik_cfg.get("add_base_anchor", defaults["palm_anchor"]),
        "weight_overrides": dict(ik_cfg.get("weight_overrides", {})),
        "base_anchor_weight": ik_cfg.get("base_anchor_weight", 1.0),
        "report_path": ik_cfg.get("report_path"),
    }


def _derive_base_pos_from_palm(pk_chain, palm_name, palm_world):
    frame = pk_chain.find_frame(palm_name)
    base_to_palm = frame.joint.offset.get_matrix()[0].to(palm_world.device)
    palm_to_base = torch.linalg.inv(base_to_palm)
    base_world = palm_world @ palm_to_base
    return base_world[:, :3, 3]


def _fk_link_positions(hand, q, link_names):
    status = hand.pk_chain.forward_kinematics(q)
    positions = []
    for name in link_names:
        positions.append(status[name].get_matrix()[:, :3, 3])
    return torch.stack(positions, dim=1)


def _write_ik_report(report_path, mode, error_stats):
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# IK Alignment Report")
    lines.append(f"- mode: {mode}")
    lines.append(f"- timestamp: {timestamp}")
    lines.append("")

    for robot_name in sorted(error_stats.keys()):
        robot_stats = error_stats[robot_name]
        lines.append(f"## {robot_name}")
        lines.append(f"- overall_mean_mm: {robot_stats['overall_mean_mm']:.3f}")
        lines.append("- worst_links_mm:")
        for name, val in robot_stats["worst_links_mm"]:
            lines.append(f"  - {name}: {val:.3f}")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_pyroki_ik(vis_data, config):
    """Run pyroki IK on step-0 (final denoised) transforms to get predict_q."""
    import jax
    import json
    import jax.numpy as jnp
    from utils.pyroki_ik import PyrokiRetarget
    from utils.optimization import process_transform
    from utils.palm_centric import get_palm_link_name

    ik_cfg = _resolve_ik_cfg(config)

    with open(os.path.join(ROOT_DIR, "data/data_urdf/robot/urdf_assets_meta.json")) as f:
        urdf_meta = json.load(f)

    error_sums = {}
    error_counts = {}

    for info in vis_data:
        if "initial_q" not in info:
            print(f"  Skipping IK for {info['robot_name']}: no initial_q in cached data")
            continue

        robot_name = info["robot_name"]
        hand = create_hand_model(robot_name, device=torch.device("cpu"))
        urdf_path = urdf_meta["urdf_path"][robot_name]
        target_links = list(hand.links_pc.keys())

        # LeapHand: 用 fixed joint 的 extra tip 链接替代 revolute joint 的 fingertip 链接
        # NOTE: thumb only mapped when ik_cfg.thumb_use_extra=True
        target_links_ik = target_links.copy()
        link_weights = None
        locked_joints = None
        if robot_name == 'leaphand':
            tip_mapping = dict(LEAPHAND_TIP_MAPPING)
            if ik_cfg["thumb_use_extra"]:
                tip_mapping["thumb_fingertip"] = "extra_thumb_tip_head"
            for old_tip, new_tip in tip_mapping.items():
                if old_tip in target_links_ik:
                    idx = target_links_ik.index(old_tip)
                    target_links_ik[idx] = new_tip
            if ik_cfg["add_base_anchor"] and "base" not in target_links_ik:
                target_links_ik.append("base")
            # Build per-link weights array
            link_weights = [LEAPHAND_LINK_WEIGHTS.get(name, 1.0) for name in target_links_ik]
            for i, name in enumerate(target_links_ik):
                if name == "base":
                    link_weights[i] = ik_cfg["base_anchor_weight"]
            for i, name in enumerate(target_links_ik):
                if name in ik_cfg["weight_overrides"]:
                    link_weights[i] = ik_cfg["weight_overrides"][name]
            # Locked joints: keep at initial value during IK
            locked_joints = LEAPHAND_LOCKED_JOINTS
            if ik_cfg["thumb_use_extra"]:
                print("  LeapHand: Using extra_*_tip_head for fingers + thumb")
            else:
                print("  LeapHand: Using extra_*_tip_head for fingers, thumb_fingertip unchanged")
            if ik_cfg["add_base_anchor"]:
                print("  LeapHand: Added base anchor target for palm orientation")
            if locked_joints:
                print(f"  LeapHand: Locked joints {locked_joints} (PIP joints)")
            print(f"  LeapHand: Link weights: palm_lower={LEAPHAND_LINK_WEIGHTS.get('palm_lower', 1.0)}")
        elif ik_cfg["weight_overrides"]:
            link_weights = [ik_cfg["weight_overrides"].get(name, 1.0) for name in target_links_ik]

        ik_solver = PyrokiRetarget(
            urdf_path, target_links_ik,
            hand_joint_names=hand.get_joint_orders(),
            link_weights=link_weights,
            locked_joint_indices=locked_joints,
        )
        batch_retarget = jax.jit(ik_solver.solve_retarget)

        # Get step-0 (final denoised) transforms
        step_0 = info["transform_dict"][0]
        optim_transform = process_transform(hand.pk_chain, step_0)
        target_pos_list = []
        palm_name = get_palm_link_name(robot_name)
        for name in target_links_ik:
            if name in optim_transform:
                target_pos_list.append(optim_transform[name])
            elif name == "base" and ik_cfg["add_base_anchor"]:
                palm_world = step_0[palm_name]
                target_pos_list.append(_derive_base_pos_from_palm(hand.pk_chain, palm_name, palm_world))
            else:
                raise KeyError(f"Missing target link '{name}' in transform_dict for {robot_name}")
        target_pos = torch.stack(target_pos_list, dim=1)  # (B, num_links, 3)

        initial_q = info["initial_q"]
        initial_q_jnp = jnp.array(initial_q.numpy())
        target_pos_jnp = jnp.array(target_pos.detach().numpy())

        predict_q_jnp = batch_retarget(
            initial_q=initial_q_jnp, target_pos=target_pos_jnp,
        )
        jax.block_until_ready(predict_q_jnp)
        predict_q = torch.from_numpy(np.array(predict_q_jnp)).float()

        # Extract fingertip joint angles from diffusion rotation
        predict_q = extract_fingertip_joints(predict_q, step_0, robot_name)

        info["predict_q"] = predict_q
        print(f"  {robot_name}: pyroki IK done, predict_q shape {info['predict_q'].shape}")

        # ---- Alignment error report (FK vs diffusion target positions) ----
        pred_pos = _fk_link_positions(hand, predict_q, target_links_ik)
        errors = (pred_pos - target_pos).norm(dim=-1)  # (B, L)
        if robot_name not in error_sums:
            error_sums[robot_name] = {}
            error_counts[robot_name] = {}
        for idx, name in enumerate(target_links_ik):
            err_sum = errors[:, idx].sum().item()
            err_count = errors[:, idx].numel()
            error_sums[robot_name][name] = error_sums[robot_name].get(name, 0.0) + err_sum
            error_counts[robot_name][name] = error_counts[robot_name].get(name, 0) + err_count

    # ---- Write report if requested ----
    if ik_cfg["report_path"]:
        report = {}
        for robot_name in error_sums:
            sums = error_sums[robot_name]
            counts = error_counts[robot_name]
            link_means = {k: (sums[k] / max(counts[k], 1)) for k in sums}
            overall_mean = sum(sums.values()) / max(sum(counts.values()), 1)
            worst_links = sorted(link_means.items(), key=lambda x: x[1], reverse=True)[:8]
            report[robot_name] = {
                "overall_mean_mm": overall_mean * 1000.0,
                "worst_links_mm": [(k, v * 1000.0) for k, v in worst_links],
            }
        _write_ik_report(ik_cfg["report_path"], ik_cfg["mode"], report)
        print(f"IK report saved to {ik_cfg['report_path']}")


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
    show_denoised_cb = server.gui.add_checkbox(
        label="show_denoised_links",
        initial_value=True,
    )
    show_ik_cb = server.gui.add_checkbox(
        label="show_ik_links",
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
        if has_object and "+" in object_name:
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

        # ---- Robot per-link point clouds (denoised) ----
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
                f"robot_denoised/{link_name}",
                positioned.numpy(),
                point_size=0.003,
                point_shape="circle",
                colors=link_colors[link_name],
                visible=show_denoised_cb.value,
            )

        # ---- Robot mesh (IK, step 0 only) ----
        has_ik = info.get("predict_q") is not None
        if has_ik and step_key == 0 and show_ik_cb.value:
            predict_q = info["predict_q"][local_idx]
            robot_trimesh = hand.get_trimesh_q(predict_q)["visual"]
            server.scene.add_mesh_simple(
                "robot_ik",
                robot_trimesh.vertices,
                robot_trimesh.faces,
                color=(100, 255, 100),
                opacity=0.8,
            )
        else:
            server.scene.add_mesh_simple(
                "robot_ik",
                np.zeros((3, 3), dtype=np.float32),
                np.zeros((1, 3), dtype=np.int32),
                visible=False,
            )

    # Register callbacks
    grasp_slider.on_update(on_update)
    step_slider.on_update(on_update)
    show_mesh_cb.on_update(on_update)
    show_pc_cb.on_update(on_update)
    show_denoised_cb.on_update(on_update)
    show_ik_cb.on_update(on_update)

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
    force_recompute = OmegaConf.select(config, "vis.ik.force_recompute", default=False)

    if args.load and os.path.exists(args.load):
        print(f"Loading cached data from {args.load}...")
        vis_data = torch.load(args.load, weights_only=False)
        # Run IK if predict_q not already cached
        if vis_data and (force_recompute or "predict_q" not in vis_data[0]):
            if force_recompute:
                for info in vis_data:
                    info.pop("predict_q", None)
            print("Running pyroki IK on loaded data...")
            run_pyroki_ik(vis_data, config)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Building model...")
        model = build_model(config, device)

        print("Running inference...")
        vis_data = run_inference(config, model, device)

        print("Running pyroki IK...")
        run_pyroki_ik(vis_data, config)

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
