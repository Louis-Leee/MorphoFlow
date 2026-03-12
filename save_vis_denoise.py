"""
Batch offline rendering of grasp visualizations (object mesh + IK hand mesh).

Renders each grasp as a PNG image using Open3D's OffscreenRenderer (headless EGL).
Reuses inference/IK pipeline from vis_denoise.py — no code duplication.

Usage:
    python save_vis_denoise.py --config config/vis_denoise_v4_ce_vae.yaml
    python save_vis_denoise.py --config config/vis_denoise_v4_ce_vae.yaml --load outputs/vis_cache_v4_vae_obj_vae50.pt
"""

import os
import sys
import argparse

import torch
import numpy as np
import trimesh
import open3d as o3d
from omegaconf import OmegaConf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from vis_denoise import (
    build_model,
    run_inference,
    run_pyroki_ik,
    _get_batch_grasp_count,
)
from utils.hand_model import create_hand_model

# ── Color palette for hand mesh ──
COLOR_PALETTE = {
    "light_green": [0.75, 0.95, 0.75],
    "light_blue": [0.80, 0.90, 0.95],
    "light_red": [1.00, 0.82, 0.85],
    "light_gray": [0.93, 0.93, 0.93],
    "light_purple": [0.88, 0.80, 0.90],
    "light_orange": [1.00, 0.88, 0.75],
}

# Object mesh: vivid coral red, near-opaque
OBJECT_COLOR = [1.0, 0.45, 0.50, 0.95]

# Hand opacity — see-through but still visible
HAND_OPACITY = 0.65


def trimesh_to_open3d(tm_mesh):
    """Convert a trimesh.Trimesh to open3d.geometry.TriangleMesh."""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(tm_mesh.vertices, dtype=np.float64)
    )
    o3d_mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(tm_mesh.faces, dtype=np.int32)
    )
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def render_grasp(obj_mesh_o3d, hand_mesh_o3d, hand_color_rgb, resolution=(1024, 768)):
    """Render object + hand mesh to a numpy image array.

    Args:
        obj_mesh_o3d: Open3D TriangleMesh for the object.
        hand_mesh_o3d: Open3D TriangleMesh for the hand.
        hand_color_rgb: [r, g, b] floats in [0, 1].
        resolution: (width, height) tuple.

    Returns:
        np.ndarray of shape (H, W, 3) uint8.
    """
    w, h = resolution
    renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    # Lighting
    renderer.scene.scene.set_sun_light([0.5, -1.0, -0.5], [1.0, 1.0, 1.0], 75000)
    renderer.scene.scene.enable_sun_light(True)

    # Object mesh (semi-transparent)
    obj_mat = o3d.visualization.rendering.MaterialRecord()
    obj_mat.shader = "defaultLitTransparency"
    obj_mat.base_color = OBJECT_COLOR
    renderer.scene.add_geometry("object", obj_mesh_o3d, obj_mat)

    # Hand mesh (semi-transparent, matching viser opacity=0.8)
    hand_mat = o3d.visualization.rendering.MaterialRecord()
    hand_mat.shader = "defaultLitTransparency"
    hand_mat.base_color = hand_color_rgb + [HAND_OPACITY]
    renderer.scene.add_geometry("hand", hand_mesh_o3d, hand_mat)

    # Auto camera: look at combined bounding box
    combined = obj_mesh_o3d + hand_mesh_o3d
    bb = combined.get_axis_aligned_bounding_box()
    center = bb.get_center().astype(np.float32)
    extent = max(bb.get_extent())

    # Camera position: slightly above and to the side for a good 3/4 view
    eye = center + np.array(
        [extent * 1.8, extent * 1.8, extent * 1.2], dtype=np.float32
    )
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    renderer.setup_camera(45.0, center, eye, up)

    img = renderer.render_to_image()
    return np.asarray(img)


def load_object_mesh(object_name):
    """Load object STL mesh from dataset, return as Open3D TriangleMesh.

    Args:
        object_name: Format "category+object_id" (e.g., "contactdb+airplane").

    Returns:
        Open3D TriangleMesh or None if not found.
    """
    if "+" not in object_name:
        return None

    parts = object_name.split("+")
    stl_path = os.path.join(
        ROOT_DIR, f"data/data_urdf/object/{parts[0]}/{parts[1]}/{parts[1]}.stl"
    )
    if not os.path.exists(stl_path):
        print(f"  Warning: object mesh not found: {stl_path}")
        return None

    tm_mesh = trimesh.load_mesh(stl_path)
    return trimesh_to_open3d(tm_mesh)


def save_images(vis_data, config):
    """Render and save grasp images for all objects in vis_data."""
    image_dir = config.vis.get("image_dir")
    if not image_dir:
        print("Error: vis.image_dir not set in config")
        return

    hand_color_name = config.vis.get("hand_color", "light_gray")
    hand_color_rgb = COLOR_PALETTE.get(hand_color_name)
    if hand_color_rgb is None:
        print(
            f"Warning: unknown color '{hand_color_name}', "
            f"available: {list(COLOR_PALETTE.keys())}. Using light_gray."
        )
        hand_color_rgb = COLOR_PALETTE["light_gray"]

    num_images = config.vis.get("num_images", 10)
    total_saved = 0

    for info in vis_data:
        robot_name = info["robot_name"]
        object_name = info["object_name"]

        if info.get("predict_q") is None:
            print(f"  Skipping {robot_name}/{object_name}: no IK results (predict_q)")
            continue

        # Load object mesh
        obj_mesh = load_object_mesh(object_name)
        if obj_mesh is None:
            continue

        # Create hand model
        hand = create_hand_model(robot_name, device=torch.device("cpu"))

        # Select grasp indices (evenly spaced)
        batch_size = _get_batch_grasp_count(info)
        if num_images >= batch_size:
            grasp_indices = list(range(batch_size))
        else:
            step = batch_size // num_images
            grasp_indices = [i * step for i in range(num_images)]

        # Sanitize object name for filesystem
        obj_dir_name = object_name.replace("+", "_")
        save_dir = os.path.join(image_dir, robot_name, obj_dir_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"  Rendering {robot_name}/{obj_dir_name}: {len(grasp_indices)} grasps")

        for gi, grasp_idx in enumerate(grasp_indices):
            predict_q = info["predict_q"][grasp_idx]
            hand_result = hand.get_trimesh_q(predict_q)
            hand_tm = hand_result["visual"]
            hand_mesh = trimesh_to_open3d(hand_tm)

            img = render_grasp(obj_mesh, hand_mesh, hand_color_rgb)

            save_path = os.path.join(save_dir, f"grasp_{gi:02d}.png")
            o3d.io.write_image(save_path, o3d.geometry.Image(img))
            total_saved += 1

    print(f"\nDone! Saved {total_saved} images to {image_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch render grasp images (object mesh + IK hand mesh)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/vis_denoise_v4_ce_vae.yaml",
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
    force_recompute = OmegaConf.select(
        config, "vis.ik.force_recompute", default=False
    )

    if args.load and os.path.exists(args.load):
        print(f"Loading cached data from {args.load}...")
        vis_data = torch.load(args.load, weights_only=False)
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

        # Save cache for reuse
        save_path = config.vis.get("save_path")
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(vis_data, save_path)
            print(f"Saved vis_data to {save_path}")

    print(f"\nTotal objects: {len(vis_data)}")
    total = sum(_get_batch_grasp_count(d) for d in vis_data)
    print(f"Total grasps: {total}")

    print("\nRendering images...")
    save_images(vis_data, config)


if __name__ == "__main__":
    main()
