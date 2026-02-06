"""
SE3 Flow Matching Scheduler with proper SO(3) geodesic interpolation.

Based on the user's SE3FlowMatchEulerContinuousScheduler implementation.
Key features:
- Translation: standard linear interpolation
- Rotation: SO(3) geodesic interpolation via log/exp map
- Supports multiple t sampling strategies: linear, logit_normal, u_shape
- Euler step integration on SE(3) manifold
"""

import math
import torch
import pytorch3d.transforms as p3dt


class SE3FlowScheduler:
    """SE3 Flow Matching Scheduler with proper SO(3) geodesic interpolation."""

    def __init__(
        self,
        sigma_min: float = 1e-5,
        t_schedule: str = "linear",
        t_schedule_kwargs: dict = None,
        shift_t: float = 1.0,
    ):
        """
        Args:
            sigma_min: Minimum noise level (prevents division by zero)
            t_schedule: Sampling strategy for t ("linear", "logit_normal", "u_shape")
            t_schedule_kwargs: Additional kwargs for t_schedule
            shift_t: Timestep shifting for inference (1.0 = no shift)
        """
        self.sigma_min = sigma_min
        self.t_schedule = t_schedule
        self.t_schedule_kwargs = t_schedule_kwargs or {}
        self.shift_t = shift_t

    def sample_t(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        Sample timestep t ∈ [0, 1].

        Args:
            batch_size: Number of samples
            device: Device to create tensor on

        Returns:
            t: [batch_size,] tensor of timesteps
        """
        if self.t_schedule == "linear":
            t = torch.rand(batch_size, device=device)
        elif self.t_schedule == "logit_normal":
            mean = self.t_schedule_kwargs.get("mean", 0.0)
            std = self.t_schedule_kwargs.get("std", 1.0)
            t = torch.sigmoid(torch.randn(batch_size, device=device) * std + mean)
        elif self.t_schedule == "u_shape":
            t = torch.rand(batch_size, device=device) * 2 - 1
            a = self.t_schedule_kwargs.get("a", 4.0)
            t = torch.asinh(t * math.sinh(a)) / a
            t = (t + 1) / 2
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule}")
        return t

    def _rot_vec_to_matrix(self, rot_vec: torch.Tensor) -> torch.Tensor:
        """Convert rotation vector to rotation matrix using SO(3) exp map."""
        # rot_vec: [..., 3] -> matrix: [..., 3, 3]
        original_shape = rot_vec.shape[:-1]
        rot_vec_flat = rot_vec.reshape(-1, 3)
        R = p3dt.so3_exp_map(rot_vec_flat)  # [N, 3, 3]
        return R.reshape(*original_shape, 3, 3)

    def _matrix_to_rot_vec(self, R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to rotation vector using SO(3) log map."""
        # R: [..., 3, 3] -> rot_vec: [..., 3]
        original_shape = R.shape[:-2]
        R_flat = R.reshape(-1, 3, 3)
        rot_vec = p3dt.so3_log_map(R_flat)  # [N, 3]
        return rot_vec.reshape(*original_shape, 3)

    def scale_noise_se3(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply flow matching interpolation on SE(3).

        x_t = interpolate(x_0, noise, t)
        - Translation: linear interpolation
        - Rotation: SO(3) geodesic interpolation

        Args:
            x_0: Clean pose [B, 6] or [B, N, 6] (trans + rot_vec)
            t: Timestep [B,] in [0, 1]
            noise: Random noise (if None, sample from appropriate distributions)

        Returns:
            x_t: Noisy pose, same shape as x_0
            noise: The noise used (for velocity computation)
        """
        device = x_0.device
        dtype = x_0.dtype

        # Handle different input shapes
        if x_0.dim() == 2:
            # [B, 6]
            B = x_0.shape[0]
            x_0_trans = x_0[:, :3]
            x_0_rot_vec = x_0[:, 3:]
            t_bc = t.view(-1, 1)
        elif x_0.dim() == 3:
            # [B, N, 6]
            B, N, _ = x_0.shape
            x_0_trans = x_0[:, :, :3]
            x_0_rot_vec = x_0[:, :, 3:]
            t_bc = t.view(-1, 1, 1)
        else:
            raise ValueError(f"x_0 must be 2D or 3D, got {x_0.dim()}D")

        # Generate noise if not provided
        if noise is None:
            noise_trans = torch.randn_like(x_0_trans)
            # For rotation, sample random rotation matrices
            if x_0.dim() == 2:
                noise_rot_quat = p3dt.random_quaternions(B, device=device, dtype=dtype)
            else:
                noise_rot_quat = p3dt.random_quaternions(B * N, device=device, dtype=dtype)
                noise_rot_quat = noise_rot_quat.reshape(B, N, 4)
            noise_rot_mat = p3dt.quaternion_to_matrix(noise_rot_quat)
            noise_rot_vec = self._matrix_to_rot_vec(noise_rot_mat)
            noise = torch.cat([noise_trans, noise_rot_vec], dim=-1)
        else:
            if x_0.dim() == 2:
                noise_trans = noise[:, :3]
                noise_rot_vec = noise[:, 3:]
            else:
                noise_trans = noise[:, :, :3]
                noise_rot_vec = noise[:, :, 3:]

        # ---- Translation: linear interpolation ----
        sigma_coef = self.sigma_min + (1 - self.sigma_min) * t_bc
        x_t_trans = (1 - t_bc) * x_0_trans + sigma_coef * noise_trans

        # ---- Rotation: SO(3) geodesic interpolation ----
        # Convert to matrices
        x_0_rot_mat = self._rot_vec_to_matrix(x_0_rot_vec)
        noise_rot_mat = self._rot_vec_to_matrix(noise_rot_vec)

        # Relative rotation: R_rel = R_noise @ R_0^T
        if x_0.dim() == 2:
            rot_rel = torch.bmm(noise_rot_mat, x_0_rot_mat.transpose(-2, -1))
        else:
            # [B, N, 3, 3]
            rot_rel = torch.matmul(noise_rot_mat, x_0_rot_mat.transpose(-2, -1))

        # Map to tangent space (log map)
        rot_rel_log = self._matrix_to_rot_vec(rot_rel)

        # Scale in tangent space
        # t_bc is [B, 1] for 2D or [B, 1, 1] for 3D, broadcasts with rot_rel_log
        rot_rel_log_scaled = (self.sigma_min + (1 - self.sigma_min) * t_bc) * rot_rel_log

        # Map back to SO(3)
        scaled_rel_rot = self._rot_vec_to_matrix(rot_rel_log_scaled)

        # Apply interpolated rotation: R_t = R_rel_scaled @ R_0
        if x_0.dim() == 2:
            x_t_rot_mat = torch.bmm(scaled_rel_rot, x_0_rot_mat)
        else:
            x_t_rot_mat = torch.matmul(scaled_rel_rot, x_0_rot_mat)

        x_t_rot_vec = self._matrix_to_rot_vec(x_t_rot_mat)

        # Combine
        x_t = torch.cat([x_t_trans, x_t_rot_vec], dim=-1)

        return x_t, noise

    def get_v_se3(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity target on SE(3).

        For flow matching, velocity points from clean to noisy:
        v = d/dt x_t = noise - x_0 (for translation)
        v_rot = log(R_noise @ R_0^T) (for rotation, constant velocity on SO(3))

        Args:
            x_0: Clean pose [B, 6] or [B, N, 6]
            noise: Noise pose, same shape as x_0
            t: Timestep [B,] (not used, but kept for API consistency)

        Returns:
            v: Velocity target, same shape as x_0
        """
        # Handle different input shapes
        if x_0.dim() == 2:
            x_0_trans = x_0[:, :3]
            x_0_rot_vec = x_0[:, 3:]
            noise_trans = noise[:, :3]
            noise_rot_vec = noise[:, 3:]
        else:
            x_0_trans = x_0[:, :, :3]
            x_0_rot_vec = x_0[:, :, 3:]
            noise_trans = noise[:, :, :3]
            noise_rot_vec = noise[:, :, 3:]

        # Translation velocity: v_trans = (1 - sigma_min) * noise - x_0
        v_trans = (1 - self.sigma_min) * noise_trans - x_0_trans

        # Rotation velocity: v_rot = log(R_noise @ R_0^T)
        x_0_rot_mat = self._rot_vec_to_matrix(x_0_rot_vec)
        noise_rot_mat = self._rot_vec_to_matrix(noise_rot_vec)

        if x_0.dim() == 2:
            rot_rel = torch.bmm(noise_rot_mat, x_0_rot_mat.transpose(-2, -1))
        else:
            rot_rel = torch.matmul(noise_rot_mat, x_0_rot_mat.transpose(-2, -1))

        v_rot = self._matrix_to_rot_vec(rot_rel)

        v = torch.cat([v_trans, v_rot], dim=-1)
        return v

    def step_se3(
        self,
        v_pred: torch.Tensor,
        t: float,
        t_prev: float,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Euler step on SE(3).

        x_{t_prev} = x_t + (t_prev - t) * v_pred

        Note: t > t_prev when integrating backward (noise → clean)

        Args:
            v_pred: Predicted velocity [B, 6] or [B, N, 6]
            t: Current timestep
            t_prev: Next timestep
            x_t: Current pose, same shape as v_pred

        Returns:
            x_prev: Updated pose, same shape as x_t
        """
        delta_t = t_prev - t  # Typically negative when going from noise to clean

        # Handle different input shapes
        if x_t.dim() == 2:
            x_t_trans = x_t[:, :3]
            x_t_rot_vec = x_t[:, 3:]
            v_trans = v_pred[:, :3]
            v_rot = v_pred[:, 3:]
        else:
            x_t_trans = x_t[:, :, :3]
            x_t_rot_vec = x_t[:, :, 3:]
            v_trans = v_pred[:, :, :3]
            v_rot = v_pred[:, :, 3:]

        # Translation: standard Euler step
        x_prev_trans = x_t_trans + delta_t * v_trans

        # Rotation: Euler step on SO(3)
        # R_prev = exp(delta_t * v_rot) @ R_t
        delta_rot = self._rot_vec_to_matrix(delta_t * v_rot)
        x_t_rot_mat = self._rot_vec_to_matrix(x_t_rot_vec)

        if x_t.dim() == 2:
            x_prev_rot_mat = torch.bmm(delta_rot, x_t_rot_mat)
        else:
            x_prev_rot_mat = torch.matmul(delta_rot, x_t_rot_mat)

        x_prev_rot_vec = self._matrix_to_rot_vec(x_prev_rot_mat)

        x_prev = torch.cat([x_prev_trans, x_prev_rot_vec], dim=-1)
        return x_prev

    def get_timesteps(
        self,
        num_steps: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Get inference timesteps with optional shifting.

        Args:
            num_steps: Number of integration steps
            device: Device for timesteps

        Returns:
            timesteps: [num_steps + 1,] tensor from 1 to 0
        """
        t_seq = torch.linspace(1, 0, num_steps + 1, device=device)

        # Apply timestep shifting
        if self.shift_t != 1.0:
            t_seq = self.shift_t * t_seq / (1 + (self.shift_t - 1) * t_seq)

        return t_seq
