"""Construct view-cell support points and shared train/inference features."""
from __future__ import annotations

import math
import numpy as np


def _camera_basis(forward: np.ndarray) -> np.ndarray:
    f = np.asarray(forward, dtype=np.float64)
    f /= max(float(np.linalg.norm(f)), 1e-12)
    up_seed = np.asarray([0.0, 0.0, 1.0] if abs(float(f[1])) > 0.98 else [0.0, 1.0, 0.0])
    right = np.cross(f, up_seed)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(right, f)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    return np.stack([right, up, f], axis=0).astype(np.float32)


def _support_points(
    center: np.ndarray,
    basis: np.ndarray,
    shape: str,
    half_extent: np.ndarray,
) -> np.ndarray:
    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(half_extent, dtype=np.float64)
    if shape == "horizontal_disk":
        angles = np.arange(8, dtype=np.float64) * (2.0 * math.pi / 8.0)
        circle = np.column_stack(
            [np.cos(angles) * half[0], np.zeros((8,)), np.sin(angles) * half[0]]
        )
        return np.concatenate([center[None, :], center[None, :] + circle], axis=0).astype(np.float32)
    if shape != "camera_aligned_box":
        raise ValueError(f"unsupported V5 view-cell shape: {shape}")
    # Registry order is half-right, half-forward, half-up.
    axes = np.stack(
        [basis[0] * half[0], basis[2] * half[1], basis[1] * half[2]], axis=0
    )
    signs = np.asarray(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )
    corners = center[None, :] + signs @ axes
    return np.concatenate([center[None, :], corners], axis=0).astype(np.float32)


def _query_geometry_numpy(
    target_centers: np.ndarray,
    target_radii: np.ndarray,
    region_center: np.ndarray,
    basis: np.ndarray,
    half_axes: np.ndarray,
    tan_fov: np.ndarray,
    region_type: float,
    near: float,
    far: float,
) -> np.ndarray:
    offset = np.asarray(region_center, dtype=np.float64)[None, :] - target_centers
    distance = np.linalg.norm(offset, axis=1)
    world = np.divide(
        offset,
        distance[:, None],
        out=np.zeros_like(offset),
        where=distance[:, None] > 1e-12,
    )
    reverse_camera = (-world) @ basis.T
    radius = target_radii.astype(np.float64)
    denominator = distance + radius
    axis_denominator = denominator + float(np.max(half_axes))
    return np.column_stack(
        [
            world,
            reverse_camera,
            np.log1p(distance / radius),
            radius / denominator,
            np.broadcast_to(half_axes[None, :], (target_centers.shape[0], 3))
            / axis_denominator[:, None],
            np.broadcast_to(tan_fov[None, :], (target_centers.shape[0], 2)),
            np.full((target_centers.shape[0],), region_type),
            np.full((target_centers.shape[0],), near) / denominator,
            np.log1p(float(far) / denominator),
        ]
    ).astype(np.float32)
