"""Procedural camera cells and conservative candidate generation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from gcof_pvs.data.pose_csr import frustum_candidate_ids_for_pose
from .geometry import PrimitiveScene, _normalize
from .protocol import DEFAULT_NEAR, MODEL_FOV_Y_DEG, RENDER_FOV_Y_DEG, SUBPOSES_PER_VIEWCELL, SYNTHETIC_CAMERA_ORBIT_FRACTION
from .raycast import raycast_primitives


def _camera_basis(forward: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = _normalize(np.asarray(forward, dtype=np.float64), [0.0, 0.0, -1.0])
    up_seed = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(f @ up_seed)) > 0.98:
        up_seed = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = _normalize(np.cross(f, up_seed), [1.0, 0.0, 0.0])
    up = _normalize(np.cross(right, f), [0.0, 1.0, 0.0])
    return f, right, up


def _camera_rays(
    origin: np.ndarray,
    forward: np.ndarray,
    width: int,
    height: int,
    fov_y_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    f, right, up = _camera_basis(forward)
    aspect = float(width) / float(height)
    tan_y = math.tan(math.radians(float(fov_y_deg)) * 0.5)
    tan_x = tan_y * aspect
    x = ((np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0) * tan_x
    y = (1.0 - (np.arange(height, dtype=np.float64) + 0.5) / height * 2.0) * tan_y
    xx, yy = np.meshgrid(x, y, indexing="xy")
    directions = f[None, None, :] + xx[..., None] * right[None, None, :] + yy[..., None] * up[None, None, :]
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    origins = np.broadcast_to(np.asarray(origin, dtype=np.float64), directions.shape).copy()
    return origins.reshape(-1, 3), directions.reshape(-1, 3)


@dataclass(frozen=True)
class ViewCell:
    viewcell_id: int
    center: np.ndarray
    forward: np.ndarray
    region_type: str
    radius: float
    half_axes: np.ndarray
    right: np.ndarray
    up: np.ndarray
    subpose_positions: np.ndarray
    subpose_forwards: np.ndarray
    yaw_deg: float
    pitch_deg: float
    back_offset: float

    def region_manifest(self) -> dict[str, Any]:
        if self.region_type == "disk":
            support = [self.center]
            angles = np.arange(8, dtype=np.float64) * math.pi / 4.0
            horizontal_forward = _normalize(self.half_axes[2], [0.0, 0.0, -1.0])
            support.extend(self.center + self.radius * (np.cos(angle) * self.right + np.sin(angle) * horizontal_forward) for angle in angles)
            return {
                "regionType": "disk",
                "center": self.center.tolist(),
                "radius": self.radius,
                "right": self.right.tolist(),
                "forward": horizontal_forward.tolist(),
                "supportPoints": np.asarray(support).tolist(),
            }
        axes = self.half_axes
        signs = np.asarray([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
        return {
            "regionType": "oriented_box",
            "center": self.center.tolist(),
            "halfAxes": axes.tolist(),
            "supportPoints": np.vstack([self.center, self.center[None, :] + signs @ axes]).tolist(),
        }


def build_viewcells(
    scene: PrimitiveScene,
    scene_entry: Mapping[str, Any],
    *,
    views_per_scene: int,
) -> list[ViewCell]:
    if views_per_scene <= 0:
        raise ValueError("views_per_scene must be positive")
    seed = int(scene_entry["generatorSeed"]) + 0x51ED2705
    rng = np.random.default_rng(seed)
    scene_center = (scene.bounds[0] + scene.bounds[1]) * 0.5
    scene_extent = scene.bounds[1] - scene.bounds[0]
    horizontal_extent = max(float(scene_extent[0]), float(scene_extent[2]), 1.0)
    # The old external orbit put the complete synthetic scene inside one 66
    # degree frustum. V5 samples view cells in the navigable scene volume so
    # candidate generation exercises the intended geometry-only culling path.
    orbit_radius = max(horizontal_extent * SYNTHETIC_CAMERA_ORBIT_FRACTION, 8.0)
    viewcells: list[ViewCell] = []
    for viewcell_id in range(views_per_scene):
        azimuth = 2.0 * math.pi * viewcell_id / views_per_scene + float(rng.uniform(-0.04, 0.04))
        elevation = float(rng.uniform(-0.10, 0.24))
        horizontal = np.asarray([math.cos(azimuth), 0.0, math.sin(azimuth)])
        position = scene_center + horizontal * (orbit_radius * float(rng.uniform(1.0, 1.25)))
        position[1] = scene_center[1] + scene_extent[1] * (0.25 + 0.5 * float(rng.random()))
        forward = _normalize(scene_center + np.asarray([0.0, scene_extent[1] * 0.18, 0.0]) - position, [0.0, 0.0, -1.0])
        f, right, up = _camera_basis(forward)
        radius = max(0.45, min(3.0, horizontal_extent * float(rng.uniform(0.018, 0.032))))
        region_type = "disk" if viewcell_id % 2 == 0 else "oriented_box"
        if region_type == "disk":
            horizontal_forward = _normalize(np.cross(right, np.asarray([0.0, 1.0, 0.0])), [0.0, 0.0, -1.0])
            # Keep the disk in the world-horizontal plane.
            half_axes = np.stack([right * radius, np.zeros(3), horizontal_forward * radius], axis=0)
            offsets = np.asarray(
                [
                    [0.0, 0.0],
                    [0.55 * radius, 0.0],
                    [-0.55 * radius, 0.0],
                    [0.0, 0.55 * radius],
                ]
            )
            subpose_positions = position[None, :] + offsets[:, 0, None] * right[None, :] + offsets[:, 1, None] * horizontal_forward[None, :]
        else:
            vertical_radius = max(0.35, min(1.5, scene_extent[1] * 0.02))
            half_axes = np.stack([right * radius, f * radius, up * vertical_radius], axis=0)
            offsets = np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.5 * radius, 0.35 * radius, 0.35 * vertical_radius],
                    [-0.5 * radius, -0.35 * radius, -0.35 * vertical_radius],
                    [0.25 * radius, -0.4 * radius, 0.4 * vertical_radius],
                ]
            )
            subpose_positions = position[None, :] + offsets[:, 0, None] * right[None, :] + offsets[:, 1, None] * f[None, :] + offsets[:, 2, None] * up[None, :]
        viewcells.append(
            ViewCell(
                viewcell_id,
                position.astype(np.float64),
                f.astype(np.float64),
                region_type,
                radius,
                half_axes.astype(np.float64),
                right.astype(np.float64),
                up.astype(np.float64),
                subpose_positions.astype(np.float64),
                np.broadcast_to(f[None, :], (SUBPOSES_PER_VIEWCELL, 3)).copy().astype(np.float64),
                math.degrees(azimuth),
                math.degrees(elevation),
                max(2.0 * radius, orbit_radius * 0.18),
            )
        )
    return viewcells


def _candidate_ids_for_viewcell(
    scene: PrimitiveScene,
    viewcell: ViewCell,
    *,
    width: int,
    height: int,
    near: float = DEFAULT_NEAR,
) -> np.ndarray:
    tan_y = math.tan(math.radians(MODEL_FOV_Y_DEG) * 0.5)
    tan_x = tan_y * float(width) / float(height)
    aabbs = scene.aabbs.reshape(scene.unit_count, 6)
    selected: list[np.ndarray] = []
    # A subpose stores the query position. Candidate generation uses the
    # corresponding backed candidate camera, matching the runtime formula
    # candidate_camera = query_center - forward * back_offset.
    candidate_origins = (
        viewcell.subpose_positions
        - viewcell.subpose_forwards * float(viewcell.back_offset)
    )
    for origin, forward in zip(candidate_origins, viewcell.subpose_forwards):
        selected.append(
            frustum_candidate_ids_for_pose(origin, forward, tan_x, tan_y, aabbs, near=near).astype(np.uint32)
        )
    if not selected:
        return np.zeros((0,), dtype=np.uint32)
    return np.unique(np.concatenate(selected)).astype(np.uint32)


def render_viewcell_color_id(
    scene: PrimitiveScene,
    viewcell: ViewCell,
    *,
    render_width: int,
    render_height: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render all subposes and aggregate IDs, max pixel-ppm, and hit counts."""

    max_weights = np.zeros(scene.unit_count, dtype=np.float32)
    hit_counts = np.zeros(scene.unit_count, dtype=np.uint16)
    for origin, forward in zip(viewcell.subpose_positions, viewcell.subpose_forwards):
        origins, directions = _camera_rays(origin, forward, render_width, render_height, RENDER_FOV_Y_DEG)
        ids = raycast_primitives(scene, origins, directions, device=device)
        valid = ids >= 0
        if not bool(valid.any()):
            continue
        counts = np.bincount(ids[valid], minlength=scene.unit_count).astype(np.float64)
        weights = (counts / float(render_width * render_height) * 1_000_000.0).astype(np.float32)
        present = counts > 0.0
        hit_counts[present] = np.minimum(hit_counts[present].astype(np.uint32) + 1, np.iinfo(np.uint16).max).astype(np.uint16)
        max_weights = np.maximum(max_weights, weights)
    visible = np.flatnonzero(max_weights > 0.0).astype(np.uint32)
    return visible, max_weights[visible].astype(np.float32), hit_counts[visible]


def _conservative_mvp(camera_world: np.ndarray, forward: np.ndarray, tan_x: float, tan_y: float) -> np.ndarray:
    f, right, up = _camera_basis(forward)
    camera = np.asarray(camera_world, dtype=np.float64)
    result = np.zeros((16,), dtype=np.float32)
    result[0], result[4], result[8], result[12] = right[0] / tan_x, right[1] / tan_x, right[2] / tan_x, -float(right @ camera) / tan_x
    result[1], result[5], result[9], result[13] = up[0] / tan_y, up[1] / tan_y, up[2] / tan_y, -float(up @ camera) / tan_y
    result[2], result[6], result[10], result[14] = f[0], f[1], f[2], -float(f @ camera)
    result[3], result[7], result[11], result[15] = f[0], f[1], f[2], -float(f @ camera)
    return result


def _camera_bounds(scene: PrimitiveScene, viewcells: Sequence[ViewCell]) -> dict[str, list[float]]:
    cameras = np.asarray([cell.center - cell.forward * cell.back_offset for cell in viewcells], dtype=np.float64)
    low = np.minimum(scene.bounds[0], cameras.min(axis=0))
    high = np.maximum(scene.bounds[1], cameras.max(axis=0))
    padding = np.maximum((high - low) * 0.05, 1.0)
    low -= padding
    high += padding
    return {
        "min": low.tolist(),
        "max": high.tolist(),
        "center": ((low + high) * 0.5).tolist(),
        "size": (high - low).tolist(),
    }


def _normalize_camera(camera: np.ndarray, bounds: Mapping[str, Any]) -> np.ndarray:
    low = np.asarray(bounds["min"], dtype=np.float64)
    high = np.asarray(bounds["max"], dtype=np.float64)
    return ((camera - low) / np.maximum(high - low, 1.0e-12) * 2.0 - 1.0).astype(np.float32)
