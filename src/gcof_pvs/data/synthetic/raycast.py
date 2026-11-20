"""Nearest-hit ray casting against procedural scene primitives."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .geometry import PrimitiveScene, _PRIMITIVE_TYPES


@dataclass(frozen=True)
class _UniformGrid:
    low: np.ndarray
    cell_size: float
    dimensions: np.ndarray
    members: np.ndarray


def _build_uniform_grid(scene: PrimitiveScene) -> _UniformGrid:
    """Build a conservative AABB cell index for ray broad-phase queries."""

    extent = np.maximum(scene.bounds[1] - scene.bounds[0], 1.0e-6)
    target_axis = max(8, min(32, int(math.ceil(scene.unit_count ** (1.0 / 3.0) * 2.0))))
    cell_size = float(np.max(extent) / target_axis)
    dimensions = np.maximum(1, np.ceil(extent / cell_size).astype(np.int64))
    cell_count = int(np.prod(dimensions))
    cells: list[list[int]] = [[] for _ in range(cell_count)]

    def linear(index: np.ndarray) -> int:
        return int(index[0] + dimensions[0] * (index[1] + dimensions[1] * index[2]))

    for unit_index, aabb in enumerate(scene.aabbs):
        minimum = np.floor((aabb[0] - scene.bounds[0]) / cell_size).astype(np.int64)
        maximum = np.floor((aabb[1] - scene.bounds[0]) / cell_size).astype(np.int64)
        minimum = np.clip(minimum, 0, dimensions - 1)
        maximum = np.clip(maximum, 0, dimensions - 1)
        for z in range(int(minimum[2]), int(maximum[2]) + 1):
            for y in range(int(minimum[1]), int(maximum[1]) + 1):
                for x in range(int(minimum[0]), int(maximum[0]) + 1):
                    cells[linear(np.asarray([x, y, z]))].append(unit_index)
    max_members = max((len(cell) for cell in cells), default=1)
    members = np.full((cell_count, max(1, max_members)), -1, dtype=np.int64)
    for cell_index, cell in enumerate(cells):
        if cell:
            members[cell_index, : len(cell)] = np.asarray(cell, dtype=np.int64)
    return _UniformGrid(scene.bounds[0].astype(np.float64), cell_size, dimensions, members)


def _scene_grid(scene: PrimitiveScene) -> _UniformGrid:
    if scene._grid is None:
        scene._grid = _build_uniform_grid(scene)
    return scene._grid


def _grid_candidate_matrix(
    scene: PrimitiveScene,
    origins: np.ndarray,
    directions: np.ndarray,
    limits: np.ndarray | None,
) -> np.ndarray:
    """Return conservative per-ray unit-index rows from a 3-D DDA."""

    grid = _scene_grid(scene)
    low = grid.low
    high = low + grid.cell_size * grid.dimensions.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (low[None, :] - origins) / directions
        t1 = (high[None, :] - origins) / directions
    t_min = np.minimum(t0, t1)
    t_max = np.maximum(t0, t1)
    parallel = np.abs(directions) <= 1.0e-12
    parallel_outside = parallel & ((origins < low[None, :] - 1.0e-9) | (origins > high[None, :] + 1.0e-9))
    entry = np.max(np.where(parallel, -np.inf, t_min), axis=1)
    exit_time = np.min(np.where(parallel, np.inf, t_max), axis=1)
    valid = (~parallel_outside.any(axis=1)) & (exit_time >= np.maximum(entry, 0.0))
    entry = np.maximum(entry, 0.0)
    if limits is not None:
        exit_time = np.minimum(exit_time, limits)
        valid &= exit_time >= entry

    max_steps = int(grid.dimensions.sum()) + 3
    max_members = grid.members.shape[1]
    candidates = np.full((origins.shape[0], max_steps * max_members), -1, dtype=np.int64)
    if not bool(valid.any()):
        return candidates
    # Move a tiny amount inside the first cell so a ray entering exactly on a
    # cell boundary uses the cell it actually travels through.
    initial_t = entry + max(1.0e-9, grid.cell_size * 1.0e-9)
    points = origins + directions * initial_t[:, None]
    cells = np.floor((points - low[None, :]) / grid.cell_size).astype(np.int64)
    cells = np.clip(cells, 0, grid.dimensions - 1)
    steps = np.sign(directions).astype(np.int64)
    abs_direction = np.abs(directions)
    delta_t = np.divide(
        grid.cell_size,
        abs_direction,
        out=np.full_like(abs_direction, np.inf),
        where=abs_direction > 1.0e-12,
    )
    boundary_index = np.where(steps > 0, cells + 1, cells)
    boundaries = low[None, :] + boundary_index * grid.cell_size
    next_t = np.divide(
        boundaries - origins,
        directions,
        out=np.full_like(directions, np.inf),
        where=abs_direction > 1.0e-12,
    )
    active = valid.copy()
    for step_index in range(max_steps):
        linear_ids = cells[:, 0] + grid.dimensions[0] * (cells[:, 1] + grid.dimensions[1] * cells[:, 2])
        rows = grid.members[np.clip(linear_ids, 0, grid.members.shape[0] - 1)]
        start = step_index * max_members
        candidates[active, start : start + max_members] = rows[active]
        axis = np.argmin(next_t, axis=1)
        next_boundary = np.min(next_t, axis=1)
        advance = active & (next_boundary <= exit_time + 1.0e-9)
        if not bool(advance.any()):
            break
        rows_index = np.flatnonzero(advance)
        axes = axis[rows_index]
        cells[rows_index, axes] += steps[rows_index, axes]
        next_t[rows_index, axes] += delta_t[rows_index, axes]
        in_bounds = np.all((cells[rows_index] >= 0) & (cells[rows_index] < grid.dimensions[None, :]), axis=1)
        active[:] = False
        active[rows_index[in_bounds]] = True
    return candidates


def _intersect_box_numpy(
    origins: np.ndarray,
    directions: np.ndarray,
    centers: np.ndarray,
    rotations: np.ndarray,
    dimensions: np.ndarray,
) -> np.ndarray:
    relative = origins[:, None, :] - centers[None, :, :]
    local_origins = np.einsum("bmi,mij->bmj", relative, rotations)
    local_directions = np.einsum("bi,mij->bmj", directions, rotations)
    half = dimensions[None, :, :] * 0.5
    lower = -half
    upper = half
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (lower - local_origins) / local_directions
        t1 = (upper - local_origins) / local_directions
    t_min = np.minimum(t0, t1)
    t_max = np.maximum(t0, t1)
    parallel = np.abs(local_directions) <= 1.0e-12
    parallel_outside = parallel & ((local_origins < lower - 1.0e-10) | (local_origins > upper + 1.0e-10))
    near = np.max(np.where(parallel, -np.inf, t_min), axis=-1)
    far = np.min(np.where(parallel, np.inf, t_max), axis=-1)
    valid = (~parallel_outside.any(axis=-1)) & (far >= np.maximum(near, 0.0)) & (far > 1.0e-8)
    result = np.where(near > 1.0e-8, near, far)
    return np.where(valid, result, np.inf)


def _intersect_cylinder_numpy(
    origins: np.ndarray,
    directions: np.ndarray,
    centers: np.ndarray,
    rotations: np.ndarray,
    dimensions: np.ndarray,
) -> np.ndarray:
    relative = origins[:, None, :] - centers[None, :, :]
    local_origins = np.einsum("bmi,mij->bmj", relative, rotations)
    local_directions = np.einsum("bi,mij->bmj", directions, rotations)
    radius = dimensions[None, :, 0] * 0.5
    half_height = dimensions[None, :, 1] * 0.5
    ox, oy, oz = local_origins[..., 0], local_origins[..., 1], local_origins[..., 2]
    dx, dy, dz = local_directions[..., 0], local_directions[..., 1], local_directions[..., 2]
    a = dx * dx + dz * dz
    b = 2.0 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - radius * radius
    discriminant = b * b - 4.0 * a * c
    valid_quadratic = (a > 1.0e-14) & (discriminant >= 0.0)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (-b - root) / (2.0 * a)
        t1 = (-b + root) / (2.0 * a)
    side0 = valid_quadratic & (t0 > 1.0e-8) & (np.abs(oy + t0 * dy) <= half_height + 1.0e-9)
    side1 = valid_quadratic & (t1 > 1.0e-8) & (np.abs(oy + t1 * dy) <= half_height + 1.0e-9)
    side_t = np.where(side0, t0, np.inf)
    side_t = np.minimum(side_t, np.where(side1, t1, np.inf))

    parallel_y = np.abs(dy) <= 1.0e-14
    with np.errstate(divide="ignore", invalid="ignore"):
        bottom_t = (-half_height - oy) / dy
        top_t = (half_height - oy) / dy
    bottom_valid = (~parallel_y) & (bottom_t > 1.0e-8) & (
        (ox + bottom_t * dx) ** 2 + (oz + bottom_t * dz) ** 2 <= radius * radius + 1.0e-9
    )
    top_valid = (~parallel_y) & (top_t > 1.0e-8) & (
        (ox + top_t * dx) ** 2 + (oz + top_t * dz) ** 2 <= radius * radius + 1.0e-9
    )
    cap_t = np.minimum(np.where(bottom_valid, bottom_t, np.inf), np.where(top_valid, top_t, np.inf))
    return np.minimum(side_t, cap_t)


def _intersect_box_torch(
    origins: Any,
    directions: Any,
    centers: Any,
    rotations: Any,
    dimensions: Any,
) -> Any:
    import torch

    relative = origins[:, None, :] - centers[None, :, :]
    local_origins = torch.einsum("bmi,mij->bmj", relative, rotations)
    local_directions = torch.einsum("bi,mij->bmj", directions, rotations)
    half = dimensions[None, :, :] * 0.5
    lower = -half
    upper = half
    t0 = torch.where(torch.abs(local_directions) > 1.0e-12, (lower - local_origins) / local_directions, torch.full_like(local_directions, -torch.inf))
    t1 = torch.where(torch.abs(local_directions) > 1.0e-12, (upper - local_origins) / local_directions, torch.full_like(local_directions, torch.inf))
    t_min = torch.minimum(t0, t1)
    t_max = torch.maximum(t0, t1)
    parallel = torch.abs(local_directions) <= 1.0e-12
    parallel_outside = parallel & ((local_origins < lower - 1.0e-10) | (local_origins > upper + 1.0e-10))
    near = torch.where(parallel, torch.full_like(t_min, -torch.inf), t_min).amax(dim=-1)
    far = torch.where(parallel, torch.full_like(t_max, torch.inf), t_max).amin(dim=-1)
    valid = (~parallel_outside.any(dim=-1)) & (far >= torch.maximum(near, torch.zeros_like(near))) & (far > 1.0e-8)
    result = torch.where(near > 1.0e-8, near, far)
    return torch.where(valid, result, torch.full_like(result, torch.inf))


def _intersect_cylinder_torch(
    origins: Any,
    directions: Any,
    centers: Any,
    rotations: Any,
    dimensions: Any,
) -> Any:
    import torch

    relative = origins[:, None, :] - centers[None, :, :]
    local_origins = torch.einsum("bmi,mij->bmj", relative, rotations)
    local_directions = torch.einsum("bi,mij->bmj", directions, rotations)
    radius = dimensions[None, :, 0] * 0.5
    half_height = dimensions[None, :, 1] * 0.5
    ox, oy, oz = local_origins[..., 0], local_origins[..., 1], local_origins[..., 2]
    dx, dy, dz = local_directions[..., 0], local_directions[..., 1], local_directions[..., 2]
    a = dx * dx + dz * dz
    b = 2.0 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - radius * radius
    discriminant = b * b - 4.0 * a * c
    valid_quadratic = (a > 1.0e-14) & (discriminant >= 0.0)
    root = torch.sqrt(torch.clamp_min(discriminant, 0.0))
    t0 = torch.where(valid_quadratic, (-b - root) / (2.0 * a), torch.full_like(a, torch.inf))
    t1 = torch.where(valid_quadratic, (-b + root) / (2.0 * a), torch.full_like(a, torch.inf))
    side0 = valid_quadratic & (t0 > 1.0e-8) & (torch.abs(oy + t0 * dy) <= half_height + 1.0e-9)
    side1 = valid_quadratic & (t1 > 1.0e-8) & (torch.abs(oy + t1 * dy) <= half_height + 1.0e-9)
    side_t = torch.minimum(torch.where(side0, t0, torch.full_like(t0, torch.inf)), torch.where(side1, t1, torch.full_like(t1, torch.inf)))
    parallel_y = torch.abs(dy) <= 1.0e-14
    bottom_t = torch.where(parallel_y, torch.full_like(dy, torch.inf), (-half_height - oy) / dy)
    top_t = torch.where(parallel_y, torch.full_like(dy, torch.inf), (half_height - oy) / dy)
    bottom_valid = (~parallel_y) & (bottom_t > 1.0e-8) & ((ox + bottom_t * dx) ** 2 + (oz + bottom_t * dz) ** 2 <= radius * radius + 1.0e-9)
    top_valid = (~parallel_y) & (top_t > 1.0e-8) & ((ox + top_t * dx) ** 2 + (oz + top_t * dz) ** 2 <= radius * radius + 1.0e-9)
    cap_t = torch.minimum(torch.where(bottom_valid, bottom_t, torch.full_like(bottom_t, torch.inf)), torch.where(top_valid, top_t, torch.full_like(top_t, torch.inf)))
    return torch.minimum(side_t, cap_t)


def _intersect_box_candidates_numpy(
    origins: np.ndarray,
    directions: np.ndarray,
    candidate_indices: np.ndarray,
    scene: PrimitiveScene,
) -> tuple[np.ndarray, np.ndarray]:
    safe = np.maximum(candidate_indices, 0)
    centers = scene.centers[safe]
    rotations = scene.rotations[safe]
    dimensions = scene.dimensions[safe]
    relative = origins[:, None, :] - centers
    local_origins = np.einsum("bki,bkij->bkj", relative, rotations)
    local_directions = np.einsum("bi,bkij->bkj", directions, rotations)
    half = dimensions * 0.5
    lower = -half
    upper = half
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (lower - local_origins) / local_directions
        t1 = (upper - local_origins) / local_directions
    t_min = np.minimum(t0, t1)
    t_max = np.maximum(t0, t1)
    parallel = np.abs(local_directions) <= 1.0e-12
    parallel_outside = parallel & ((local_origins < lower - 1.0e-10) | (local_origins > upper + 1.0e-10))
    near = np.max(np.where(parallel, -np.inf, t_min), axis=-1)
    far = np.min(np.where(parallel, np.inf, t_max), axis=-1)
    valid = (~parallel_outside.any(axis=-1)) & (far >= np.maximum(near, 0.0)) & (far > 1.0e-8)
    result = np.where(valid, np.where(near > 1.0e-8, near, far), np.inf)
    result[candidate_indices < 0] = np.inf
    return result, scene.unit_ids[safe]


def _intersect_cylinder_candidates_numpy(
    origins: np.ndarray,
    directions: np.ndarray,
    candidate_indices: np.ndarray,
    scene: PrimitiveScene,
) -> tuple[np.ndarray, np.ndarray]:
    safe = np.maximum(candidate_indices, 0)
    centers = scene.centers[safe]
    rotations = scene.rotations[safe]
    dimensions = scene.dimensions[safe]
    relative = origins[:, None, :] - centers
    local_origins = np.einsum("bki,bkij->bkj", relative, rotations)
    local_directions = np.einsum("bi,bkij->bkj", directions, rotations)
    radius = dimensions[..., 0] * 0.5
    half_height = dimensions[..., 1] * 0.5
    ox, oy, oz = local_origins[..., 0], local_origins[..., 1], local_origins[..., 2]
    dx, dy, dz = local_directions[..., 0], local_directions[..., 1], local_directions[..., 2]
    a = dx * dx + dz * dz
    b = 2.0 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - radius * radius
    discriminant = b * b - 4.0 * a * c
    valid_quadratic = (a > 1.0e-14) & (discriminant >= 0.0)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (-b - root) / (2.0 * a)
        t1 = (-b + root) / (2.0 * a)
    side0 = valid_quadratic & (t0 > 1.0e-8) & (np.abs(oy + t0 * dy) <= half_height + 1.0e-9)
    side1 = valid_quadratic & (t1 > 1.0e-8) & (np.abs(oy + t1 * dy) <= half_height + 1.0e-9)
    side_t = np.minimum(np.where(side0, t0, np.inf), np.where(side1, t1, np.inf))
    parallel_y = np.abs(dy) <= 1.0e-14
    with np.errstate(divide="ignore", invalid="ignore"):
        bottom_t = (-half_height - oy) / dy
        top_t = (half_height - oy) / dy
    bottom_valid = (~parallel_y) & (bottom_t > 1.0e-8) & ((ox + bottom_t * dx) ** 2 + (oz + bottom_t * dz) ** 2 <= radius * radius + 1.0e-9)
    top_valid = (~parallel_y) & (top_t > 1.0e-8) & ((ox + top_t * dx) ** 2 + (oz + top_t * dz) ** 2 <= radius * radius + 1.0e-9)
    cap_t = np.minimum(np.where(bottom_valid, bottom_t, np.inf), np.where(top_valid, top_t, np.inf))
    result = np.minimum(side_t, cap_t)
    result[candidate_indices < 0] = np.inf
    return result, scene.unit_ids[safe]


def _nearest_candidate_numpy(
    scene: PrimitiveScene,
    origins: np.ndarray,
    directions: np.ndarray,
    candidate_indices: np.ndarray,
    limit: np.ndarray,
    ignored: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    primitive_types = scene.primitive_types[candidate_indices.clip(min=0)]
    distances = np.full(candidate_indices.shape, np.inf, dtype=np.float64)
    candidate_ids = scene.unit_ids[np.maximum(candidate_indices, 0)]
    box_mask = (primitive_types != "cylinder") & (candidate_indices >= 0)
    cylinder_mask = (primitive_types == "cylinder") & (candidate_indices >= 0)
    if bool(box_mask.any()):
        box_candidates = np.where(box_mask, candidate_indices, -1)
        box_distances, _ = _intersect_box_candidates_numpy(origins, directions, box_candidates, scene)
        distances = np.where(box_mask, box_distances, distances)
    if bool(cylinder_mask.any()):
        cylinder_candidates = np.where(cylinder_mask, candidate_indices, -1)
        cylinder_distances, _ = _intersect_cylinder_candidates_numpy(origins, directions, cylinder_candidates, scene)
        distances = np.where(cylinder_mask, cylinder_distances, distances)
    if ignored is not None:
        distances = np.where(candidate_ids == ignored[:, None], np.inf, distances)
    distances = np.where(distances <= limit[:, None], distances, np.inf)
    finite = np.isfinite(distances)
    best_t = np.min(distances, axis=1)
    with np.errstate(invalid="ignore"):
        tie = finite & (np.abs(distances - best_t[:, None]) <= 1.0e-10)
    best_ids = np.min(np.where(tie, candidate_ids, np.iinfo(np.int64).max), axis=1)
    best_ids[~np.isfinite(best_t)] = -1
    return best_ids, best_t


def _nearest_candidate_torch(
    origins: Any,
    directions: Any,
    candidate_indices: np.ndarray,
    scene: PrimitiveScene,
    limit: Any,
    ignored: Any | None,
    scene_centers: Any,
    scene_rotations: Any,
    scene_dimensions: Any,
) -> tuple[Any, Any]:
    import torch

    safe_np = np.maximum(candidate_indices, 0)
    candidate_ids = torch.as_tensor(scene.unit_ids[safe_np], dtype=torch.int64, device=origins.device)
    cylinder_mask_np = (scene.primitive_types[safe_np] == "cylinder") & (candidate_indices >= 0)
    box_mask_np = (~cylinder_mask_np) & (candidate_indices >= 0)
    candidate_index_tensor = torch.as_tensor(candidate_indices, dtype=torch.int64, device=origins.device)
    distances = torch.full(candidate_index_tensor.shape, torch.inf, dtype=origins.dtype, device=origins.device)
    if bool(box_mask_np.any()):
        box_candidates = torch.where(
            torch.as_tensor(box_mask_np, dtype=torch.bool, device=origins.device),
            candidate_index_tensor,
            torch.full_like(candidate_index_tensor, -1),
        )
        box_distances = _intersect_box_candidates_torch(
            origins, directions, box_candidates, scene_centers, scene_rotations, scene_dimensions
        )
        distances = torch.where(torch.as_tensor(box_mask_np, dtype=torch.bool, device=origins.device), box_distances, distances)
    if bool(cylinder_mask_np.any()):
        cylinder_candidates = torch.where(
            torch.as_tensor(cylinder_mask_np, dtype=torch.bool, device=origins.device),
            candidate_index_tensor,
            torch.full_like(candidate_index_tensor, -1),
        )
        cylinder_distances = _intersect_cylinder_candidates_torch(
            origins, directions, cylinder_candidates, scene_centers, scene_rotations, scene_dimensions
        )
        distances = torch.where(torch.as_tensor(cylinder_mask_np, dtype=torch.bool, device=origins.device), cylinder_distances, distances)
    if ignored is not None:
        distances = torch.where(candidate_ids == ignored[:, None], torch.inf, distances)
    distances = torch.where(distances <= limit[:, None], distances, torch.inf)
    finite = torch.isfinite(distances)
    best_t = distances.amin(dim=1)
    ties = finite & (torch.abs(distances - best_t[:, None]) <= 1.0e-10)
    max_id = torch.full_like(candidate_ids, torch.iinfo(torch.int64).max)
    best_ids = torch.where(ties, candidate_ids, max_id).amin(dim=1)
    best_ids = torch.where(torch.isfinite(best_t), best_ids, torch.full_like(best_ids, -1))
    return best_ids, best_t


def _intersect_box_candidates_torch(
    origins: Any,
    directions: Any,
    candidate_indices: Any,
    scene_centers: Any,
    scene_rotations: Any,
    scene_dimensions: Any,
) -> Any:
    import torch

    safe = torch.clamp_min(candidate_indices, 0)
    centers = scene_centers[safe]
    rotations = scene_rotations[safe]
    dimensions = scene_dimensions[safe]
    relative = origins[:, None, :] - centers
    local_origins = torch.einsum("bki,bkij->bkj", relative, rotations)
    local_directions = torch.einsum("bi,bkij->bkj", directions, rotations)
    half = dimensions * 0.5
    lower, upper = -half, half
    parallel = torch.abs(local_directions) <= 1.0e-12
    t0 = torch.where(parallel, torch.full_like(local_directions, -torch.inf), (lower - local_origins) / local_directions)
    t1 = torch.where(parallel, torch.full_like(local_directions, torch.inf), (upper - local_origins) / local_directions)
    t_min, t_max = torch.minimum(t0, t1), torch.maximum(t0, t1)
    parallel_outside = parallel & ((local_origins < lower - 1.0e-10) | (local_origins > upper + 1.0e-10))
    near = torch.where(parallel, torch.full_like(t_min, -torch.inf), t_min).amax(dim=-1)
    far = torch.where(parallel, torch.full_like(t_max, torch.inf), t_max).amin(dim=-1)
    valid = (~parallel_outside.any(dim=-1)) & (far >= torch.maximum(near, torch.zeros_like(near))) & (far > 1.0e-8)
    result = torch.where(valid, torch.where(near > 1.0e-8, near, far), torch.full_like(far, torch.inf))
    return torch.where(candidate_indices >= 0, result, torch.full_like(result, torch.inf))


def _intersect_cylinder_candidates_torch(
    origins: Any,
    directions: Any,
    candidate_indices: Any,
    scene_centers: Any,
    scene_rotations: Any,
    scene_dimensions: Any,
) -> Any:
    import torch

    safe = torch.clamp_min(candidate_indices, 0)
    centers = scene_centers[safe]
    rotations = scene_rotations[safe]
    dimensions = scene_dimensions[safe]
    relative = origins[:, None, :] - centers
    local_origins = torch.einsum("bki,bkij->bkj", relative, rotations)
    local_directions = torch.einsum("bi,bkij->bkj", directions, rotations)
    radius = dimensions[..., 0] * 0.5
    half_height = dimensions[..., 1] * 0.5
    ox, oy, oz = local_origins[..., 0], local_origins[..., 1], local_origins[..., 2]
    dx, dy, dz = local_directions[..., 0], local_directions[..., 1], local_directions[..., 2]
    a = dx * dx + dz * dz
    b = 2.0 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - radius * radius
    discriminant = b * b - 4.0 * a * c
    valid_quadratic = (a > 1.0e-14) & (discriminant >= 0.0)
    root = torch.sqrt(torch.clamp_min(discriminant, 0.0))
    t0 = torch.where(valid_quadratic, (-b - root) / (2.0 * a), torch.full_like(a, torch.inf))
    t1 = torch.where(valid_quadratic, (-b + root) / (2.0 * a), torch.full_like(a, torch.inf))
    side0 = valid_quadratic & (t0 > 1.0e-8) & (torch.abs(oy + t0 * dy) <= half_height + 1.0e-9)
    side1 = valid_quadratic & (t1 > 1.0e-8) & (torch.abs(oy + t1 * dy) <= half_height + 1.0e-9)
    side_t = torch.minimum(torch.where(side0, t0, torch.full_like(t0, torch.inf)), torch.where(side1, t1, torch.full_like(t1, torch.inf)))
    parallel_y = torch.abs(dy) <= 1.0e-14
    bottom_t = torch.where(parallel_y, torch.full_like(dy, torch.inf), (-half_height - oy) / dy)
    top_t = torch.where(parallel_y, torch.full_like(dy, torch.inf), (half_height - oy) / dy)
    bottom_valid = (~parallel_y) & (bottom_t > 1.0e-8) & ((ox + bottom_t * dx) ** 2 + (oz + bottom_t * dz) ** 2 <= radius * radius + 1.0e-9)
    top_valid = (~parallel_y) & (top_t > 1.0e-8) & ((ox + top_t * dx) ** 2 + (oz + top_t * dz) ** 2 <= radius * radius + 1.0e-9)
    cap_t = torch.minimum(torch.where(bottom_valid, bottom_t, torch.full_like(bottom_t, torch.inf)), torch.where(top_valid, top_t, torch.full_like(top_t, torch.inf)))
    result = torch.minimum(side_t, cap_t)
    return torch.where(candidate_indices >= 0, result, torch.full_like(result, torch.inf))


def _update_nearest(
    best_t: np.ndarray,
    best_ids: np.ndarray,
    distances: np.ndarray,
    unit_ids: np.ndarray,
    max_distance: np.ndarray,
    ignore_ids: np.ndarray | None,
) -> None:
    if ignore_ids is not None:
        distances = np.where(unit_ids[None, :] == ignore_ids[:, None], np.inf, distances)
    distances = np.where(distances <= max_distance[:, None], distances, np.inf)
    candidate = unit_ids[None, :]
    finite = np.isfinite(distances)
    with np.errstate(invalid="ignore"):
        better = finite & (
            (distances < best_t[:, None] - 1.0e-10)
            | ((np.abs(distances - best_t[:, None]) <= 1.0e-10) & (candidate < best_ids[:, None]))
        )
    selected_distance = np.min(np.where(better, distances, np.inf), axis=1)
    selected_index = np.argmin(np.where(better, distances, np.inf), axis=1)
    has_update = np.isfinite(selected_distance)
    rows = np.flatnonzero(has_update)
    if rows.size:
        best_t[rows] = selected_distance[rows]
        best_ids[rows] = unit_ids[selected_index[rows]]


def _raycast_cpu(
    scene: PrimitiveScene,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    max_distance: np.ndarray | None,
    ignore_unit_ids: np.ndarray | None,
    candidate_matrix: np.ndarray | None = None,
    use_grid: bool = False,
    ray_chunk: int = 4096,
    unit_chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    count = origins.shape[0]
    output_ids = np.full(count, -1, dtype=np.int64)
    output_distances = np.full(count, np.inf, dtype=np.float64)
    type_values = scene.primitive_types
    for ray_start in range(0, count, ray_chunk):
        ray_end = min(count, ray_start + ray_chunk)
        local_origins = origins[ray_start:ray_end]
        local_directions = directions[ray_start:ray_end]
        best_t = np.full(ray_end - ray_start, np.inf, dtype=np.float64)
        best_ids = np.full(ray_end - ray_start, np.iinfo(np.int64).max, dtype=np.int64)
        limit = np.full(ray_end - ray_start, np.inf, dtype=np.float64) if max_distance is None else max_distance[ray_start:ray_end]
        ignored = None if ignore_unit_ids is None else ignore_unit_ids[ray_start:ray_end]
        if candidate_matrix is not None or use_grid:
            local_candidates = (
                candidate_matrix[ray_start:ray_end]
                if candidate_matrix is not None
                else _grid_candidate_matrix(scene, local_origins, local_directions, limit)
            )
            selected_ids, selected_distances = _nearest_candidate_numpy(
                scene,
                local_origins,
                local_directions,
                local_candidates,
                limit,
                ignored,
            )
            valid = selected_ids >= 0
            output_ids[ray_start:ray_end][valid] = selected_ids[valid]
            output_distances[ray_start:ray_end] = selected_distances
            continue
        for primitive_type in _PRIMITIVE_TYPES:
            indices = np.flatnonzero(type_values == primitive_type)
            for unit_start in range(0, indices.size, unit_chunk):
                selected = indices[unit_start : unit_start + unit_chunk]
                centers = scene.centers[selected]
                rotations = scene.rotations[selected]
                dimensions = scene.dimensions[selected]
                if primitive_type == "cylinder":
                    distances = _intersect_cylinder_numpy(local_origins, local_directions, centers, rotations, dimensions)
                else:
                    distances = _intersect_box_numpy(local_origins, local_directions, centers, rotations, dimensions)
                _update_nearest(best_t, best_ids, distances, scene.unit_ids[selected], limit, ignored)
        valid = np.isfinite(best_t)
        output_ids[ray_start:ray_end][valid] = best_ids[valid]
        output_distances[ray_start:ray_end] = best_t
    return output_ids, output_distances


def _raycast_cuda(
    scene: PrimitiveScene,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    max_distance: np.ndarray | None,
    ignore_unit_ids: np.ndarray | None,
    candidate_matrix: np.ndarray | None = None,
    use_grid: bool = False,
    ray_chunk: int = 4096,
    unit_chunk: int = 512,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    count = origins.shape[0]
    output_ids = np.full(count, -1, dtype=np.int64)
    output_distances = np.full(count, np.inf, dtype=np.float64)
    type_values = scene.primitive_types
    scene_centers = torch.as_tensor(scene.centers, dtype=torch.float64, device=target)
    scene_rotations = torch.as_tensor(scene.rotations, dtype=torch.float64, device=target)
    scene_dimensions = torch.as_tensor(scene.dimensions, dtype=torch.float64, device=target)
    for ray_start in range(0, count, ray_chunk):
        ray_end = min(count, ray_start + ray_chunk)
        local_origins = torch.as_tensor(origins[ray_start:ray_end], dtype=torch.float64, device=target)
        local_directions = torch.as_tensor(directions[ray_start:ray_end], dtype=torch.float64, device=target)
        best_t = torch.full((ray_end - ray_start,), torch.inf, dtype=torch.float64, device=target)
        best_ids = torch.full((ray_end - ray_start,), np.iinfo(np.int64).max, dtype=torch.int64, device=target)
        limit = torch.full((ray_end - ray_start,), torch.inf, dtype=torch.float64, device=target) if max_distance is None else torch.as_tensor(max_distance[ray_start:ray_end], dtype=torch.float64, device=target)
        ignored = None if ignore_unit_ids is None else torch.as_tensor(ignore_unit_ids[ray_start:ray_end], dtype=torch.int64, device=target)
        if candidate_matrix is not None or use_grid:
            local_candidates = (
                candidate_matrix[ray_start:ray_end]
                if candidate_matrix is not None
                else _grid_candidate_matrix(
                    scene,
                    origins[ray_start:ray_end],
                    directions[ray_start:ray_end],
                    None if max_distance is None else max_distance[ray_start:ray_end],
                )
            )
            selected_ids, selected_distances = _nearest_candidate_torch(
                local_origins,
                local_directions,
                local_candidates,
                scene,
                limit,
                ignored,
                scene_centers,
                scene_rotations,
                scene_dimensions,
            )
            ids_cpu = selected_ids.detach().cpu().numpy()
            distances_cpu = selected_distances.detach().cpu().numpy()
            valid = ids_cpu >= 0
            output_ids[ray_start:ray_end][valid] = ids_cpu[valid]
            output_distances[ray_start:ray_end] = distances_cpu
            continue
        for primitive_type in _PRIMITIVE_TYPES:
            indices_np = np.flatnonzero(type_values == primitive_type)
            for unit_start in range(0, indices_np.size, unit_chunk):
                selected_np = indices_np[unit_start : unit_start + unit_chunk]
                selected = torch.as_tensor(selected_np, dtype=torch.int64, device=target)
                selected_ids = torch.as_tensor(scene.unit_ids[selected_np], dtype=torch.int64, device=target)
                if primitive_type == "cylinder":
                    distances = _intersect_cylinder_torch(
                        local_origins,
                        local_directions,
                        scene_centers[selected],
                        scene_rotations[selected],
                        scene_dimensions[selected],
                    )
                else:
                    distances = _intersect_box_torch(
                        local_origins,
                        local_directions,
                        scene_centers[selected],
                        scene_rotations[selected],
                        scene_dimensions[selected],
                    )
                selected_ids = selected_ids[None, :]
                if ignored is not None:
                    distances = torch.where(selected_ids == ignored[:, None], torch.inf, distances)
                distances = torch.where(distances <= limit[:, None], distances, torch.inf)
                better = (distances < best_t[:, None] - 1.0e-10) | ((torch.abs(distances - best_t[:, None]) <= 1.0e-10) & (selected_ids < best_ids[:, None]))
                candidate_t = torch.where(better, distances, torch.inf)
                selected_distance, selected_slot = candidate_t.min(dim=1)
                update = torch.isfinite(selected_distance)
                best_t = torch.where(update, selected_distance, best_t)
                ids_for_update = selected[selected_slot]
                best_ids = torch.where(update, ids_for_update, best_ids)
        ids_cpu = best_ids.detach().cpu().numpy()
        distances_cpu = best_t.detach().cpu().numpy()
        valid = np.isfinite(distances_cpu)
        output_ids[ray_start:ray_end][valid] = ids_cpu[valid]
        output_distances[ray_start:ray_end] = distances_cpu
    return output_ids, output_distances


def raycast_primitive_hits(
    scene: PrimitiveScene,
    origins: Any,
    directions: Any,
    *,
    device: str = "cpu",
    max_distance: Any | None = None,
    ignore_unit_ids: Any | None = None,
    ray_chunk: int = 4096,
    unit_chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest analytic primitive IDs and distances for a ray batch.

    ``-1``/``inf`` denotes no hit.  ``ignore_unit_ids`` is per-ray and is used
    to optionally skip the ray origin primitive. CPU and CUDA
    use the same box/cylinder intersection equations and deterministic ID tie
    break.
    """

    ray_origins = np.asarray(origins, dtype=np.float64)
    ray_directions = np.asarray(directions, dtype=np.float64)
    if ray_origins.ndim != 2 or ray_origins.shape[1] != 3 or ray_directions.shape != ray_origins.shape:
        raise ValueError("origins and directions must both have shape [R,3]")
    if ray_origins.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    lengths = np.linalg.norm(ray_directions, axis=1)
    if np.any(~np.isfinite(ray_origins)) or np.any(~np.isfinite(ray_directions)) or np.any(lengths <= 1.0e-12):
        raise ValueError("ray origins/directions must be finite and directions non-zero")
    ray_directions = ray_directions / lengths[:, None]
    limits = None if max_distance is None else np.asarray(max_distance, dtype=np.float64).reshape(-1)
    if limits is not None and (limits.shape != (ray_origins.shape[0],) or np.any(~np.isfinite(limits)) or np.any(limits <= 0.0)):
        raise ValueError("max_distance must be positive finite values with shape [R]")
    ignored = None if ignore_unit_ids is None else np.asarray(ignore_unit_ids, dtype=np.int64).reshape(-1)
    if ignored is not None and ignored.shape != (ray_origins.shape[0],):
        raise ValueError("ignore_unit_ids must have shape [R]")
    resolved = str(device).lower()
    if resolved == "auto":
        try:
            import torch

            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            resolved = "cpu"
    if resolved == "cuda":
        return _raycast_cuda(
            scene,
            ray_origins,
            ray_directions,
            max_distance=limits,
            ignore_unit_ids=ignored,
            candidate_matrix=None,
            use_grid=scene.unit_count > 64,
            ray_chunk=int(ray_chunk),
            unit_chunk=int(unit_chunk),
            device=resolved,
        )
    if resolved != "cpu":
        raise ValueError("device must be cpu, cuda, or auto")
    return _raycast_cpu(
        scene,
        ray_origins,
        ray_directions,
        max_distance=limits,
        ignore_unit_ids=ignored,
        candidate_matrix=None,
        use_grid=scene.unit_count > 64,
        ray_chunk=int(ray_chunk),
        unit_chunk=int(unit_chunk),
    )


def raycast_primitives(
    scene: PrimitiveScene,
    origins: Any,
    directions: Any,
    *,
    device: str = "cpu",
    max_distance: Any | None = None,
    ignore_unit_ids: Any | None = None,
) -> np.ndarray:
    """Convenience wrapper returning only the nearest unit IDs."""

    return raycast_primitive_hits(
        scene,
        origins,
        directions,
        device=device,
        max_distance=max_distance,
        ignore_unit_ids=ignore_unit_ids,
    )[0]
