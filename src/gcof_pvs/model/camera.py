"""Camera-relative query feature layout."""
from __future__ import annotations

import torch

from ._tensor import _finite_float

QUERY_GEOMETRY_DIM = 16


QUERY_GEOMETRY_CHANNELS = (
    "target_to_region_world_direction_xyz",
    "region_to_target_camera_right_up_forward",
    "log_distance_over_radius",
    "radius_over_distance_plus_radius",
    "normalized_region_half_axes",
    "tan_half_fov_xy",
    "region_type",
    "near_over_distance_plus_radius",
    "log_one_plus_far_over_distance_plus_radius",
)


def _as_query_geometry(query_geometry: torch.Tensor) -> torch.Tensor:
    query = _finite_float(query_geometry, "query_geometry")
    if query.ndim == 1:
        query = query.unsqueeze(0)
    if query.ndim != 2 or query.shape[1] != QUERY_GEOMETRY_DIM:
        raise ValueError(f"query_geometry must have shape [B, {QUERY_GEOMETRY_DIM}]")
    return query


def build_query_geometry(
    target_to_region_world_direction: torch.Tensor,
    camera_right_up_forward: torch.Tensor,
    distance: torch.Tensor,
    target_radius: torch.Tensor,
    region_half_axes: torch.Tensor,
    fov_tangent_xy: torch.Tensor,
    region_type: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
) -> torch.Tensor:
    """Build the exact V6.1 query geometry layout.

    ``target_to_region_world_direction`` is the target-to-region direction.
    The second channel group is computed from its reverse direction in the
    supplied camera ``[right, up, forward]`` basis.  FOV inputs are already
    ``tan(FOV_x/2), tan(FOV_y/2)``; the function does not substitute scene
    constants or a camera hash.
    """
    world = torch.as_tensor(target_to_region_world_direction).float()
    if world.ndim == 1:
        world = world.unsqueeze(0)
    if world.ndim != 2 or world.shape[1] != 3:
        raise ValueError("target_to_region_world_direction must have shape [B, 3]")
    batch = world.shape[0]
    if not bool(torch.isfinite(world).all()):
        raise ValueError("target_to_region_world_direction contains non-finite values")
    world_norm = torch.linalg.vector_norm(world, dim=-1, keepdim=True)
    eps = torch.finfo(world.dtype).eps
    world_unit = torch.where(
        world_norm > eps,
        world / world_norm.clamp_min(eps),
        torch.zeros_like(world),
    )

    basis = torch.as_tensor(camera_right_up_forward, device=world.device).float()
    if basis.ndim == 2 and basis.shape == (3, 3):
        basis = basis.unsqueeze(0)
    if basis.ndim != 3 or basis.shape[1:] != (3, 3) or basis.shape[0] not in (1, batch):
        raise ValueError("camera_right_up_forward must have shape [B, 3, 3]")
    if basis.shape[0] == 1 and batch != 1:
        basis = basis.expand(batch, -1, -1)
    if not bool(torch.isfinite(basis).all()):
        raise ValueError("camera_right_up_forward contains non-finite values")
    basis_norm = torch.linalg.vector_norm(basis, dim=-1, keepdim=True)
    if bool((basis_norm <= eps).any()):
        raise ValueError("camera right/up/forward axes must be non-zero")
    basis = basis / basis_norm
    reverse_camera_components = torch.einsum(
        "bi,bji->bj", -world_unit, basis
    )

    def rows(value: torch.Tensor, width: int, name: str) -> torch.Tensor:
        result = torch.as_tensor(value, device=world.device).float()
        if result.ndim == 0:
            result = result.reshape(1, 1)
        elif result.ndim == 1:
            if result.numel() == width:
                result = result.reshape(1, width)
            else:
                result = result.reshape(-1, 1)
        if result.ndim != 2 or result.shape[1] != width or result.shape[0] not in (1, batch):
            raise ValueError(f"{name} must have shape [{width}] or [B, {width}]")
        if result.shape[0] == 1 and batch != 1:
            result = result.expand(batch, -1)
        if not bool(torch.isfinite(result).all()):
            raise ValueError(f"{name} contains non-finite values")
        return result

    d = rows(distance, 1, "distance")
    radius = rows(target_radius, 1, "target_radius")
    axes = rows(region_half_axes, 3, "region_half_axes")
    fov = rows(fov_tangent_xy, 2, "fov_tangent_xy")
    region = rows(region_type, 1, "region_type")
    near_value = rows(near, 1, "near")
    far_value = rows(far, 1, "far")
    if bool((d < 0).any()) or bool((radius <= 0).any()):
        raise ValueError("distance must be non-negative and target_radius positive")
    if bool((axes < 0).any()) or bool((fov <= 0).any()):
        raise ValueError("region_half_axes must be non-negative and FOV tangents positive")
    if bool(((region != 0) & (region != 1)).any()):
        raise ValueError("region_type must be 0 for disk or 1 for oriented box")
    if bool((near_value < 0).any()) or bool((far_value < 0).any()):
        raise ValueError("near and far must be non-negative")

    denominator = d + radius
    axis_scale = denominator + axes.abs().amax(dim=-1, keepdim=True)
    normalized_axes = axes / axis_scale
    log_distance = torch.log1p(d / radius)
    radius_fraction = radius / denominator
    near_fraction = near_value / denominator
    log_far = torch.log1p(far_value / denominator)
    result = torch.cat(
        [
            world_unit,
            reverse_camera_components,
            log_distance,
            radius_fraction,
            normalized_axes,
            fov,
            region,
            near_fraction,
            log_far,
        ],
        dim=-1,
    )
    if result.shape[1] != QUERY_GEOMETRY_DIM:
        raise RuntimeError(f"query geometry packing produced width {result.shape[1]}")
    return _finite_float(result, "query_geometry")
