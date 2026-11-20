#!/usr/bin/env python3
"""One optimization step, including bounded-memory geometry backpropagation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import torch

from gcof_pvs.model.network import GCOFPVSV5
from .objective import RobustBoundaryState, pose_balanced_bce, shared_boundary_risks
from gcof_pvs.data.prepared import PoseBatch, V5SceneTrainingData
from gcof_pvs.protocol import ROBUST_BOUNDARY_OBJECTIVE


@dataclass(frozen=True)
class TrainStepResult:
    scene_id: str
    dual_group_id: str
    loss: float
    risk_extra: float
    risk_safety: float
    risk_robust: float | None
    domain_risk_ema: float | None
    domain_weight: float | None
    importance_weight: float | None
    lambda_shared: float | None
    candidate_count: int
    target_unit_count: int
    geometry_unit_count: int


def yaw_rotation(quarter_turns: int, device: torch.device) -> torch.Tensor:
    angle = (int(quarter_turns) % 4) * (math.pi / 2.0)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=torch.float32,
        device=device,
    )


def _rotate_surface(
    points: torch.Tensor,
    size_ratios: torch.Tensor,
    rotation: torch.Tensor,
    quarter_turns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = points.clone()
    result[..., :3] = result[..., :3] @ rotation.T
    result[..., 3:6] = result[..., 3:6] @ rotation.T
    ratios = size_ratios
    if int(quarter_turns) % 2:
        ratios = size_ratios[:, [2, 1, 0]]
    return result, ratios


def _torch_relation(
    relation: Mapping[str, Any],
    rotation: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    features = torch.as_tensor(relation["edge_features"], dtype=torch.float32, device=device).clone()
    features[..., :3] = features[..., :3] @ rotation.T
    return {
        "source_ids": torch.as_tensor(relation["source_ids"], dtype=torch.long, device=device),
        "valid_mask": torch.as_tensor(relation["valid_mask"], dtype=torch.bool, device=device),
        "edge_features": features,
        "metadata": relation["metadata"],
    }


def _encode_without_graph(
    model: GCOFPVSV5,
    scene: V5SceneTrainingData,
    unit_ids: np.ndarray,
    rotation: torch.Tensor,
    quarter_turns: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, unit_ids.size, chunk_size):
            ids = unit_ids[start : start + chunk_size]
            points, ratios = scene.geometry_tensors(ids, device)
            points, ratios = _rotate_surface(points, ratios, rotation, quarter_turns)
            rows.append(model.encode_geometry(points, ratios))
    return torch.cat(rows, dim=0)


def _recompute_geometry_backward(
    model: GCOFPVSV5,
    scene: V5SceneTrainingData,
    unit_ids: np.ndarray,
    cached_gradient: torch.Tensor,
    rotation: torch.Tensor,
    quarter_turns: int,
    chunk_size: int,
    device: torch.device,
) -> None:
    for start in range(0, unit_ids.size, chunk_size):
        ids = unit_ids[start : start + chunk_size]
        points, ratios = scene.geometry_tensors(ids, device)
        points, ratios = _rotate_surface(points, ratios, rotation, quarter_turns)
        encoded = model.encode_geometry(points, ratios)
        torch.autograd.backward(
            encoded,
            grad_tensors=cached_gradient[start : start + ids.size],
        )


def _rotated_pose_tensors(
    batch: PoseBatch,
    rotation: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    query = torch.from_numpy(batch.query_geometry).to(device).clone()
    query[:, :3] = query[:, :3] @ rotation.T
    centers = torch.from_numpy(batch.target_centers).to(device) @ rotation.T
    supports = torch.from_numpy(batch.support_points).to(device) @ rotation.T
    return {
        "candidate_ids": torch.from_numpy(batch.candidate_ids).long().to(device),
        "pose_rows": torch.from_numpy(batch.pose_rows).long().to(device),
        "targets": torch.from_numpy(batch.targets).float().to(device),
        "weights": torch.from_numpy(batch.visible_weights).float().to(device),
        "query": query,
        "centers": centers,
        "supports": supports,
        "radii": torch.from_numpy(batch.target_radii).float().to(device),
    }


def train_step(
    *,
    model: GCOFPVSV5,
    scene: V5SceneTrainingData,
    pose_batch: PoseBatch,
    optimizer: torch.optim.Optimizer,
    dual_state: RobustBoundaryState | None,
    dual_group_id: str,
    device: torch.device,
    quarter_turns: int,
    geometry_chunk_size: int = 512,
    objective: str = ROBUST_BOUNDARY_OBJECTIVE,
    boundary_rho: float = 0.1,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> TrainStepResult:
    """Run one exact one-scene optimizer step."""

    if geometry_chunk_size <= 0:
        raise ValueError("geometry_chunk_size must be positive")
    objective_name = str(objective).upper()
    if objective_name not in {ROBUST_BOUNDARY_OBJECTIVE, "PBCE_OBJECTIVE"}:
        raise ValueError(f"unsupported objective: {objective_name}")
    if objective_name == "PBCE_OBJECTIVE" and model.variant != "FULL":
        raise ValueError("PBCE_OBJECTIVE uses the FULL architecture")
    if (dual_state is None) != (objective_name == "PBCE_OBJECTIVE"):
        raise ValueError("only the shared-boundary objective uses a dual variable")

    target_ids = np.unique(pose_batch.candidate_ids.astype(np.int64, copy=False))
    if model.variant == "GEOMETRY_FIELD":
        dependency_ids = target_ids
        relation = None
    else:
        dependency_ids = scene.geometry_dependency_ids(target_ids)
        relation = scene.relation_slice(target_ids)

    rotation = yaw_rotation(quarter_turns, device)
    z_cached = _encode_without_graph(
        model,
        scene,
        dependency_ids,
        rotation,
        quarter_turns,
        geometry_chunk_size,
        device,
    )
    z_leaf = z_cached.detach().requires_grad_(True)
    geometry_table = z_leaf.new_zeros((scene.num_units, z_leaf.shape[1])).index_copy(
        0,
        torch.from_numpy(dependency_ids).long().to(device),
        z_leaf,
    )
    relation_tensors = None if relation is None else _torch_relation(relation, rotation, device)
    anchor_directions = None
    if model.relation_compiler is not None:
        anchor_directions = model.relation_compiler.anchors @ rotation.T
    targets_tensor = torch.from_numpy(target_ids).long().to(device)
    compiled = model.compile_outputs(
        geometry_table,
        relation_tensors,
        targets_tensor,
        anchor_directions,
    )
    pose = _rotated_pose_tensors(pose_batch, rotation, device)
    candidate_local = torch.searchsorted(targets_tensor, pose["candidate_ids"])
    geometry_candidates = compiled["geometry"][candidate_local]

    if model.variant == "GENERIC_RELATION_28":
        latent = compiled["generic_latent"][candidate_local]
        head_input = torch.cat([geometry_candidates, latent, pose["query"]], dim=-1)
    else:
        fields = compiled["field"]
        candidate_fields = fields[candidate_local]
        stats = model.region_statistics(
            candidate_fields,
            pose["centers"],
            pose["supports"],
            pose["radii"],
        )
        head_input = torch.cat([geometry_candidates, stats, pose["query"]], dim=-1)
    logits = model.visibility_head(head_input).reshape(-1)
    risks = shared_boundary_risks(
        logits,
        pose["targets"],
        pose["weights"],
        float(pose_batch.denominators.eligible_pose_count) / float(pose_batch.pose_ids.size),
        pose_batch.denominators.visible_occurrences,
        pose_batch.denominators.visible_weight_sum,
        rho=float(boundary_rho),
    )
    if dual_state is None:
        loss = pose_balanced_bce(
            logits, pose["targets"], pose["pose_rows"], int(pose_batch.pose_ids.size)
        )
    else:
        loss = dual_state.loss(dual_group_id, risks)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if z_leaf.grad is None or not bool(torch.isfinite(z_leaf.grad).all()):
        raise FloatingPointError("V5 cached geometry gradient is missing or non-finite")
    _recompute_geometry_backward(
        model,
        scene,
        dependency_ids,
        z_leaf.grad.detach(),
        rotation,
        quarter_turns,
        geometry_chunk_size,
        device,
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    diagnostics = {}
    if dual_state is not None:
        dual_state.update(dual_group_id, risks)
        diagnostics = dual_state.diagnostics(dual_group_id)
    return TrainStepResult(
        scene_id=scene.scene_id,
        dual_group_id=str(dual_group_id),
        loss=float(loss.detach().item()),
        risk_extra=float(risks.extra.detach().item()),
        risk_safety=float(risks.safety.detach().item()),
        risk_robust=diagnostics.get("robustRisk"),
        domain_risk_ema=diagnostics.get("domainRiskEma"),
        domain_weight=diagnostics.get("domainWeight"),
        importance_weight=diagnostics.get("importanceWeight"),
        lambda_shared=diagnostics.get("lambdaShared"),
        candidate_count=int(pose_batch.candidate_ids.size),
        target_unit_count=int(target_ids.size),
        geometry_unit_count=int(dependency_ids.size),
    )
