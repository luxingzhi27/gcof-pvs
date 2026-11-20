"""Compile geometry-only relations into per-direction occlusion responses."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from gcof_pvs.data.directions import icosahedron12_directions
from gcof_pvs.data.schemas import RELATION_SCHEMA
from ._tensor import _finite_float, _mlp
from .geometry_encoder import GEOMETRY_DIM
from .survival import SURVIVAL_PARAMETER_DIM, first_order_direction_basis

ANCHOR_COUNT = 12
TOP_K = 8
EDGE_FEATURE_DIM = 8
EDGE_HIDDEN_DIM = 64
MESSAGE_DIM = 32


def icosahedron_anchors(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the canonical ordered unit-norm icosahedron anchors."""
    return torch.as_tensor(
        icosahedron12_directions(dtype=np.float64),
        dtype=dtype,
        device=device,
    )


def fixed_direction_projection(
    anchors: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build the fixed Moore-Penrose projection ``[12, 4] -> [4, 12]``."""
    directions = (
        icosahedron_anchors(dtype=torch.float64, device=device)
        if anchors is None
        else torch.as_tensor(anchors, dtype=torch.float64, device=device)
    )
    if directions.shape != (ANCHOR_COUNT, 3):
        raise ValueError(f"anchors must have shape [{ANCHOR_COUNT}, 3]")
    basis = first_order_direction_basis(directions)
    projection = torch.linalg.pinv(basis)
    return projection.to(dtype=dtype)


def _metadata(relation: Any) -> Mapping[str, Any]:
    if isinstance(relation, Mapping):
        raw = relation.get("metadata", relation.get("relation_metadata", {}))
    else:
        raw = getattr(relation, "metadata", {})
    return raw if isinstance(raw, Mapping) else {}


def _metadata_value(metadata: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metadata:
            return metadata[name]
    return None


def _relation_value(relation: Any, *names: str) -> Any:
    if isinstance(relation, Mapping):
        for name in names:
            if name in relation:
                return relation[name]
        return None
    for name in names:
        if hasattr(relation, name):
            return getattr(relation, name)
    return None


def _validate_relation_provenance(relation: Any) -> None:
    metadata = _metadata(relation)
    schema = _relation_value(relation, "schema")
    if schema is None:
        schema = _metadata_value(metadata, "schema")
    if schema is not None and schema != RELATION_SCHEMA:
        raise ValueError(
            f"V5 requires geometry-only relation schema {RELATION_SCHEMA}"
        )
    uses_labels = _metadata_value(metadata, "usesVisibilityLabels", "uses_visibility_labels")
    if uses_labels is not False:
        raise ValueError("V5 geometry relation must declare usesVisibilityLabels=false")
    source = _metadata_value(metadata, "source", "relationSource", "relation_source")
    if source is not None and str(source).lower() not in {
        "geometry_only",
        "geometry-only",
        "aabb_orthographic_overlap",
    }:
        raise ValueError("V5 relation source must be geometry-only")
    relation_directions = np.asarray(metadata.get("anchorDirections"), dtype=np.float64)
    canonical_directions = icosahedron12_directions(dtype=np.float64)
    if relation_directions.shape != canonical_directions.shape or not bool(
        np.allclose(relation_directions, canonical_directions, atol=1.0e-7, rtol=0.0)
    ):
        raise ValueError("V5 relation anchorDirections disagree with the canonical ordered table")


def _relation_tensors(
    relation: Mapping[str, Any] | Any,
    *,
    num_instances: int,
    target_ids: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Normalize a full or target-sliced geometry proxy relation.

    Source IDs always index the global geometry table.  Dense relations may
    have either a full first dimension ``N`` or the selected first dimension
    ``T``.  Sparse target IDs are global unless metadata explicitly declares a
    selected/local ID space.
    """
    _validate_relation_provenance(relation)
    selected_targets = torch.as_tensor(target_ids, dtype=torch.long, device=device).reshape(-1)
    if (
        selected_targets.numel()
        and (
            bool((selected_targets < 0).any())
            or bool((selected_targets >= num_instances).any())
            or torch.unique(selected_targets).numel() != selected_targets.numel()
        )
    ):
        raise ValueError("target_ids must be unique global geometry-table IDs")
    target_count = selected_targets.numel()
    source_raw = _relation_value(relation, "source_ids", "sourceIds")
    edge_raw = _relation_value(relation, "edge_features", "edgeFeatures", "features")
    if source_raw is None or edge_raw is None:
        raise ValueError("geometry relation requires source_ids and edge_features")
    source = torch.as_tensor(source_raw, dtype=torch.long, device=device)
    edge = torch.as_tensor(edge_raw, dtype=torch.float32, device=device)
    valid_raw = _relation_value(relation, "valid_mask", "validMask", "mask")

    if source.ndim == 3:
        if source.shape[1] != ANCHOR_COUNT:
            raise ValueError(f"dense source_ids must have shape [N, {ANCHOR_COUNT}, K]")
        if source.shape[0] == num_instances:
            source = source[selected_targets]
            if edge.ndim != 4 or edge.shape[0] != num_instances:
                raise ValueError("full dense edge_features must have the geometry-table first dimension")
            edge = edge[selected_targets]
            if valid_raw is not None:
                valid_raw = torch.as_tensor(valid_raw, dtype=torch.bool, device=device)[selected_targets]
        elif source.shape[0] == target_count:
            if edge.ndim != 4 or tuple(edge.shape[:3]) != tuple(source.shape):
                raise ValueError("selected dense edge_features must align with source_ids")
        else:
            raise ValueError("dense relation must contain all targets or exactly selected targets")
        if edge.ndim != 4 or tuple(edge.shape[:3]) != tuple(source.shape) or edge.shape[3] != EDGE_FEATURE_DIM:
            raise ValueError("dense edge_features must have shape [N, 12, K, 8]")
        target = torch.arange(target_count, device=device)[:, None, None].expand_as(source)
        anchor = torch.arange(ANCHOR_COUNT, device=device)[None, :, None].expand_as(source)
        if valid_raw is None:
            valid = source >= 0
        else:
            valid = torch.as_tensor(valid_raw, dtype=torch.bool, device=device)
            if valid.shape != source.shape:
                raise ValueError("valid_mask must have the same shape as source_ids")
        source = source.reshape(-1)
        target = target.reshape(-1)
        anchor = anchor.reshape(-1)
        edge = edge.reshape(-1, EDGE_FEATURE_DIM)
        valid = valid.reshape(-1)
    elif source.ndim == 1:
        if edge.ndim != 2 or tuple(edge.shape) != (source.numel(), EDGE_FEATURE_DIM):
            raise ValueError("sparse edge_features must have shape [E, 8]")
        target_raw = _relation_value(relation, "target_ids", "targetIds")
        anchor_raw = _relation_value(relation, "anchor_ids", "anchorIds", "direction_ids")
        if target_raw is None or anchor_raw is None:
            raise ValueError("sparse relations require target_ids and anchor_ids")
        target_raw_tensor = torch.as_tensor(target_raw, dtype=torch.long, device=device).reshape(-1)
        anchor = torch.as_tensor(anchor_raw, dtype=torch.long, device=device).reshape(-1)
        if target_raw_tensor.shape != source.shape or anchor.shape != source.shape:
            raise ValueError("sparse relation index arrays must have equal length")
        if valid_raw is None:
            valid = torch.ones_like(source, dtype=torch.bool)
        else:
            valid = torch.as_tensor(valid_raw, dtype=torch.bool, device=device).reshape(-1)
            if valid.shape != source.shape:
                raise ValueError("valid_mask must align with sparse relation edges")
        metadata = _metadata(relation)
        target_space = str(
            _metadata_value(metadata, "targetIdSpace", "target_id_space") or "global"
        ).lower()
        if target_space in {"local", "selected", "slice"}:
            if bool((target_raw_tensor[valid] < 0).any()) or bool(
                (target_raw_tensor[valid] >= target_count).any()
            ):
                raise ValueError("local sparse target_ids are outside the selected target slice")
            target = torch.full_like(target_raw_tensor, -1)
            target[valid] = target_raw_tensor[valid]
            valid = valid & (target >= 0)
        else:
            if bool((target_raw_tensor[valid] < 0).any()) or bool(
                (target_raw_tensor[valid] >= num_instances).any()
            ):
                raise ValueError("global sparse target_ids are outside the geometry table")
            inverse = torch.full((num_instances,), -1, dtype=torch.long, device=device)
            inverse[selected_targets] = torch.arange(target_count, device=device)
            target = inverse[target_raw_tensor.clamp_min(0)]
            valid = valid & (target >= 0)
    else:
        raise ValueError("source_ids must be sparse [E] or dense [N, 12, K]")

    if not bool(torch.isfinite(edge).all()):
        raise ValueError("edge_features contain non-finite values")
    if bool((target[valid] < 0).any()) or bool((target[valid] >= target_count).any()):
        raise ValueError("target_ids are outside the geometry table")
    if bool((anchor[valid] < 0).any()) or bool((anchor[valid] >= ANCHOR_COUNT).any()):
        raise ValueError("anchor_ids are outside the twelve fixed anchors")
    if bool((source[valid] < 0).any()) or bool((source[valid] >= num_instances).any()):
        raise ValueError("source_ids are outside the geometry table")
    return {
        "source_ids": source,
        "target_ids": target,
        "anchor_ids": anchor,
        "edge_features": edge,
        "valid_mask": valid,
    }


def _segment_softmax(logits: torch.Tensor, groups: torch.Tensor, group_count: int) -> torch.Tensor:
    if logits.ndim != 1 or groups.ndim != 1 or logits.numel() != groups.numel():
        raise ValueError("segment attention inputs must be aligned vectors")
    if logits.numel() == 0:
        return logits.new_empty((0,))
    maxima = torch.full(
        (group_count,), -torch.inf, dtype=logits.dtype, device=logits.device
    )
    maxima.scatter_reduce_(0, groups, logits, reduce="amax", include_self=True)
    exponent = torch.exp(logits - maxima[groups])
    denominator = logits.new_zeros((group_count,))
    denominator.index_add_(0, groups, exponent)
    return exponent / denominator[groups].clamp_min(torch.finfo(logits.dtype).tiny)


class SingleLayerRelationFieldCompiler(nn.Module):
    """Compile one geometry-only proxy graph into a structured directional field."""

    def __init__(self) -> None:
        super().__init__()
        self.edge_message = _mlp(GEOMETRY_DIM * 2 + EDGE_FEATURE_DIM, EDGE_HIDDEN_DIM, MESSAGE_DIM)
        self.attention_logit = nn.Linear(MESSAGE_DIM, 1)
        self.base_head = _mlp(GEOMETRY_DIM + 3, MESSAGE_DIM, SURVIVAL_PARAMETER_DIM)
        self.delta_head = _mlp(MESSAGE_DIM + 2 + 3, MESSAGE_DIM, SURVIVAL_PARAMETER_DIM)
        anchors = icosahedron_anchors()
        self.register_buffer("anchors", anchors, persistent=True)
        self.register_buffer(
            "direction_projection",
            fixed_direction_projection(anchors),
            persistent=True,
        )

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def compile_anchor_responses(
        self,
        geometry: torch.Tensor,
        relation: Mapping[str, Any] | Any | None = None,
        target_ids: torch.Tensor | None = None,
        anchor_directions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``q[T,12,7]`` for all or selected global target rows.

        ``geometry`` remains the global ``[N,32]`` table so source descriptors
        can be gathered by global ID.  Only selected targets allocate the
        anchor pooling tensors, which avoids an ``N x 12`` workspace for a
        large scene when a step touches a small target batch.
        """
        z = _finite_float(geometry, "geometry")
        if z.ndim != 2 or z.shape[1] != GEOMETRY_DIM:
            raise ValueError(f"geometry must have shape [N, {GEOMETRY_DIM}]")
        device = z.device
        if target_ids is None:
            selected_target_ids = torch.arange(z.shape[0], dtype=torch.long, device=device)
        else:
            selected_target_ids = torch.as_tensor(
                target_ids, dtype=torch.long, device=device
            ).reshape(-1)
        if (
            selected_target_ids.numel()
            and (
                bool((selected_target_ids < 0).any())
                or bool((selected_target_ids >= z.shape[0]).any())
                or torch.unique(selected_target_ids).numel() != selected_target_ids.numel()
            )
        ):
            raise ValueError("target_ids must be unique global geometry-table IDs")
        selected_geometry = z[selected_target_ids]
        count = selected_target_ids.numel()
        anchors = self.anchors
        if anchor_directions is not None:
            anchors = _finite_float(anchor_directions, "anchor_directions").to(device)
            if anchors.shape != (ANCHOR_COUNT, 3):
                raise ValueError(f"anchor_directions must have shape [{ANCHOR_COUNT}, 3]")
            anchors = F.normalize(anchors, dim=-1)
        group_count = count * ANCHOR_COUNT
        pooled = selected_geometry.new_zeros((group_count, MESSAGE_DIM))
        neighbor_count = selected_geometry.new_zeros((group_count,))
        overlap_sum = selected_geometry.new_zeros((group_count,))
        attention_grid = selected_geometry.new_zeros((group_count,))
        source_ids = selected_target_ids.new_empty((0,))
        local_target_ids = selected_target_ids.new_empty((0,))
        anchor_ids = selected_target_ids.new_empty((0,))
        if relation is not None:
            tensors = _relation_tensors(
                relation,
                num_instances=z.shape[0],
                target_ids=selected_target_ids,
                device=device,
            )
            valid = tensors["valid_mask"]
            source_ids = tensors["source_ids"][valid]
            local_target_ids = tensors["target_ids"][valid]
            anchor_ids = tensors["anchor_ids"][valid]
            edge = tensors["edge_features"][valid]
            if source_ids.numel():
                messages = self.edge_message(
                    torch.cat(
                        [
                            selected_geometry[local_target_ids],
                            z[source_ids],
                            edge,
                        ],
                        dim=-1,
                    )
                )
                logits = self.attention_logit(messages).squeeze(-1)
                groups = local_target_ids * ANCHOR_COUNT + anchor_ids
                weights = _segment_softmax(logits, groups, group_count)
                pooled.index_add_(0, groups, weights[:, None] * messages)
                neighbor_count.index_add_(0, groups, torch.ones_like(logits))
                overlap_sum.index_add_(0, groups, edge[:, 5].clamp_min(0.0))
                attention_grid.index_add_(0, groups, weights)
            else:
                messages = z.new_empty((0, MESSAGE_DIM))
                weights = z.new_empty((0,))
        else:
            messages = z.new_empty((0, MESSAGE_DIM))
            weights = z.new_empty((0,))

        h = pooled.reshape(count, ANCHOR_COUNT, MESSAGE_DIM)
        counts = neighbor_count.reshape(count, ANCHOR_COUNT)
        overlaps = overlap_sum.reshape(count, ANCHOR_COUNT)
        base_input = torch.cat(
            [
                selected_geometry[:, None, :].expand(-1, ANCHOR_COUNT, -1),
                anchors[None, :, :].expand(count, -1, -1),
            ],
            dim=-1,
        )
        base = self.base_head(base_input)
        delta_input = torch.cat(
            [
                h,
                torch.log1p(counts).unsqueeze(-1),
                overlaps.unsqueeze(-1),
                anchors[None, :, :].expand(count, -1, -1),
            ],
            dim=-1,
        )
        delta = self.delta_head(delta_input)
        has_neighbor = (counts > 0).unsqueeze(-1).to(delta.dtype)
        responses = base + has_neighbor * delta
        diagnostics = {
            "anchor_responses": responses,
            "anchor_messages": h,
            "neighbor_count": counts,
            "overlap_sum": overlaps,
            "attention_weights": weights,
            "attention_grid_sum": attention_grid.reshape(count, ANCHOR_COUNT),
            "source_ids": source_ids,
            "target_ids": selected_target_ids,
            "local_target_ids": local_target_ids,
            "anchor_ids": anchor_ids,
        }
        return responses, diagnostics

    def project_anchor_responses(
        self,
        responses: torch.Tensor,
        anchor_directions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = _finite_float(responses, "anchor_responses")
        if values.ndim != 3 or tuple(values.shape[1:]) != (ANCHOR_COUNT, SURVIVAL_PARAMETER_DIM):
            raise ValueError(f"anchor_responses must have shape [N, {ANCHOR_COUNT}, 7]")
        projection = self.direction_projection.to(values)
        if anchor_directions is not None:
            projection = fixed_direction_projection(
                anchor_directions,
                dtype=values.dtype,
                device=values.device,
            )
        field = torch.einsum("rk,nkp->nrp", projection, values)
        if not bool(torch.isfinite(field).all()):
            raise FloatingPointError("directional field projection produced non-finite values")
        return field

    def forward(
        self,
        geometry: torch.Tensor,
        relation: Mapping[str, Any] | Any | None = None,
        *,
        target_ids: torch.Tensor | None = None,
        anchor_directions: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        responses, diagnostics = self.compile_anchor_responses(
            geometry, relation, target_ids, anchor_directions
        )
        field = self.project_anchor_responses(responses, anchor_directions)
        if return_diagnostics:
            return {"field": field, **diagnostics}
        return field
