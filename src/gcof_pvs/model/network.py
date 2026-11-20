"""GCOF-PVS model: scene compilation and camera-relative visibility query."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from gcof_pvs.protocol import MODEL_VARIANTS
from ._tensor import _finite_float, _mlp
from .camera import QUERY_GEOMETRY_DIM, _as_query_geometry
from .geometry_encoder import GEOMETRY_DIM, LocalSurfaceGeometryEncoder, POINT_COUNT
from .relation import ANCHOR_COUNT, EDGE_FEATURE_DIM, TOP_K, SingleLayerRelationFieldCompiler
from .survival import (
    FIELD_RANK, SUPPORT_POINT_COUNT, SURVIVAL_PARAMETER_DIM,
    DoubleTruncatedLogisticSurvival, region_survival_statistics,
)

MODEL_SCHEMA = "gcof-pvs-v5"
FIELD_SHAPE = (FIELD_RANK, SURVIVAL_PARAMETER_DIM)
FIELD_DIM = FIELD_RANK * SURVIVAL_PARAMETER_DIM
REGION_STATS_DIM = 4
FULL_HEAD_INPUT_DIM = GEOMETRY_DIM + REGION_STATS_DIM + QUERY_GEOMETRY_DIM
GENERIC_HEAD_INPUT_DIM = GEOMETRY_DIM + FIELD_DIM + QUERY_GEOMETRY_DIM
FULL_HEAD_HIDDEN_DIM = 32
GENERIC_HEAD_HIDDEN_DIM = 22


def _capacity_matched_geometry_hidden(relation_parameters: int) -> int:
    """Match the Full relation compiler with a two-layer geometry MLP."""
    # Parameters for Linear(32,h), Linear(h,28), including biases.
    width = round((relation_parameters - FIELD_DIM) / (GEOMETRY_DIM + 1 + FIELD_DIM))
    candidates = range(max(1, width - 4), width + 5)
    return min(
        candidates,
        key=lambda candidate: abs(
            (GEOMETRY_DIM + FIELD_DIM + 1) * candidate + FIELD_DIM
            - relation_parameters
        ),
    )


def _select_rows(
    values: torch.Tensor,
    batch: int,
    instance_ids: torch.Tensor | None,
    name: str,
) -> torch.Tensor:
    tensor = values
    if instance_ids is not None:
        ids = torch.as_tensor(instance_ids, dtype=torch.long, device=tensor.device).reshape(-1)
        if ids.numel() != batch or bool((ids < 0).any()) or bool((ids >= tensor.shape[0]).any()):
            raise ValueError(f"{name} instance_ids do not select {batch} valid rows")
        return tensor[ids]
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch, *tensor.shape[1:])
    raise ValueError(f"{name} has {tensor.shape[0]} rows but query batch is {batch}")


class GCOFPVSV5(nn.Module):
    """Full GCOF-PVS V5 and its two registered representation controls."""

    def __init__(self, variant: str = "FULL") -> None:
        super().__init__()
        normalized_variant = str(variant).upper()
        if normalized_variant not in MODEL_VARIANTS:
            raise ValueError(f"variant must be one of {MODEL_VARIANTS}")
        self.variant = normalized_variant
        self.geometry_encoder = LocalSurfaceGeometryEncoder()
        self.survival = DoubleTruncatedLogisticSurvival()

        if normalized_variant in {"FULL", "GENERIC_RELATION_28"}:
            self.relation_compiler: SingleLayerRelationFieldCompiler | None = (
                SingleLayerRelationFieldCompiler()
            )
        else:
            self.relation_compiler = None

        if normalized_variant == "GEOMETRY_FIELD":
            assert self.relation_compiler is None
            matched_width = _capacity_matched_geometry_hidden(
                SingleLayerRelationFieldCompiler().parameter_count
            )
            self.geometry_field = _mlp(GEOMETRY_DIM, matched_width, FIELD_DIM)
            self.geometry_field_hidden_dim = matched_width
        else:
            self.geometry_field = None
            self.geometry_field_hidden_dim = None

        if normalized_variant == "GENERIC_RELATION_28":
            # The unrestricted control receives the complete ordered 12x7
            # evidence. Its runtime latent remains exactly 28 values.
            self.generic_projection = nn.Linear(
                ANCHOR_COUNT * SURVIVAL_PARAMETER_DIM, FIELD_DIM, bias=False
            )
            head_input_dim = GENERIC_HEAD_INPUT_DIM
            head_hidden_dim = GENERIC_HEAD_HIDDEN_DIM
        else:
            self.generic_projection = None
            head_input_dim = FULL_HEAD_INPUT_DIM
            head_hidden_dim = FULL_HEAD_HIDDEN_DIM
        self.head_input_dim = head_input_dim
        self.visibility_head = nn.Sequential(
            nn.Linear(head_input_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Linear(head_hidden_dim, 1),
        )

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    @property
    def runtime_field_shape(self) -> tuple[int, int] | None:
        return FIELD_SHAPE if self.variant != "GENERIC_RELATION_28" else None

    @property
    def config(self) -> dict[str, Any]:
        return {
            "schema": MODEL_SCHEMA,
            "variant": self.variant,
            "pointsPerUnit": POINT_COUNT,
            "geometryDim": GEOMETRY_DIM,
            "anchors": ANCHOR_COUNT,
            "relationTopK": TOP_K,
            "edgeFeatureDim": EDGE_FEATURE_DIM,
            "fieldShape": list(FIELD_SHAPE),
            "supportPointCount": SUPPORT_POINT_COUNT,
            "queryGeometryDim": QUERY_GEOMETRY_DIM,
            "headShape": [self.head_input_dim, self.visibility_head[0].out_features, 1],
        }

    def encode_geometry(
        self,
        points: torch.Tensor,
        size_ratios: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.geometry_encoder(points, size_ratios)

    def _geometry_table(
        self,
        geometry_or_points: torch.Tensor,
        size_ratios: torch.Tensor | None,
    ) -> torch.Tensor:
        values = torch.as_tensor(geometry_or_points)
        if values.ndim == 2 and values.shape[1] == GEOMETRY_DIM:
            return _finite_float(values, "geometry")
        return self.encode_geometry(values, size_ratios)

    def compile_outputs(
        self,
        geometry: torch.Tensor,
        relation: Mapping[str, Any] | Any | None = None,
        target_ids: torch.Tensor | None = None,
        anchor_directions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z = self._geometry_table(geometry, None)
        selected_ids = (
            torch.arange(z.shape[0], dtype=torch.long, device=z.device)
            if target_ids is None
            else torch.as_tensor(target_ids, dtype=torch.long, device=z.device).reshape(-1)
        )
        if self.variant == "GEOMETRY_FIELD":
            if relation is not None:
                raise ValueError("GEOMETRY_FIELD does not consume relation input")
            assert self.geometry_field is not None
            selected_z = _select_rows(z, selected_ids.numel(), selected_ids, "geometry")
            field = self.geometry_field(selected_z).reshape(-1, *FIELD_SHAPE)
            return {"geometry": selected_z, "field": field, "target_ids": selected_ids}

        assert self.relation_compiler is not None
        responses, diagnostics = self.relation_compiler.compile_anchor_responses(
            z, relation, selected_ids, anchor_directions
        )
        if self.variant == "FULL":
            field = self.relation_compiler.project_anchor_responses(
                responses, anchor_directions
            )
            return {"geometry": z[selected_ids], "field": field, **diagnostics}

        assert self.generic_projection is not None
        latent = self.generic_projection(responses.flatten(start_dim=1))
        return {"geometry": z[selected_ids], "generic_latent": latent, **diagnostics}

    def compile_field(
        self,
        geometry: torch.Tensor,
        relation: Mapping[str, Any] | Any | None = None,
        target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compile Full/GEOMETRY_FIELD coefficients; reject generic latent use."""
        outputs = self.compile_outputs(geometry, relation, target_ids)
        if "field" not in outputs:
            raise ValueError("GENERIC_RELATION_28 does not expose an analytic survival field")
        return outputs["field"]

    def relation_latent(
        self,
        geometry: torch.Tensor,
        relation: Mapping[str, Any] | Any | None = None,
        target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.variant != "GENERIC_RELATION_28":
            raise ValueError("relation_latent is only defined for GENERIC_RELATION_28")
        return self.compile_outputs(geometry, relation, target_ids)["generic_latent"]

    def region_statistics(
        self,
        field: torch.Tensor,
        target_center: torch.Tensor,
        support_points: torch.Tensor,
        target_radius: torch.Tensor | float,
    ) -> torch.Tensor:
        return region_survival_statistics(field, target_center, support_points, target_radius)

    def query(
        self,
        geometry: torch.Tensor,
        query_geometry: torch.Tensor,
        relation: Mapping[str, Any] | Any | None = None,
        *,
        instance_ids: torch.Tensor | None = None,
        field: torch.Tensor | None = None,
        generic_latent: torch.Tensor | None = None,
        field_stats: torch.Tensor | None = None,
        support_points: torch.Tensor | None = None,
        target_center: torch.Tensor | None = None,
        target_radius: torch.Tensor | float | None = None,
        size_ratios: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        query = _as_query_geometry(query_geometry)
        batch = query.shape[0]
        z = self._geometry_table(geometry, size_ratios)
        selected_z = _select_rows(z, batch, instance_ids, "geometry")

        outputs: dict[str, torch.Tensor]
        if field is not None or generic_latent is not None:
            outputs = {"geometry": z}
            if field is not None:
                values = _finite_float(field, "field")
                if values.ndim == 2:
                    values = values.reshape(-1, *FIELD_SHAPE)
                if values.ndim != 3 or tuple(values.shape[1:]) != FIELD_SHAPE:
                    raise ValueError(f"field must have shape [N, {FIELD_RANK}, {SURVIVAL_PARAMETER_DIM}]")
                outputs["field"] = values
            if generic_latent is not None:
                latent = _finite_float(generic_latent, "generic_latent")
                if latent.ndim == 1:
                    latent = latent.unsqueeze(0)
                if latent.ndim != 2 or latent.shape[1] != FIELD_DIM:
                    raise ValueError(f"generic_latent must have shape [N, {FIELD_DIM}]")
                outputs["generic_latent"] = latent
        else:
            outputs = self.compile_outputs(z, relation)

        if self.variant == "GENERIC_RELATION_28":
            if "generic_latent" not in outputs:
                raise ValueError("GENERIC_RELATION_28 requires generic_latent or relation input")
            latent = _select_rows(outputs["generic_latent"], batch, instance_ids, "generic_latent")
            head_input = torch.cat([selected_z, latent, query], dim=-1)
            stats = None
        else:
            if "field" not in outputs:
                raise ValueError("structured variants require a 4x7 field")
            selected_field = _select_rows(outputs["field"], batch, instance_ids, "field")
            if field_stats is not None:
                stats = _finite_float(field_stats, "field_stats")
                if stats.ndim == 1:
                    stats = stats.unsqueeze(0)
                if stats.shape != (batch, REGION_STATS_DIM):
                    raise ValueError(f"field_stats must have shape [B, {REGION_STATS_DIM}]")
            else:
                if support_points is None or target_center is None or target_radius is None:
                    raise ValueError(
                        "structured variants require field_stats or nine support points, target_center, and target_radius"
                    )
                centers = _finite_float(target_center, "target_center")
                points = _finite_float(support_points, "support_points")
                centers = _select_rows(centers if centers.ndim == 2 else centers.unsqueeze(0), batch, instance_ids, "target_center")
                points = _select_rows(points if points.ndim == 3 else points.unsqueeze(0), batch, instance_ids, "support_points")
                radii = torch.as_tensor(target_radius).float()
                if radii.ndim == 0:
                    radii = radii.reshape(1)
                if radii.ndim == 2 and radii.shape[1] == 1:
                    radii = radii[:, 0]
                radii = _select_rows(radii.reshape(-1, 1), batch, instance_ids, "target_radius").reshape(-1)
                stats = region_survival_statistics(selected_field, centers, points, radii)
            head_input = torch.cat([selected_z, stats, query], dim=-1)

        if head_input.shape[1] != self.head_input_dim:
            raise RuntimeError(f"visibility head expected {self.head_input_dim} inputs")
        logits = self.visibility_head(head_input)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("V5 visibility head produced non-finite logits")
        if return_details:
            result: dict[str, torch.Tensor] = {
                "logits": logits,
                "geometry": selected_z,
                "query_geometry": query,
                "query_input": head_input,
            }
            if stats is not None:
                result["field_stats"] = stats
            if "field" in outputs and self.variant != "GENERIC_RELATION_28":
                result["field"] = _select_rows(outputs["field"], batch, instance_ids, "field")
            if "generic_latent" in outputs:
                result["generic_latent"] = _select_rows(
                    outputs["generic_latent"], batch, instance_ids, "generic_latent"
                )
            return result
        return logits

    def forward(
        self,
        geometry: torch.Tensor,
        query_geometry: torch.Tensor,
        relation: Mapping[str, Any] | Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        return self.query(geometry, query_geometry, relation, **kwargs)
