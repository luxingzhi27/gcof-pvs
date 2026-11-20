"""GCOF-PVS V5 visibility loss and shared dual state.

The training loop processes one source scene per optimizer step.  The public
interfaces in this module therefore keep one scene's global ``G_s`` and
``W_s`` explicit instead of silently averaging minibatch denominators.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

ROBUST_BOUNDARY_STATE_SCHEMA = "gcof-pvs-v5-robust-boundary-state-v1"
SAFETY_BUDGET = 0.01
_LN2 = math.log(2.0)


def _float_tensor(
    value: Any,
    name: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.to(device=device) if device is not None else value
    else:
        result = torch.as_tensor(value, device=device)
    if not result.is_floating_point():
        result = result.float()
    if dtype is not None:
        result = result.to(dtype=dtype)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _zero_like(value: torch.Tensor) -> torch.Tensor:
    return value.sum() * 0.0


def _scalar_total(value: Any, name: str, *, positive: bool) -> float:
    tensor = _float_tensor(value, name, dtype=torch.float64).reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be a scalar")
    result = float(tensor.item())
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if not positive and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class SharedBoundaryRisks:
    """One source scene's efficiency objective and unified safety risk."""

    extra: torch.Tensor
    safety: torch.Tensor


def shared_boundary_risks(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    pose_scale: torch.Tensor | float,
    G_s: torch.Tensor | float,
    W_s: torch.Tensor | float,
    *,
    rho: float,
) -> SharedBoundaryRisks:
    """Estimate fixed-zero retention and blended miss risk with global denominators."""

    if not math.isfinite(float(rho)) or not 0.0 <= float(rho) <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    values = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
    if not values.is_floating_point():
        values = values.float()
    labels = _float_tensor(targets, "targets", device=values.device, dtype=values.dtype)
    visible = _float_tensor(weights, "weights", device=values.device, dtype=values.dtype)
    if labels.numel() != values.numel() or visible.numel() != values.numel():
        raise ValueError("logits, targets and weights must have equal item counts")
    labels = labels.reshape(-1)
    visible = visible.reshape(-1)
    if bool(((labels != 0.0) & (labels != 1.0)).any()):
        raise ValueError("targets must contain only zero and one")
    if bool((visible < 0.0).any()):
        raise ValueError("weights must be finite and non-negative")
    occurrence_total = _scalar_total(G_s, "G_s", positive=True)
    visual_total = _scalar_total(W_s, "W_s", positive=True)
    scale = _candidate_scale(pose_scale, values).reshape(-1)
    flat_logits = values.reshape(-1)
    positive = labels == 1.0
    negative = ~positive
    h_keep = F.softplus(flat_logits) / _LN2
    h_miss = F.softplus(-flat_logits) / _LN2
    extra = (h_keep[negative] * scale[negative]).sum() / occurrence_total
    mean_visual = visual_total / occurrence_total
    blended = float(rho) + (1.0 - float(rho)) * visible[positive] / mean_visual
    safety = (h_miss[positive] * blended * scale[positive]).sum() / occurrence_total
    if not bool(torch.isfinite(torch.stack([extra, safety])).all()):
        raise FloatingPointError("shared-boundary risks are non-finite")
    return SharedBoundaryRisks(extra=extra, safety=safety)


class RobustBoundaryState:
    """SmoothMax over running scene-domain risks with a shared multiplier."""

    def __init__(
        self,
        domain_ids: Sequence[Hashable],
        sample_probabilities: Mapping[Hashable, float],
        *,
        dual_lr: float,
        safety_budget: float = SAFETY_BUDGET,
        alpha: float = 16.0,
        ema_beta: float = 0.99,
        lambda_initial: float = 1.0,
        device: torch.device | str | None = None,
    ) -> None:
        self.domain_ids = tuple(domain_ids)
        if not self.domain_ids or len(set(self.domain_ids)) != len(self.domain_ids):
            raise ValueError("domain_ids must be non-empty and unique")
        self._domain_index = {domain: index for index, domain in enumerate(self.domain_ids)}
        for name, value, lower, upper in (
            ("dual_lr", dual_lr, 0.0, math.inf),
            ("safety_budget", safety_budget, 0.0, math.inf),
            ("alpha", alpha, 0.0, math.inf),
            ("ema_beta", ema_beta, 0.0, 1.0),
            ("lambda_initial", lambda_initial, 0.0, math.inf),
        ):
            if not math.isfinite(float(value)) or not lower <= float(value) < upper:
                raise ValueError(f"{name} is outside its valid range")
        self.dual_lr = float(dual_lr)
        self.safety_budget = float(safety_budget)
        self.alpha = float(alpha)
        self.ema_beta = float(ema_beta)
        self.device = torch.device("cpu" if device is None else device)
        probabilities = torch.as_tensor(
            [float(sample_probabilities.get(domain, -1.0)) for domain in self.domain_ids],
            dtype=torch.float64,
            device=self.device,
        )
        if not bool(torch.isfinite(probabilities).all()) or bool((probabilities <= 0.0).any()):
            raise ValueError("every robust domain must have a positive sampling probability")
        if abs(float(probabilities.sum().item()) - 1.0) > 1e-8:
            raise ValueError("robust domain sampling probabilities must sum to one")
        self.sample_probabilities = probabilities
        self.risk_ema = torch.ones(len(self.domain_ids), dtype=torch.float64, device=self.device)
        self.lambdas = torch.full(
            (1,), float(lambda_initial), dtype=torch.float64, device=self.device
        )

    def _index(self, domain_id: Hashable) -> int:
        try:
            return self._domain_index[domain_id]
        except KeyError as exc:
            raise KeyError(f"unknown robust boundary domain: {domain_id!r}") from exc

    def domain_weights(self) -> torch.Tensor:
        return torch.softmax(self.alpha * self.risk_ema, dim=0)

    def robust_risk(self) -> torch.Tensor:
        return torch.logsumexp(self.alpha * self.risk_ema, dim=0) / self.alpha - (
            math.log(len(self.domain_ids)) / self.alpha
        )

    def loss(self, domain_id: Hashable, risks: SharedBoundaryRisks) -> torch.Tensor:
        index = self._index(domain_id)
        target_probability = 1.0 / len(self.domain_ids)
        importance = target_probability / float(self.sample_probabilities[index].item())
        extra = risks.extra
        safety = risks.safety
        domain_weight = float(self.domain_weights()[index].item())
        risk_coefficient = importance * len(self.domain_ids) * domain_weight
        multiplier = self.lambdas[0].detach().to(extra)
        total = importance * extra + multiplier * (risk_coefficient * safety - self.safety_budget)
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("robust-boundary loss is non-finite")
        return total

    def update(self, domain_id: Hashable, risks: SharedBoundaryRisks) -> None:
        index = self._index(domain_id)
        value = float(risks.safety.detach().item())
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("robust-boundary safety risk must be finite and non-negative")
        with torch.no_grad():
            self.risk_ema[index].mul_(self.ema_beta).add_((1.0 - self.ema_beta) * value)
            self.lambdas[0].copy_(torch.clamp(
                self.lambdas[0] + self.dual_lr * (self.robust_risk() - self.safety_budget),
                min=0.0,
            ))

    def diagnostics(self, domain_id: Hashable) -> dict[str, float]:
        index = self._index(domain_id)
        target_probability = 1.0 / len(self.domain_ids)
        return {
            "robustRisk": float(self.robust_risk().item()),
            "domainRiskEma": float(self.risk_ema[index].item()),
            "domainWeight": float(self.domain_weights()[index].item()),
            "importanceWeight": target_probability / float(self.sample_probabilities[index].item()),
            "lambdaShared": float(self.lambdas[0].item()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": ROBUST_BOUNDARY_STATE_SCHEMA,
            "version": 1,
            "domain_ids": list(self.domain_ids),
            "mode": "smoothmax",
            "dual_lr": self.dual_lr,
            "safety_budget": self.safety_budget,
            "alpha": self.alpha,
            "ema_beta": self.ema_beta,
            "sample_probabilities": self.sample_probabilities.detach().cpu().clone(),
            "risk_ema": self.risk_ema.detach().cpu().clone(),
            "lambdas": self.lambdas.detach().cpu().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any], strict: bool = True) -> None:
        if state.get("schema") != ROBUST_BOUNDARY_STATE_SCHEMA or int(state.get("version", -1)) != 1:
            raise ValueError("unsupported robust-boundary checkpoint state")
        if strict and tuple(state.get("domain_ids", ())) != self.domain_ids:
            raise ValueError("robust-boundary checkpoint domains disagree")
        for name, expected in (
            ("mode", "smoothmax"),
            ("dual_lr", self.dual_lr),
            ("safety_budget", self.safety_budget),
            ("alpha", self.alpha),
            ("ema_beta", self.ema_beta),
        ):
            if state.get(name) != expected:
                raise ValueError(f"robust-boundary checkpoint {name} disagrees")
        probabilities = torch.as_tensor(state["sample_probabilities"], dtype=torch.float64)
        risk_ema = torch.as_tensor(state["risk_ema"], dtype=torch.float64)
        lambdas = torch.as_tensor(state["lambdas"], dtype=torch.float64)
        if probabilities.shape != self.sample_probabilities.shape or not torch.equal(
            probabilities, self.sample_probabilities.cpu()
        ):
            raise ValueError("robust-boundary sampling probabilities disagree")
        if risk_ema.shape != self.risk_ema.shape or lambdas.shape != self.lambdas.shape:
            raise ValueError("robust-boundary checkpoint tensor shapes disagree")
        if not bool(torch.isfinite(risk_ema).all() and torch.isfinite(lambdas).all()):
            raise ValueError("robust-boundary checkpoint contains non-finite state")
        if bool((risk_ema < 0.0).any()) or bool((lambdas < 0.0).any()):
            raise ValueError("robust-boundary checkpoint contains negative state")
        self.risk_ema.copy_(risk_ema.to(self.device))
        self.lambdas.copy_(lambdas.to(self.device))


def _candidate_scale(pose_scale: Any, logits: torch.Tensor) -> torch.Tensor:
    scale = _float_tensor(
        pose_scale,
        "pose_scale",
        device=logits.device,
        dtype=logits.dtype,
    )
    if scale.ndim == 0:
        scale = scale.expand_as(logits)
    elif scale.shape == logits.shape:
        pass
    elif logits.ndim == 2 and scale.numel() == logits.shape[0]:
        scale = scale.reshape(logits.shape[0], 1).expand_as(logits)
    elif scale.numel() == logits.numel():
        scale = scale.reshape_as(logits)
    else:
        raise ValueError("pose_scale must be scalar, pose-aligned, or candidate-aligned")
    if bool((scale < 0.0).any()):
        raise ValueError("pose_scale must be finite and non-negative")
    return scale.reshape(-1)


def _pose_ranges(
    pose_rows: Any,
    item_count: int,
    num_poses: int,
) -> list[tuple[int, int]]:
    if isinstance(pose_rows, torch.Tensor):
        rows = pose_rows.detach().cpu()
    else:
        rows = torch.as_tensor(pose_rows)
    if rows.ndim == 1:
        values = rows.to(dtype=torch.long).reshape(-1)
        if values.numel() == num_poses + 1 and int(values[0]) == 0 and int(values[-1]) == item_count:
            if bool((values[1:] < values[:-1]).any()):
                raise ValueError("pose offset rows must be non-decreasing")
            return [(int(values[i]), int(values[i + 1])) for i in range(num_poses)]
        if values.numel() != item_count:
            raise ValueError("pose_rows must be pose IDs, offsets, or [num_poses,2] ranges")
        if bool((values < 0).any()) or bool((values >= num_poses).any()):
            raise ValueError("pose IDs are outside [0, num_poses)")
        return [
            (
                int(torch.nonzero(values == pose, as_tuple=False).min().item())
                if bool((values == pose).any())
                else 0,
                int(torch.nonzero(values == pose, as_tuple=False).max().item()) + 1
                if bool((values == pose).any())
                else 0,
            )
            for pose in range(num_poses)
        ]
    if rows.ndim == 2 and tuple(rows.shape) == (num_poses, 2):
        ranges = rows.to(dtype=torch.long).tolist()
        result = [(int(start), int(end)) for start, end in ranges]
        if any(start < 0 or end < start or end > item_count for start, end in result):
            raise ValueError("pose ranges are outside the flattened batch")
        return result
    raise ValueError("pose_rows must be pose IDs, offsets, or [num_poses,2] ranges")


def pose_balanced_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pose_rows: Any,
    num_poses: int,
) -> torch.Tensor:
    """Average positive/negative-balanced BCE per pose.

    ``pose_rows`` may be one pose ID per flattened candidate, ``num_poses+1``
    offsets, or ``[num_poses, 2]`` half-open ranges.  A pose with no positive
    or no negative candidates uses the class that is present; empty poses are
    skipped.
    """
    if int(num_poses) <= 0:
        raise ValueError("num_poses must be positive")
    values = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
    if not values.is_floating_point():
        values = values.float()
    flat_logits = values.reshape(-1)
    labels = _float_tensor(targets, "targets", device=values.device, dtype=values.dtype).reshape(-1)
    if labels.numel() != flat_logits.numel():
        raise ValueError("logits and targets must align")
    if bool(((labels != 0.0) & (labels != 1.0)).any()):
        raise ValueError("targets must contain only zero and one")
    pose_losses: list[torch.Tensor] = []
    for start, end in _pose_ranges(pose_rows, flat_logits.numel(), int(num_poses)):
        if start == end:
            continue
        local_logits = flat_logits[start:end]
        local_labels = labels[start:end]
        positive = local_labels == 1.0
        negative = ~positive
        class_losses: list[torch.Tensor] = []
        if bool(positive.any()):
            class_losses.append(F.softplus(-local_logits[positive]).mean())
        if bool(negative.any()):
            class_losses.append(F.softplus(local_logits[negative]).mean())
        pose_losses.append(torch.stack(class_losses).mean())
    total = torch.stack(pose_losses).mean() if pose_losses else _zero_like(values)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("pose-balanced BCE is non-finite")
    return total
