"""Streaming pose-set metrics for the columnar GCOF-PVS V5 benchmark."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from gcof_pvs.evaluation.culling import candidate_normalized_occlusion_recall

from .score_bundle import PoseScores


def average_precision(scores: np.ndarray, targets: np.ndarray) -> float | None:
    """Compute non-interpolated AP after merging equal score values."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(targets, dtype=np.uint8).reshape(-1)
    if values.size != labels.size:
        raise ValueError("scores and targets must have equal lengths")
    if not bool(np.isfinite(values).all()) or bool(np.any(labels > 1)):
        raise ValueError("AP inputs are invalid")
    positive_count = int(labels.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-values, kind="mergesort")
    ordered_labels = labels[order].astype(np.float64, copy=False)
    ordered_scores = values[order]
    starts = np.concatenate(
        [
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]).astype(np.int64) + 1,
        ]
    )
    group_positive = np.add.reduceat(ordered_labels, starts)
    group_total = np.diff(np.concatenate([starts, np.asarray([ordered_labels.size], dtype=np.int64)])).astype(np.float64)
    cumulative_positive = np.cumsum(group_positive)
    cumulative_total = np.cumsum(group_total)
    return float(np.sum((group_positive / positive_count) * (cumulative_positive / np.maximum(cumulative_total, 1.0))))


def weighted_recall_lower_confidence_bound(
    weighted_tp: np.ndarray,
    weighted_gt: np.ndarray,
    replicates: int = 10000,
    seed: int = 0,
) -> float:
    """Return the percentile lower bound used by calibration safety gates.

    The resampling unit is a pose.  ``weighted_tp`` and ``weighted_gt`` are
    therefore per-pose weighted hit and GT mass, not already aggregated
    ratios.  A zero replicate count is reserved for lightweight diagnostics;
    formal calibration always passes at least 10,000.
    """
    tp = np.asarray(weighted_tp, dtype=np.float64).reshape(-1)
    gt = np.asarray(weighted_gt, dtype=np.float64).reshape(-1)
    if tp.size == 0 or tp.size != gt.size:
        return 0.0
    # Empty-candidate poses have no visible weight and must not affect the
    # aggregate bootstrap denominator.  Keeping them in the raw pose stream
    # is useful for pose-count diagnostics, but resampling them as successful
    # recall observations would make the lower bound depend on empty poses.
    valid = np.isfinite(tp) & np.isfinite(gt) & (gt > 1e-12)
    tp = tp[valid]
    gt = gt[valid]
    if tp.size == 0:
        return 1.0
    if int(replicates) <= 0:
        denominator = float(gt.sum())
        return 1.0 if denominator <= 1e-12 else float(tp.sum() / denominator)
    rng = np.random.default_rng(int(seed))
    # Keep memory bounded for larger calibration sets while retaining exactly
    # the requested number of bootstrap draws.
    values = np.empty((int(replicates),), dtype=np.float64)
    chunk = max(1, min(int(replicates), 1_000_000 // max(1, tp.size)))
    for start in range(0, int(replicates), chunk):
        end = min(int(replicates), start + chunk)
        indices = rng.integers(0, tp.size, size=(end - start, tp.size), endpoint=False)
        numerator = tp[indices].sum(axis=1)
        denominator = gt[indices].sum(axis=1)
        values[start:end] = np.divide(
            numerator,
            denominator,
            out=np.ones_like(numerator),
            where=denominator > 1e-12,
        )
    return float(np.quantile(values, 0.05))


TARGET_WEIGHTED_RECALL = 0.99
AGGREGATE_AP_MAX_VALUES = 2_000_000
METRIC_FIELDS = (
    "weighted_recall",
    "weighted_recall_lcb",
    "ordinary_recall",
    "pose_weighted_recall",
    "pose_recall",
    "pose_precision",
    "pose_specificity",
    "pose_accuracy",
    "pose_balanced_accuracy",
    "pose_f1",
    "pose_jaccard",
    "pose_useful_cull",
    "pose_bad_cull",
    "fn_over_gt",
    "bad_cull",
    "cnor",
    "useful_cull",
    "fp_over_gt",
    "pred_over_gt",
    "pose_pr_auc",
    "pose_prevalence",
    "pose_ap_lift",
    "pose_ap_improvement_percent",
    "precision",
    "specificity",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "jaccard",
    "avg_candidate_count",
    "avg_gt_count",
    "avg_pred_count",
    "aggregate_pr_auc",
    "aggregate_prevalence",
    "aggregate_ap_lift",
    "aggregate_ap_improvement_percent",
)


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float(default)


def _weighted_lcb(
    weighted_tp: np.ndarray,
    weighted_gt: np.ndarray,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> float | None:
    if int(bootstrap_replicates) <= 0:
        return None
    valid = np.isfinite(weighted_tp) & np.isfinite(weighted_gt) & (weighted_gt > 1e-12)
    if not bool(np.any(valid)):
        return 1.0
    return float(
        weighted_recall_lower_confidence_bound(
            weighted_tp[valid],
            weighted_gt[valid],
            replicates=int(bootstrap_replicates),
            seed=int(seed),
        )
    )


def _pose_metrics(record: PoseScores, threshold: float) -> dict[str, Any]:
    predicted = record.scores >= float(threshold)
    truth = record.targets.astype(bool, copy=False)
    negative = ~truth
    candidate_count = int(truth.size)
    gt_count = int(truth.sum())
    pred_count = int(predicted.sum())
    tp = int(np.logical_and(predicted, truth).sum())
    fp = int(np.logical_and(predicted, negative).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    tn = int(np.logical_and(~predicted, negative).sum())
    weighted_tp = float(record.visible_weights[np.logical_and(predicted, truth)].sum())
    weighted_gt = float(record.visible_weights[truth].sum())
    recall = _safe_ratio(tp, gt_count, 1.0)
    specificity = _safe_ratio(tn, tn + fp, 1.0)
    pose_ap = (
        average_precision(record.scores, record.targets)
        if gt_count > 0
        else None
    )
    return {
        "pose_id": str(record.pose_index),
        "pose_index": int(record.pose_index),
        "candidate_count": candidate_count,
        "gt_count": gt_count,
        "pred_count": pred_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "weighted_tp": weighted_tp,
        "weighted_gt": weighted_gt,
        "ordinary_recall": recall,
        "weighted_recall": _safe_ratio(weighted_tp, weighted_gt, 1.0),
        "precision": _safe_ratio(tp, tp + fp, 1.0),
        "specificity": specificity,
        "accuracy": _safe_ratio(tp + tn, candidate_count, 1.0),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": _safe_ratio(2.0 * tp, 2.0 * tp + fp + fn),
        "jaccard": _safe_ratio(tp, tp + fp + fn, 1.0),
        "useful_cull": _safe_ratio(tn, candidate_count),
        "bad_cull": _safe_ratio(fn, candidate_count),
        "fp_over_gt": _safe_ratio(fp, gt_count),
        "fn_over_gt": _safe_ratio(fn, gt_count),
        "pred_over_gt": _safe_ratio(pred_count, gt_count),
        "prevalence": _safe_ratio(gt_count, candidate_count),
        "average_precision": pose_ap,
    }


def _record_iterator(records: Iterable[PoseScores]) -> Iterable[PoseScores]:
    scene: str | None = None
    split: str | None = None
    seen = 0
    for record in records:
        if not isinstance(record, PoseScores):
            raise TypeError("V5 metrics require lazy PoseScores records from a columnar sidecar")
        if scene is None:
            scene = record.scene
            split = record.split
        elif record.scene != scene or record.split != split:
            raise ValueError("V5 evaluation records must belong to one scene and split")
        seen += 1
        yield record
    if seen == 0:
        raise ValueError("V5 evaluation requires at least one pose")


def evaluate_scene(
    records: Iterable[PoseScores],
    threshold: float,
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
    include_pose_rows: bool = False,
    max_aggregate_score_values: int | None = None,
) -> dict[str, Any]:
    """Evaluate a score stream without materializing a pose-row JSON document.

    The score sidecar is memory-mapped and only the current pose is expanded.
    Pose-macro AP and all confusion statistics remain exact. Aggregate AP is
    also exact by default. Callers may set ``max_aggregate_score_values`` to
    impose an explicit memory bound; only then can aggregate AP be omitted.
    """
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value):
        raise ValueError("V5 threshold must be finite")
    tp = fp = fn = tn = 0
    weighted_tp = weighted_gt = 0.0
    candidate_count = gt_count = pred_count = 0
    positive_pose_count = 0
    pose_count = 0
    pose_recall_values: list[float] = []
    pose_weighted_recall_values: list[float] = []
    pose_precision_values: list[float] = []
    pose_specificity_values: list[float] = []
    pose_accuracy_values: list[float] = []
    pose_balanced_accuracy_values: list[float] = []
    pose_f1_values: list[float] = []
    pose_jaccard_values: list[float] = []
    pose_useful_cull_values: list[float] = []
    pose_bad_cull_values: list[float] = []
    pose_ap_values: list[float] = []
    pose_prevalence_values: list[float] = []
    per_pose_candidate: list[float] = []
    per_pose_tn: list[float] = []
    per_pose_fp: list[float] = []
    weighted_tp_rows: list[float] = []
    weighted_gt_rows: list[float] = []
    pose_rows: list[dict[str, Any]] = []
    aggregate_scores: list[np.ndarray] = []
    aggregate_targets: list[np.ndarray] = []
    aggregate_value_count = 0
    scene: str | None = None
    split: str | None = None

    for record in _record_iterator(records):
        scene = record.scene if scene is None else scene
        split = record.split if split is None else split
        row = _pose_metrics(record, threshold_value)
        pose_count += 1
        tp += int(row["tp"])
        fp += int(row["fp"])
        fn += int(row["fn"])
        tn += int(row["tn"])
        weighted_tp += float(row["weighted_tp"])
        weighted_gt += float(row["weighted_gt"])
        candidate_count += int(row["candidate_count"])
        gt_count += int(row["gt_count"])
        pred_count += int(row["pred_count"])
        per_pose_candidate.append(float(row["candidate_count"]))
        per_pose_tn.append(float(row["tn"]))
        per_pose_fp.append(float(row["fp"]))
        weighted_tp_rows.append(float(row["weighted_tp"]))
        weighted_gt_rows.append(float(row["weighted_gt"]))
        pose_weighted_recall_values.append(float(row["weighted_recall"]))
        pose_recall_values.append(float(row["ordinary_recall"]))
        pose_precision_values.append(float(row["precision"]))
        pose_specificity_values.append(float(row["specificity"]))
        pose_accuracy_values.append(float(row["accuracy"]))
        pose_balanced_accuracy_values.append(float(row["balanced_accuracy"]))
        pose_f1_values.append(float(row["f1"]))
        pose_jaccard_values.append(float(row["jaccard"]))
        pose_useful_cull_values.append(float(row["useful_cull"]))
        pose_bad_cull_values.append(float(row["bad_cull"]))
        if int(row["gt_count"]) > 0:
            positive_pose_count += 1
            pose_prevalence_values.append(float(row["prevalence"]))
            if row["average_precision"] is not None:
                pose_ap_values.append(float(row["average_precision"]))
        if include_pose_rows:
            pose_rows.append(row)
        if max_aggregate_score_values is None or (
            aggregate_value_count + int(record.scores.size) <= int(max_aggregate_score_values)
        ):
            aggregate_scores.append(np.asarray(record.scores, dtype=np.float32).copy())
            aggregate_targets.append(np.asarray(record.targets, dtype=np.uint8).copy())
            aggregate_value_count += int(record.scores.size)
        else:
            aggregate_scores.clear()
            aggregate_targets.clear()
            aggregate_value_count = int(max_aggregate_score_values) + 1

    lcb = _weighted_lcb(
        np.asarray(weighted_tp_rows, dtype=np.float64),
        np.asarray(weighted_gt_rows, dtype=np.float64),
        bootstrap_replicates=int(bootstrap_replicates),
        seed=int(bootstrap_seed),
    )
    aggregate_ap: float | None = None
    if aggregate_scores and (
        max_aggregate_score_values is None
        or aggregate_value_count <= int(max_aggregate_score_values)
    ):
        aggregate_ap = average_precision(
            np.concatenate(aggregate_scores), np.concatenate(aggregate_targets)
        )
    ordinary_recall = _safe_ratio(tp, gt_count, 1.0)
    weighted_recall = _safe_ratio(weighted_tp, weighted_gt, 1.0)
    pose_pr_auc = float(np.mean(pose_ap_values)) if pose_ap_values else None
    pose_prevalence = float(np.mean(pose_prevalence_values)) if pose_prevalence_values else None
    aggregate_prevalence = _safe_ratio(gt_count, candidate_count)
    pose_ap_lift = (
        _safe_ratio(pose_pr_auc, pose_prevalence)
        if pose_pr_auc is not None and pose_prevalence is not None
        else None
    )
    aggregate_ap_lift = (
        _safe_ratio(aggregate_ap, aggregate_prevalence)
        if aggregate_ap is not None
        else None
    )
    return {
        "scene": scene,
        "split": split,
        "threshold": threshold_value,
        "pose_count": pose_count,
        "positive_pose_count": positive_pose_count,
        "empty_gt_pose_count": pose_count - positive_pose_count,
        "candidate_count": candidate_count,
        "gt_count": gt_count,
        "pred_count": pred_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "weighted_tp": weighted_tp,
        "weighted_gt": weighted_gt,
        "weighted_recall": weighted_recall,
        "weighted_recall_lcb": lcb,
        "ordinary_recall": ordinary_recall,
        "pose_weighted_recall": float(np.mean(pose_weighted_recall_values)),
        "pose_recall": float(np.mean(pose_recall_values)),
        "pose_precision": float(np.mean(pose_precision_values)),
        "pose_specificity": float(np.mean(pose_specificity_values)),
        "pose_accuracy": float(np.mean(pose_accuracy_values)),
        "pose_balanced_accuracy": float(np.mean(pose_balanced_accuracy_values)),
        "pose_f1": float(np.mean(pose_f1_values)),
        "pose_jaccard": float(np.mean(pose_jaccard_values)),
        "pose_useful_cull": float(np.mean(pose_useful_cull_values)),
        "pose_bad_cull": float(np.mean(pose_bad_cull_values)),
        "fn_over_gt": _safe_ratio(fn, gt_count),
        "bad_cull": _safe_ratio(fn, candidate_count),
        "cnor": candidate_normalized_occlusion_recall(
            np.asarray(per_pose_tn, dtype=np.float64),
            np.asarray(per_pose_fp, dtype=np.float64),
            np.asarray(per_pose_candidate, dtype=np.float64),
        ),
        "useful_cull": _safe_ratio(tn, candidate_count),
        "fp_over_gt": _safe_ratio(fp, gt_count),
        "pred_over_gt": _safe_ratio(pred_count, gt_count),
        "precision": _safe_ratio(tp, tp + fp, 1.0),
        "specificity": _safe_ratio(tn, tn + fp, 1.0),
        "accuracy": _safe_ratio(tp + tn, candidate_count, 1.0),
        "balanced_accuracy": 0.5 * (ordinary_recall + _safe_ratio(tn, tn + fp, 1.0)),
        "f1": _safe_ratio(2.0 * tp, 2.0 * tp + fp + fn),
        "jaccard": _safe_ratio(tp, tp + fp + fn, 1.0),
        "avg_candidate_count": _safe_ratio(candidate_count, pose_count),
        "avg_gt_count": _safe_ratio(gt_count, pose_count),
        "avg_pred_count": _safe_ratio(pred_count, pose_count),
        "pose_pr_auc": pose_pr_auc,
        "pose_prevalence": pose_prevalence,
        "pose_ap_lift": pose_ap_lift,
        "pose_ap_improvement_percent": (
            100.0 * (pose_ap_lift - 1.0) if pose_ap_lift is not None else None
        ),
        "aggregate_pr_auc": aggregate_ap,
        "aggregate_prevalence": aggregate_prevalence,
        "aggregate_ap_lift": aggregate_ap_lift,
        "aggregate_ap_improvement_percent": (
            100.0 * (aggregate_ap_lift - 1.0)
            if aggregate_ap_lift is not None
            else None
        ),
        "per_pose": pose_rows if include_pose_rows else None,
        "test_read": split == "test",
        "aggregate_ap_status": (
            "exact"
            if aggregate_ap is not None
            else "omitted_configured_limit"
        ),
    }


def metric_projection(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {field: metrics.get(field) for field in METRIC_FIELDS}


__all__ = [
    "AGGREGATE_AP_MAX_VALUES",
    "METRIC_FIELDS",
    "TARGET_WEIGHTED_RECALL",
    "evaluate_scene",
    "metric_projection",
    "_weighted_lcb",
]
