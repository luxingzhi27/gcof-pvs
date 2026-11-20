"""Shared-scene optimizer, schedule, and artifact lifecycle."""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from gcof_pvs.data.registry import DEFAULT_DATA_ROOT, DEFAULT_REGISTRY
from gcof_pvs.io import write_json as _json_dump
from gcof_pvs.model.network import GCOFPVSV5
from gcof_pvs.protocol import ROBUST_BOUNDARY_OBJECTIVE, RUN_SCHEMA, TRAINING_VARIANTS
from .checkpoint import (
    _atomic_torch_save, _prepare_metrics_log, _resume_contract, _validate_checkpoint, capture_rng_states,
    checkpoint_payload, restore_rng_states, seed_everything,
)
from .objective import RobustBoundaryState
from .scenes import DEFAULT_SYNTHETIC_BASE_SEED, _SceneDataStore, build_training_context
from .schedule import FORMAL_REAL_UPDATES_PER_SCENE, balanced_step_schedule, expected_total_updates
from .step import train_step


def _human_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"gcof_pvs_v5.{output_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def _model_variant_and_objective(variant: str) -> tuple[str, str]:
    normalized = str(variant).upper()
    if normalized not in TRAINING_VARIANTS:
        raise ValueError(f"variant must be one of {TRAINING_VARIANTS}")
    if normalized == "PBCE_OBJECTIVE":
        return "FULL", "PBCE_OBJECTIVE"
    return normalized, ROBUST_BOUNDARY_OBJECTIVE


def _robust_boundary_config(objective: str) -> dict[str, Any] | None:
    if str(objective).upper() != ROBUST_BOUNDARY_OBJECTIVE:
        return None
    return {
        "mode": "smoothmax",
        "rho": 0.1,
        "alpha": 16.0,
        "emaBeta": 0.99,
        # Frozen source-train max_s J_extra(z=0): HKUST is the maximum.
        "lambdaInitial": 46.081927403062075,
        "threshold": 0.0,
        "safetyBudget": 0.01,
    }


def train_run(
    *,
    protocol: str,
    variant: str,
    seed: int,
    output_dir: Path,
    registry_path: Path = DEFAULT_REGISTRY,
    data_root: Path = DEFAULT_DATA_ROOT,
    synthetic_root: Path | None = None,
    synthetic_base_seed: int = DEFAULT_SYNTHETIC_BASE_SEED,
    held_out_scene: str | None = None,
    model_learning_rate: float = 2e-4,
    dual_learning_rate: float = 3e-3,
    weight_decay: float = 1e-5,
    total_updates: int | None = None,
    real_updates_per_scene: int | None = None,
    device: str = "cuda",
    resume: Path | None = None,
    checkpoint_every: int = 500,
    log_every: int = 100,
    pose_count: int = 4,
    geometry_chunk_size: int = 512,
) -> dict[str, Any]:
    """Train one registered V5 member and return its non-test summary."""

    variant = variant.upper()
    model_variant, objective = _model_variant_and_objective(variant)
    boundary_config = _robust_boundary_config(objective)
    if protocol == "loso" and variant == "PBCE_OBJECTIVE":
        raise ValueError(f"{variant} is a shared ablation only")
    if checkpoint_every <= 0 or log_every <= 0 or pose_count <= 0:
        raise ValueError("checkpoint, log, and pose counts must be positive")
    if model_learning_rate <= 0.0 or dual_learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("optimizer parameters are invalid")
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    specs, policy = build_training_context(
        protocol=protocol,
        held_out_scene=held_out_scene,
        registry_path=Path(registry_path),
        data_root=data_root,
        synthetic_root=synthetic_root,
        synthetic_base_seed=synthetic_base_seed,
        require_files=True,
    )
    real_ids = tuple(spec.scene_id for spec in specs if spec.source_kind == "real")
    synthetic_ids = tuple(spec.scene_id for spec in specs if spec.source_kind == "synthetic")
    if total_updates is None and real_updates_per_scene is None:
        real_updates_per_scene = FORMAL_REAL_UPDATES_PER_SCENE
    total = expected_total_updates(
        len(real_ids),
        real_updates_per_scene=real_updates_per_scene,
        total_updates=total_updates,
    )
    schedule_seed = int(seed) + 0x5EED
    scene_ids = tuple(spec.scene_id for spec in specs)
    dual_group_by_scene = {spec.scene_id: spec.dual_group_id for spec in specs}
    dual_group_ids = tuple(dict.fromkeys(dual_group_by_scene.values()))
    schedule_arguments = (
        {"real_updates_per_scene": int(real_updates_per_scene)}
        if real_updates_per_scene is not None
        else {"total_updates": int(total_updates)}
    )
    risk_domain_step_counts = {group_id: 0 for group_id in dual_group_ids}
    for assignment in balanced_step_schedule(
        real_ids,
        synthetic_ids,
        **schedule_arguments,
        seed=schedule_seed,
    ):
        risk_domain_step_counts[dual_group_by_scene[assignment.scene_id]] += 1
    if sum(risk_domain_step_counts.values()) != total or any(
        count <= 0 for count in risk_domain_step_counts.values()
    ):
        raise ValueError("risk-domain schedule counts are incomplete")
    risk_domain_probabilities = {
        group_id: count / float(total)
        for group_id, count in risk_domain_step_counts.items()
    }
    config: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "protocol": str(protocol),
        "variant": str(variant).upper(),
        "objective": objective,
        "modelVariant": model_variant,
        "seed": int(seed),
        "heldOutScene": held_out_scene,
        "modelLearningRate": float(model_learning_rate),
        "dualLearningRate": float(dual_learning_rate),
        "weightDecay": float(weight_decay),
        "totalUpdates": int(total),
        "realUpdatesPerScene": None if real_updates_per_scene is None else int(real_updates_per_scene),
        "scheduleSeed": schedule_seed,
        "poseCount": int(pose_count),
        "geometryChunkSize": int(geometry_chunk_size),
        "sourceSceneIds": list(scene_ids),
        "realSceneIds": list(real_ids),
        "syntheticSceneIds": list(synthetic_ids),
        "dualGrouping": "real_scene_and_synthetic_family_v1",
        "dualGroupIds": list(dual_group_ids),
        "dualGroupByScene": dual_group_by_scene,
        "riskDomainStepCounts": risk_domain_step_counts,
        "riskDomainSamplingProbabilities": risk_domain_probabilities,
        "riskAggregation": None if boundary_config is None else boundary_config["mode"],
        "safetyWeightFloor": None if boundary_config is None else boundary_config["rho"],
        "smoothMaxAlpha": None if boundary_config is None else boundary_config["alpha"],
        "riskEmaBeta": None if boundary_config is None else boundary_config["emaBeta"],
        "sharedDualInitial": None if boundary_config is None else boundary_config["lambdaInitial"],
        "sharedDecisionThreshold": None if boundary_config is None else boundary_config["threshold"],
        "safetyBudget": None if boundary_config is None else boundary_config["safetyBudget"],
        "labelSplitsRead": ["train"],
        "selectionSplitsRead": [],
        "testRead": False,
        "trainingPermissions": policy.to_manifest(),
    }
    config["resumeContract"] = _resume_contract(config)
    run_config_path = output_dir / "run_config.json"
    metrics_path = output_dir / "train_metrics.jsonl"
    if resume is None and any(
        path.exists()
        for path in (
            run_config_path,
            metrics_path,
            output_dir / "checkpoint_last.pt",
            output_dir / "training_summary.json",
        )
    ):
        raise FileExistsError(
            f"refusing to overwrite an existing V5 run without --resume: {output_dir}"
        )
    resume_payload: Mapping[str, Any] | None = None
    if resume is not None:
        resume_payload = torch.load(Path(resume), map_location="cpu", weights_only=False)
        _validate_checkpoint(resume_payload, config)
    _json_dump(run_config_path, config)
    logger = _human_logger(output_dir)
    logger.info(
        "start V5 run protocol=%s variant=%s seed=%s total_updates=%s device=%s testRead=false",
        protocol, variant, seed, total, device_obj,
    )
    logger.info("source scenes real=%d synthetic=%d; calibration/validation/test are not read by trainer", len(real_ids), len(synthetic_ids))

    seed_everything(int(seed))
    data_rng = np.random.default_rng(int(seed) + 0xDADA)
    model = GCOFPVSV5(model_variant).to(device_obj)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_learning_rate), weight_decay=float(weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(total), eta_min=0.0
    )
    if boundary_config is None:
        dual_state = None
    else:
        dual_state = RobustBoundaryState(
            dual_group_ids,
            risk_domain_probabilities,
            dual_lr=dual_learning_rate,
            safety_budget=float(boundary_config["safetyBudget"]),
            alpha=float(boundary_config["alpha"]),
            ema_beta=float(boundary_config["emaBeta"]),
            lambda_initial=float(boundary_config["lambdaInitial"]),
            device="cpu",
        )
    scene_updates = {scene_id: 0 for scene_id in scene_ids}
    global_step = 0
    if resume_payload is not None:
        payload = resume_payload
        if payload["modelConfig"] != model.config:
            raise ValueError("checkpoint model configuration disagrees with the requested variant")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        if dual_state is not None:
            dual_state.load_state_dict(payload["dualState"], strict=True)
        global_step = int(payload["globalStep"])
        saved_updates = {str(key): int(value) for key, value in payload["sceneUpdates"].items()}
        if set(saved_updates) != set(scene_updates) or any(
            value < 0 or value > total for value in saved_updates.values()
        ):
            raise ValueError("checkpoint scene IDs disagree with current source schedule")
        scene_updates = saved_updates
        restore_rng_states(payload["rngStates"], data_rng)
        logger.info("resumed checkpoint=%s at global_step=%d", Path(resume), global_step)

    dual_group_updates = {group_id: 0 for group_id in dual_group_ids}
    for scene_id, update_count in scene_updates.items():
        dual_group_updates[dual_group_by_scene[scene_id]] += int(update_count)

    store = _SceneDataStore(specs)
    assignments = balanced_step_schedule(
        real_ids,
        synthetic_ids,
        **schedule_arguments,
        seed=schedule_seed,
    )
    _prepare_metrics_log(
        metrics_path,
        resume_step=(global_step if resume is not None else None),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with metrics_path.open("a", encoding="utf-8") as metrics_stream:
        for assignment in itertools.islice(assignments, global_step, None):
            current_count = int(scene_updates[assignment.scene_id])
            if current_count != int(assignment.scene_update):
                raise ValueError(
                    f"schedule/checkpoint mismatch for {assignment.scene_id}: "
                    f"expected {assignment.scene_update}, restored {current_count}"
                )
            scene = store.get(assignment.scene_id)
            pose_batch = scene.sample_pose_batch(data_rng, pose_count=pose_count)
            result = train_step(
                model=model,
                scene=scene,
                pose_batch=pose_batch,
                optimizer=optimizer,
                dual_state=dual_state,
                dual_group_id=dual_group_by_scene[assignment.scene_id],
                device=device_obj,
                quarter_turns=assignment.yaw_quarter_turns,
                geometry_chunk_size=geometry_chunk_size,
                objective=objective,
                boundary_rho=(0.1 if boundary_config is None else float(boundary_config["rho"])),
                scheduler=scheduler,
            )
            scene_updates[assignment.scene_id] += 1
            dual_group_updates[result.dual_group_id] += 1
            global_step += 1
            row = {
                "globalStep": global_step,
                "sceneId": result.scene_id,
                "sourceKind": assignment.source_kind,
                "dualGroupId": result.dual_group_id,
                "dualGroupUpdate": dual_group_updates[result.dual_group_id],
                "sceneUpdate": scene_updates[assignment.scene_id],
                "yawQuarterTurns": assignment.yaw_quarter_turns,
                "loss": result.loss,
                "riskExtra": result.risk_extra,
                "riskSafety": result.risk_safety,
                "riskRobust": result.risk_robust,
                "domainRiskEma": result.domain_risk_ema,
                "domainWeight": result.domain_weight,
                "importanceWeight": result.importance_weight,
                "lambdaShared": result.lambda_shared,
                "candidateCount": result.candidate_count,
                "targetUnitCount": result.target_unit_count,
                "geometryUnitCount": result.geometry_unit_count,
                "learningRate": float(optimizer.param_groups[0]["lr"]),
            }
            metrics_stream.write(json.dumps(row, allow_nan=False) + "\n")
            metrics_stream.flush()
            if global_step % log_every == 0 or global_step == 1:
                elapsed = max(time.monotonic() - started, 1e-6)
                logger.info(
                    "step=%d/%d scene=%s risk_domain=%s loss=%.6g extra=%.6g safety=%.6g robust=%s lambda=%s domain_weight=%s lr=%.4g %.2f step/s",
                    global_step, total, result.scene_id, result.dual_group_id,
                    result.loss, result.risk_extra, result.risk_safety, result.risk_robust,
                    result.lambda_shared, result.domain_weight, optimizer.param_groups[0]["lr"],
                    global_step / elapsed,
                )
            if global_step % checkpoint_every == 0:
                payload = checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    dual_state=dual_state,
                    global_step=global_step,
                    scene_updates=scene_updates,
                    rng_states=capture_rng_states(data_rng),
                    config=config,
                )
                _atomic_torch_save(payload, output_dir / "checkpoint_last.pt")
                _atomic_torch_save(payload, output_dir / f"checkpoint_step_{global_step:07d}.pt")

    if global_step != total:
        raise RuntimeError(f"V5 schedule ended at {global_step} updates, expected {total}")
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dual_state=dual_state,
        global_step=global_step,
        scene_updates=scene_updates,
        rng_states=capture_rng_states(data_rng),
        config=config,
    )
    _atomic_torch_save(payload, output_dir / "checkpoint_last.pt")
    _atomic_torch_save(payload, output_dir / f"checkpoint_step_{global_step:07d}.pt")
    summary = {
        "schema": RUN_SCHEMA,
        "run": config,
        "completed": True,
        "globalStep": global_step,
        "sceneUpdates": scene_updates,
        "testRead": False,
        "labelSplitsRead": ["train"],
        "checkpoint": str((output_dir / "checkpoint_last.pt").resolve()),
    }
    _json_dump(output_dir / "training_summary.json", summary)
    logger.info("completed V5 run at %d updates", global_step)
    return summary
