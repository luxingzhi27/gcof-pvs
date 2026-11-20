"""Reproducible checkpoint state, validation, and log recovery."""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from gcof_pvs.model.network import GCOFPVSV5
from gcof_pvs.protocol import CHECKPOINT_SCHEMA, RUN_SCHEMA
from .objective import RobustBoundaryState

RNG_SCHEMA = "gcof-pvs-v5-rng-state-v1"


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_states(data_rng: np.random.Generator) -> dict[str, Any]:
    return {
        "schema": RNG_SCHEMA,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "data": data_rng.bit_generator.state,
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_states(states: Mapping[str, Any], data_rng: np.random.Generator) -> None:
    if states.get("schema") != RNG_SCHEMA:
        raise ValueError("unsupported V5 RNG checkpoint schema")
    random.setstate(tuple(states["python"]))
    numpy_state = states["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), numpy_state[2], numpy_state[3], numpy_state[4]))
    data_rng.bit_generator.state = states["data"]
    torch.random.set_rng_state(torch.as_tensor(states["torch"], dtype=torch.uint8, device="cpu"))
    cuda_states = states.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.as_tensor(value, dtype=torch.uint8, device="cpu") for value in cuda_states])


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _prepare_metrics_log(path: Path, *, resume_step: int | None) -> None:
    """Create a fresh metric stream or rewind it to the resumed checkpoint."""

    if resume_step is None:
        if path.exists():
            raise FileExistsError(
                f"refusing to append a new V5 run to an existing metric log: {path}"
            )
        return
    if not path.exists():
        if resume_step != 0:
            raise FileNotFoundError(
                f"resume checkpoint step {resume_step} has no metric log: {path}"
            )
        return
    retained: list[str] = []
    previous = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row.get("globalStep", -1))
        if step <= previous:
            raise ValueError(
                f"V5 metric log steps are not strictly increasing at line {line_number}"
            )
        previous = step
        if step <= resume_step:
            retained.append(json.dumps(row, allow_nan=False))
    if resume_step > 0 and not retained:
        raise ValueError(
            f"metric log does not contain resumed checkpoint step {resume_step}"
        )
    if retained and int(json.loads(retained[-1])["globalStep"]) != resume_step:
        raise ValueError(
            f"metric log does not contain resumed checkpoint step {resume_step}"
        )
    path.write_text(("\n".join(retained) + "\n") if retained else "", encoding="utf-8")


def _resume_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "protocol", "variant", "objective", "seed", "heldOutScene",
        "modelLearningRate", "dualLearningRate", "weightDecay",
        "totalUpdates", "realUpdatesPerScene", "scheduleSeed",
        "poseCount", "geometryChunkSize", "sourceSceneIds",
        "dualGrouping", "dualGroupByScene",
        "riskAggregation", "safetyWeightFloor",
        "smoothMaxAlpha", "riskEmaBeta", "sharedDecisionThreshold",
        "riskDomainSamplingProbabilities", "riskDomainStepCounts", "sharedDualInitial",
        "safetyBudget",
    )
    return {key: config.get(key) for key in keys}


def _validate_checkpoint(payload: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    required = {
        "schema", "modelConfig", "model", "optimizer", "scheduler", "dualState",
        "globalStep", "sceneUpdates", "rngStates", "config", "testRead",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"V5 checkpoint is missing fields: {', '.join(missing)}")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not a current V5 training checkpoint")
    if payload.get("testRead") is not False:
        raise ValueError("V5 training checkpoints must declare testRead=false")
    stored_config = payload["config"]
    if not isinstance(stored_config, Mapping):
        raise ValueError("V5 checkpoint config is invalid")
    if stored_config.get("schema") != RUN_SCHEMA:
        raise ValueError("checkpoint config is not a current V5 run contract")
    if _resume_contract(stored_config) != _resume_contract(config):
        raise ValueError("checkpoint training contract disagrees with the requested run")
    if int(payload["globalStep"]) < 0 or int(payload["globalStep"]) > int(config["totalUpdates"]):
        raise ValueError("checkpoint global step is outside the requested schedule")
    if not isinstance(payload["sceneUpdates"], Mapping):
        raise ValueError("checkpoint scene update counts are invalid")
    if payload["rngStates"].get("schema") != RNG_SCHEMA:
        raise ValueError("checkpoint RNG state is invalid")


def checkpoint_payload(
    *,
    model: GCOFPVSV5,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dual_state: RobustBoundaryState | None,
    global_step: int,
    scene_updates: Mapping[str, int],
    rng_states: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "modelConfig": model.config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "dualState": dual_state.state_dict() if dual_state is not None else None,
        "globalStep": int(global_step),
        "sceneUpdates": {str(key): int(value) for key, value in scene_updates.items()},
        "rngStates": dict(rng_states),
        "config": dict(config),
        "testRead": False,
    }
