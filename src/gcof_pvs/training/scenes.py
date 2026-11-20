"""Resolve and preflight source scene data without accessing held-out labels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gcof_pvs.data.permissions import FoldAccessPolicy, make_loso_training_policy, make_universal_training_policy
from gcof_pvs.data.prepared import V5SceneTrainingData
from gcof_pvs.data.registry import DEFAULT_DATA_ROOT, DEFAULT_REGISTRY, load_registry
from gcof_pvs.data.synthetic.catalog import generate_synthetic_scene_manifest
from gcof_pvs.data.synthetic.protocol import DEFAULT_BASE_SEED as DEFAULT_SYNTHETIC_BASE_SEED
from gcof_pvs.io import write_json as _json_dump


@dataclass(frozen=True)
class TrainingSceneSpec:
    """The geometry and source-train paths needed by one training scene."""

    scene_id: str
    source_kind: str
    dual_group_id: str
    synthetic_family: str | None
    pose_dataset: Path
    runtime_meta: Path
    compiled_dir: Path
    viewcell_shape: str
    viewcell_half_extent_m: tuple[float, float, float]
    camera_clip_m: tuple[float, float]
    region_manifest: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sceneId": self.scene_id,
            "sourceKind": self.source_kind,
            "dualGroupId": self.dual_group_id,
            "syntheticFamily": self.synthetic_family,
            "poseDataset": str(self.pose_dataset),
            "runtimeMeta": str(self.runtime_meta),
            "compiledDir": str(self.compiled_dir),
            "viewcellShape": self.viewcell_shape,
            "viewcellHalfExtentM": list(self.viewcell_half_extent_m),
            "cameraClipM": list(self.camera_clip_m),
            "regionManifest": None if self.region_manifest is None else str(self.region_manifest),
        }


def _real_scene_specs(
    registry: Mapping[str, Any],
    *,
    source_scene_ids: Sequence[str],
    require_files: bool,

) -> list[TrainingSceneSpec]:
    entries = {str(entry["id"]): entry for entry in registry["scenes"]}
    compiled_root = Path(registry["compiledOutputRoot"])
    result: list[TrainingSceneSpec] = []
    for scene_id in source_scene_ids:
        if scene_id not in entries:
            raise ValueError(f"unknown registered V5 scene: {scene_id}")
        entry = entries[scene_id]
        spec = TrainingSceneSpec(
            scene_id=scene_id,
            source_kind="real",
            dual_group_id=scene_id,
            synthetic_family=None,
            pose_dataset=Path(entry["poseDataset"]),
            runtime_meta=Path(entry["runtimeMeta"]),
            compiled_dir=compiled_root / scene_id,
            viewcell_shape=str(entry["viewCell"]["shape"]),
            viewcell_half_extent_m=tuple(float(x) for x in entry["viewCell"]["halfExtentM"]),
            camera_clip_m=tuple(float(x) for x in entry["cameraClipM"]),
        )
        if require_files:
            _require_scene_files(spec)
        result.append(spec)
    return result


def _synthetic_scene_specs(
    root: Path,
    *,
    base_seed: int,
    require_files: bool,

) -> list[TrainingSceneSpec]:
    """Build the 96 seed-level synthetic source scenes without reading labels."""

    catalog = generate_synthetic_scene_manifest(int(base_seed))
    result: list[TrainingSceneSpec] = []
    for entry in sorted(catalog["scenes"], key=lambda item: str(item["sceneId"])):
        if str(entry["split"]) != "train":
            continue
        scene_id = str(entry["sceneId"])
        family = str(entry["structureFamily"])
        scene_root = root / scene_id
        spec = TrainingSceneSpec(
            scene_id=scene_id,
            source_kind="synthetic",
            dual_group_id=f"synthetic_family:{family}",
            synthetic_family=family,
            pose_dataset=scene_root / "pose_csr",
            runtime_meta=scene_root / "runtimeVisibilityMeta.json",
            compiled_dir=scene_root / "compiled",
            # The synthetic manifest carries per-pose disk/box support points.
            # These values are only the fixed constructor fallback.
            viewcell_shape="horizontal_disk",
            viewcell_half_extent_m=(0.75, 0.75, 0.0),
            camera_clip_m=(0.05, 1_000_000.0),
            region_manifest=scene_root / "scene_manifest.json",
        )
        if require_files:
            _require_scene_files(spec)
        result.append(spec)
    if len(result) != 96:
        raise ValueError(f"synthetic source catalog must contain 96 train scenes, found {len(result)}")
    return result


def _require_scene_files(spec: TrainingSceneSpec) -> None:
    required = (
        spec.pose_dataset / "dataset_meta.json",
        spec.runtime_meta,
        spec.compiled_dir / "surface/surface_manifest.json",
        spec.compiled_dir / "relation/relation_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if spec.region_manifest is not None and not spec.region_manifest.is_file():
        missing.append(str(spec.region_manifest))
    if missing:
        raise FileNotFoundError(f"V5 source scene {spec.scene_id} is incomplete: {', '.join(missing)}")


def build_training_context(
    *,
    protocol: str,
    held_out_scene: str | None,
    registry_path: Path = DEFAULT_REGISTRY,
    data_root: Path = DEFAULT_DATA_ROOT,
    synthetic_root: Path | None = None,
    synthetic_base_seed: int = DEFAULT_SYNTHETIC_BASE_SEED,
    require_files: bool,

) -> tuple[tuple[TrainingSceneSpec, ...], FoldAccessPolicy]:
    """Resolve only source scenes for a shared or LOSO training run."""

    protocol = str(protocol).lower()
    if protocol not in {"shared", "loso"}:
        raise ValueError("protocol must be shared or loso")
    registry = load_registry(Path(registry_path), data_root=data_root, validate_files=require_files)
    real_ids = [str(scene["id"]) for scene in registry["scenes"]]
    if protocol == "loso":
        if not held_out_scene or held_out_scene not in real_ids:
            raise ValueError("LOSO requires one registered --held-out-scene")
        source_real_ids = [scene_id for scene_id in real_ids if scene_id != held_out_scene]
    else:
        if held_out_scene is not None:
            raise ValueError("shared training cannot declare a held-out scene")
        source_real_ids = real_ids
    real_specs = _real_scene_specs(
        registry,
        source_scene_ids=source_real_ids,
        require_files=require_files,
    )
    synthetic_specs = _synthetic_scene_specs(
        Path(synthetic_root) if synthetic_root is not None else data_root / "synthetic",
        base_seed=int(synthetic_base_seed),
        require_files=require_files,
    )
    source_ids = [spec.scene_id for spec in (*real_specs, *synthetic_specs)]
    if protocol == "loso":
        policy = make_loso_training_policy(source_ids, str(held_out_scene))
    else:
        policy = make_universal_training_policy(source_ids)
    return tuple((*real_specs, *synthetic_specs)), policy


class _SceneDataStore:
    def __init__(
        self,
        specs: Sequence[TrainingSceneSpec],
    ) -> None:
        self.specs = {spec.scene_id: spec for spec in specs}
        self.loaded: dict[str, V5SceneTrainingData] = {}

    def get(self, scene_id: str) -> V5SceneTrainingData:
        if scene_id not in self.specs:
            raise KeyError(f"unknown source training scene: {scene_id}")
        if scene_id not in self.loaded:
            spec = self.specs[scene_id]
            data = V5SceneTrainingData(
                scene_id=spec.scene_id,
                pose_dataset=spec.pose_dataset,
                runtime_meta=spec.runtime_meta,
                compiled_dir=spec.compiled_dir,
                viewcell_shape=spec.viewcell_shape,
                viewcell_half_extent_m=spec.viewcell_half_extent_m,
                camera_clip_m=spec.camera_clip_m,
                region_manifest=spec.region_manifest,
            )
            self.loaded[scene_id] = data
        return self.loaded[scene_id]


def preflight_training_assets(
    *,
    protocol: str,
    output: Path,
    registry_path: Path = DEFAULT_REGISTRY,
    data_root: Path = DEFAULT_DATA_ROOT,
    synthetic_root: Path | None = None,
    synthetic_base_seed: int = DEFAULT_SYNTHETIC_BASE_SEED,
    held_out_scene: str | None = None,
    seed: int = 20260918,
) -> dict[str, Any]:
    """Open every source asset and sample a train-only pose from each source."""

    specs, policy = build_training_context(
        protocol=protocol,
        held_out_scene=held_out_scene,
        registry_path=registry_path,
        data_root=data_root,
        synthetic_root=synthetic_root,
        synthetic_base_seed=synthetic_base_seed,
        require_files=True,
    )
    store = _SceneDataStore(specs)
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for spec in specs:
        scene = store.get(spec.scene_id)
        pose = scene.sample_pose_batch(rng, pose_count=1)
        if not bool(
            np.isfinite(pose.query_geometry).all()
            and np.isfinite(pose.visible_weights).all()
        ):
            raise ValueError(f"non-finite V5 training input in {spec.scene_id}")
        rows.append({
            "sceneId": spec.scene_id,
            "sourceKind": spec.source_kind,
            "dualGroupId": spec.dual_group_id,
            "syntheticFamily": spec.synthetic_family,
            "numUnits": scene.num_units,
            "validUnits": int(np.count_nonzero(scene.valid_unit_mask)),
            "trainPoseCount": int(scene.train_split.pose_indices.size),
            "eligiblePoseCount": int(scene.denominators.eligible_pose_count),
            "candidateEmptyPoseCount": int(
                scene.denominators.pose_count - scene.denominators.eligible_pose_count
            ),
            "trainCandidateOccurrences": int(scene.denominators.candidate_occurrences),
            "trainVisibleOccurrences": int(scene.denominators.visible_occurrences),
            "initialRiskExtraAtZeroLogit": float(
                (scene.denominators.candidate_occurrences - scene.denominators.visible_occurrences)
                / scene.denominators.visible_occurrences
            ),
            "initialRiskSafetyAtZeroLogit": 1.0,
            "sampleCandidateCount": int(pose.candidate_ids.size),
            "sampleVisibleCount": int(np.count_nonzero(pose.targets)),
        })
    payload = {
        "schema": "gcof-pvs-v5-training-preflight-v1",
        "protocol": str(protocol),
        "heldOutScene": held_out_scene,
        "sourceSceneCount": len(rows),
        "realSceneCount": sum(row["sourceKind"] == "real" for row in rows),
        "syntheticSceneCount": sum(row["sourceKind"] == "synthetic" for row in rows),
        "labelSplitsRead": ["train"],
        "selectionSplitsRead": [],
        "testRead": False,
        "scenes": rows,
    }
    _json_dump(output, payload)
    return payload
