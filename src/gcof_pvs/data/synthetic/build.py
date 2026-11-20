"""Serialize procedural scenes, view-cell labels and compiled geometry."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gcof_pvs.data.pose_csr import DIRECTIONAL_POSE_DTYPE
from gcof_pvs.data.relations import build_proxy_relation_graph, write_proxy_relation_graph
from gcof_pvs.data.schemas import validate_relation_manifest
from gcof_pvs.data.surface import build_local_surface_asset, write_local_surface_asset
from gcof_pvs.io import write_json as _json_dump
from .camera import ViewCell, _camera_bounds, _candidate_ids_for_viewcell, _conservative_mvp, _normalize_camera, build_viewcells, render_viewcell_color_id
from .geometry import PrimitiveScene, _PRIMITIVE_TYPES, build_primitive_scene
from .protocol import (
    COMPILED_SCHEMA, DATASET_SCHEMA, DEFAULT_FAR, DEFAULT_NEAR, DEFAULT_RENDER_HEIGHT,
    DEFAULT_RENDER_WIDTH, DEFAULT_VIEWS_PER_SCENE, EXPERIMENT_NAME,
    MODEL_FOV_Y_DEG, RENDER_FOV_Y_DEG, SCENE_OUTPUT_SCHEMA, SUBPOSES_PER_VIEWCELL,
)


_FAMILY_CATEGORY_IDS = {
    "rooms_doorways_long_corridors": 0,
    "multi_floor_campus_courtyards_colonnades": 1,
    "city_street_canyons_dense_mixed_height": 2,
    "industrial_pipes_equipment_beams_platforms": 3,
    "repeated_and_unique_cluttered_units": 4,
}
_SPLIT_IDS = {"train": 0, "validation": 1, "calibration": 2, "test": 3}
_POSE_SPLIT_ORDER = ("train", "calibration", "validation", "test")


def _write_runtime_assets(scene: PrimitiveScene, scene_root: Path, streaming_granularity: int) -> None:
    resources: dict[int, list[int]] = {}
    for unit in scene.units:
        resources.setdefault(unit.resource_id, []).append(unit.unit_id)
    resource_ids = sorted(resources)
    component_records = []
    for unit in scene.units:
        component_records.append(
            {
                "componentGlobalId": unit.unit_id,
                "globalGlbId": unit.resource_id,
                "primitiveType": unit.primitive_type,
                "bounds": {"min": unit.world_aabb[0].tolist(), "max": unit.world_aabb[1].tolist()},
            }
        )
    resource_records = [
        {
            "globalGlbId": resource_id,
            "path": f"synthetic/{scene.scene_id}/resource_{resource_id:05d}.glb",
            "byteLength": int(streaming_granularity) * 1024,
            "componentGlobalIds": resources[resource_id],
        }
        for resource_id in resource_ids
    ]
    runtime = {
        "schema": "synthetic-runtime-visibility-meta-v1",
        "sceneId": scene.scene_id,
        "instanceCount": scene.unit_count,
        "componentCount": scene.unit_count,
        "globalGlbCount": len(resource_records),
        "sceneBounds": {"min": scene.bounds[0].tolist(), "max": scene.bounds[1].tolist()},
        "componentRecords": component_records,
        "globalGlbRecords": resource_records,
        "instanceToGlobalGlb": [unit.resource_id for unit in scene.units],
        "geometrySource": "deterministic_closed_box_cylinder_beam_panel_meshes",
        "streamingGranularityKiB": int(streaming_granularity),
    }
    glb_index = {
        "schema": "synthetic-glb-index-v1",
        "sceneId": scene.scene_id,
        "total": len(resource_records),
        "entries": resource_records,
    }
    _json_dump(scene_root / "runtimeVisibilityMeta.json", runtime)
    _json_dump(scene_root / "glbIndex.json", glb_index)


def _pose_split(index: int, count: int) -> str:
    if count <= 1:
        return "train"
    train_end = max(1, int(math.ceil(count * 0.75)))
    calibration_end = max(train_end, int(math.ceil(count * 0.875)))
    validation_end = max(calibration_end, int(math.ceil(count * 0.9375)))
    if index < train_end:
        return "train"
    if index < calibration_end:
        return "calibration"
    if index < validation_end:
        return "validation"
    return "test"


def _write_pose_csr(
    scene: PrimitiveScene,
    scene_entry: Mapping[str, Any],
    viewcells: Sequence[ViewCell],
    rendered: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    pose_dir: Path,
    render_width: int,
    render_height: int,
) -> dict[str, Any]:
    pose_dir.mkdir(parents=True, exist_ok=True)
    view_count = len(viewcells)
    aspect = float(render_width) / float(render_height)
    model_tan_y = math.tan(math.radians(MODEL_FOV_Y_DEG) * 0.5)
    model_tan_x = model_tan_y * aspect
    camera_bounds = _camera_bounds(scene, viewcells)
    poses = np.zeros((view_count,), dtype=DIRECTIONAL_POSE_DTYPE)
    visible_offsets = np.zeros((view_count + 1,), dtype=np.uint64)
    candidate_offsets = np.zeros((view_count + 1,), dtype=np.uint64)
    subpose_offsets = np.zeros((view_count + 1,), dtype=np.uint64)
    visible_ids: list[int] = []
    visible_weights: list[float] = []
    visible_hits: list[int] = []
    candidate_ids: list[int] = []
    subpose_positions: list[list[float]] = []
    subpose_forwards: list[list[float]] = []
    subpose_pose_indices: list[int] = []
    subpose_params: list[list[float]] = []
    mvps = np.zeros((view_count, 16), dtype=np.float32)
    viewcell_split_ids = np.zeros((view_count,), dtype=np.uint8)
    category_id = _FAMILY_CATEGORY_IDS.get(str(scene_entry.get("structureFamily")), 255)
    visible_subset_failures = 0
    per_view_stats: list[dict[str, Any]] = []
    for index, (viewcell, labels) in enumerate(zip(viewcells, rendered)):
        visible, weights, hits = labels
        candidates = _candidate_ids_for_viewcell(scene, viewcell, width=render_width, height=render_height)
        missing = np.setdiff1d(visible, candidates, assume_unique=True)
        if missing.size:
            visible_subset_failures += 1
            raise ValueError(
                f"{scene.scene_id} view-cell {index} has {missing.size} visible units "
                "outside the four-subpose 66-degree AABB candidate union"
            )
        pose_split = _pose_split(index, view_count)
        poses["camera_world"][index] = viewcell.center - viewcell.forward * viewcell.back_offset
        poses["camera_forward"][index] = viewcell.forward.astype(np.float32)
        poses["camera_view"][index] = np.asarray([model_tan_x, model_tan_y], dtype=np.float32)
        poses["split"][index] = _SPLIT_IDS[pose_split]
        poses["category"][index] = category_id
        visible_ids.extend(int(value) for value in visible.tolist())
        visible_weights.extend(float(value) for value in weights.tolist())
        visible_hits.extend(int(value) for value in hits.tolist())
        candidate_ids.extend(int(value) for value in candidates.tolist())
        visible_offsets[index + 1] = len(visible_ids)
        candidate_offsets[index + 1] = len(candidate_ids)
        subpose_positions.extend(viewcell.subpose_positions.astype(np.float32).tolist())
        subpose_forwards.extend(viewcell.subpose_forwards.astype(np.float32).tolist())
        subpose_pose_indices.extend(range(index * SUBPOSES_PER_VIEWCELL, (index + 1) * SUBPOSES_PER_VIEWCELL))
        subpose_params.extend([[RENDER_FOV_Y_DEG, aspect, float(render_width), float(render_height)]] * SUBPOSES_PER_VIEWCELL)
        subpose_offsets[index + 1] = len(subpose_positions)
        mvps[index] = _conservative_mvp(poses["camera_world"][index], viewcell.forward, model_tan_x, model_tan_y)
        viewcell_split_ids[index] = _SPLIT_IDS[pose_split]
        per_view_stats.append({"viewcellId": index, "poseSplit": pose_split, "candidateCount": int(candidates.size), "visibleCount": int(visible.size), "missingCandidateCount": int(missing.size)})
    poses["camera_norm"] = np.asarray([_normalize_camera(value, camera_bounds) for value in poses["camera_world"]], dtype=np.float32)
    poses.tofile(pose_dir / "poses.bin")
    mvps.tofile(pose_dir / "mvp.bin")
    visible_offsets.tofile(pose_dir / "visible_offsets.bin")
    np.asarray(visible_ids, dtype="<u4").tofile(pose_dir / "visible_ids.bin")
    np.asarray(visible_weights, dtype="<f4").tofile(pose_dir / "visible_weights.bin")
    np.asarray(visible_hits, dtype="<u2").tofile(pose_dir / "visible_hit_counts.bin")
    candidate_offsets.tofile(pose_dir / "candidate_offsets.bin")
    np.asarray(candidate_ids, dtype="<u4").tofile(pose_dir / "candidate_ids.bin")
    # Preserve the exact geometry-only candidate table under the audit names.
    np.asarray(candidate_ids, dtype="<u4").tofile(pose_dir / "raw_candidate_ids.bin")
    candidate_offsets.tofile(pose_dir / "raw_candidate_offsets.bin")
    np.asarray([cell.viewcell_id for cell in viewcells], dtype="<u4").tofile(pose_dir / "viewcell_ids.bin")
    np.asarray([cell.center for cell in viewcells], dtype="<f4").tofile(pose_dir / "viewcell_centers.bin")
    np.asarray([cell.center for cell in viewcells], dtype="<f4").tofile(pose_dir / "query_center_world.bin")
    np.asarray(poses["camera_world"], dtype="<f4").tofile(pose_dir / "candidate_camera_world.bin")
    np.asarray([cell.radius for cell in viewcells], dtype="<f4").tofile(pose_dir / "viewcell_radius_m.bin")
    np.asarray([cell.forward for cell in viewcells], dtype="<f4").tofile(pose_dir / "viewcell_forwards.bin")
    np.asarray(
        [[cell.radius, MODEL_FOV_Y_DEG, RENDER_FOV_Y_DEG, math.degrees(2.0 * math.atan(model_tan_x)), cell.back_offset, cell.yaw_deg, cell.pitch_deg, SUBPOSES_PER_VIEWCELL] for cell in viewcells],
        dtype="<f4",
    ).tofile(pose_dir / "viewcell_params.bin")
    np.full((view_count,), category_id, dtype="u1").tofile(pose_dir / "viewcell_category_ids.bin")
    viewcell_split_ids.tofile(pose_dir / "viewcell_split_ids.bin")
    subpose_offsets.tofile(pose_dir / "subpose_offsets.bin")
    np.asarray(subpose_positions, dtype="<f4").tofile(pose_dir / "subpose_camera_pos.bin")
    np.asarray(subpose_forwards, dtype="<f4").tofile(pose_dir / "subpose_camera_forward.bin")
    np.asarray(subpose_pose_indices, dtype="<u4").tofile(pose_dir / "subpose_pose_indices.bin")
    np.asarray(subpose_params, dtype="<f4").tofile(pose_dir / "subpose_params.bin")
    split_counts = {name: int(np.count_nonzero(poses["split"] == split_id)) for name, split_id in _SPLIT_IDS.items()}
    meta = {
        "schema": "pose-csr-explicit-four-way-split-v1",
        "sourceAggregationSchema": "viewcell-csr-color-id-approximate-primitive-v1",
        "experiment": EXPERIMENT_NAME,
        "sourceSampler": "synthetic_vectorized_analytic_color_id",
        "runtimeMeta": "../runtimeVisibilityMeta.json",
        "sceneId": scene.scene_id,
        "sceneSeedSplit": str(scene_entry["split"]),
        "viewcellCount": view_count,
        "poseCount": view_count,
        "numInstances": scene.unit_count,
        "poseStrideBytes": int(DIRECTIONAL_POSE_DTYPE.itemsize),
        "mvpStrideBytes": 64,
        "visibleCount": len(visible_ids),
        "candidateCount": len(candidate_ids),
        "candidateSemantics": "union of geometry-only 66-degree AABB frusta at the four backed subpose candidate cameras",
        "candidateVisibleUnionAllowed": False,
        "rawCandidateFile": "raw_candidate_ids.bin",
        "rawCandidateOffsets": "raw_candidate_offsets.bin",
        "rawCandidateSemantics": "same geometry-only candidate table before label aggregation",
        "cameraSemantics": "poses.camera_world is the backed 66-degree candidate camera for the view-cell center",
        "subposeCandidateCameraSemantics": "subpose_camera_pos - subpose_camera_forward * back_offset",
        "queryCenterSemantics": "view-cell center used by the nine-point region contract",
        "viewcellGeometrySemantics": "disk and camera-oriented box regions share one forward direction and four offline subposes",
        "sourceViewcellCount": view_count,
        "sourceSubposeCount": view_count * SUBPOSES_PER_VIEWCELL,
        "gtSemantics": "union of exact analytic primitive Color-ID hits over four same-direction subposes",
        "visibleWeightSemantics": "max pooled Color-ID pixel count in parts per million over subposes",
        "visibleWeightDtype": "float32",
        "modelInputFovYDeg": MODEL_FOV_Y_DEG,
        "frontendRenderFovYDeg": RENDER_FOV_Y_DEG,
        "renderWidth": render_width,
        "renderHeight": render_height,
        "cameraBounds": camera_bounds,
        "splitIds": _SPLIT_IDS,
        "categoryIds": _FAMILY_CATEGORY_IDS,
        "categoryCounts": {str(scene_entry.get("structureFamily")): view_count},
        "splitCounts": split_counts,
        "stats": {
            "avgCandidate": float(len(candidate_ids) / max(1, view_count)),
            "avgVisible": float(len(visible_ids) / max(1, view_count)),
            "avgSubposesPerViewcell": float(SUBPOSES_PER_VIEWCELL),
            "visibleSubsetFailures": visible_subset_failures,
            "viewcellStats": per_view_stats,
        },
        "files": {
            "poses": "poses.bin",
            "mvp": "mvp.bin",
            "viewcellIds": "viewcell_ids.bin",
            "viewcellCenters": "viewcell_centers.bin",
            "queryCenterWorld": "query_center_world.bin",
            "candidateCameraWorld": "candidate_camera_world.bin",
            "viewcellRadiusM": "viewcell_radius_m.bin",
            "viewcellForwards": "viewcell_forwards.bin",
            "viewcellParams": "viewcell_params.bin",
            "viewcellCategoryIds": "viewcell_category_ids.bin",
            "viewcellSplitIds": "viewcell_split_ids.bin",
            "subposeOffsets": "subpose_offsets.bin",
            "subposeCameraPos": "subpose_camera_pos.bin",
            "subposeCameraForward": "subpose_camera_forward.bin",
            "subposePoseIndices": "subpose_pose_indices.bin",
            "subposeParams": "subpose_params.bin",
            "visibleOffsets": "visible_offsets.bin",
            "visibleIds": "visible_ids.bin",
            "visibleWeights": "visible_weights.bin",
            "visibleHitCounts": "visible_hit_counts.bin",
            "candidateOffsets": "candidate_offsets.bin",
            "candidateIds": "candidate_ids.bin",
            "rawCandidateOffsets": "raw_candidate_offsets.bin",
            "rawCandidateIds": "raw_candidate_ids.bin",
        },
    }
    _json_dump(pose_dir / "dataset_meta.json", meta)
    return meta


def generate_scene_dataset(
    scene_entry: Mapping[str, Any],
    output_root: str | Path,
    *,
    device: str = "cpu",
    views_per_scene: int = DEFAULT_VIEWS_PER_SCENE,
    render_width: int = DEFAULT_RENDER_WIDTH,
    render_height: int = DEFAULT_RENDER_HEIGHT,
    allow_small_scene: bool = False,
) -> dict[str, Any]:
    """Generate one scene and return its output manifest."""

    if render_width <= 0 or render_height <= 0:
        raise ValueError("render dimensions must be positive")
    scene = build_primitive_scene(scene_entry, allow_small=allow_small_scene)
    root = Path(output_root)
    scene_root = root / scene.scene_id
    compiled_root = scene_root / "compiled"
    surface_root = compiled_root / "surface"
    relation_root = compiled_root / "relation"
    pose_root = scene_root / "pose_csr"
    surface_asset = build_local_surface_asset(scene.surface_inputs(), sampling_seed=int(scene_entry["generatorSeed"]) + 17)
    write_local_surface_asset(surface_asset, surface_root)
    relation = build_proxy_relation_graph(scene.aabbs, unit_ids=scene.unit_ids.astype(np.uint64))
    write_proxy_relation_graph(relation, relation_root)
    viewcells = build_viewcells(scene, scene_entry, views_per_scene=views_per_scene)
    rendered = []
    for viewcell in viewcells:
        rendered.append(
            render_viewcell_color_id(
                scene,
                viewcell,
                render_width=render_width,
                render_height=render_height,
                device=device,
            )
        )
    pose_meta = _write_pose_csr(
        scene,
        scene_entry,
        viewcells,
        rendered,
        pose_dir=pose_root,
        render_width=render_width,
        render_height=render_height,
    )
    _write_runtime_assets(scene, scene_root, int(scene_entry["streamingGranularityKiB"]))
    _json_dump(
        compiled_root / "primitive_units.json",
        {
            "schema": "gcof-pvs-v5-procedural-primitive-units-v1",
            "sceneId": scene.scene_id,
            "unitCount": scene.unit_count,
            "primitiveTypes": list(_PRIMITIVE_TYPES),
            "units": [unit.mapping() for unit in scene.units],
        },
    )
    _json_dump(
        compiled_root / "compiled_manifest.json",
        {
            "schema": COMPILED_SCHEMA,
            "version": 1,
            "sceneId": scene.scene_id,
            "numUnits": scene.unit_count,
            "surface": "surface",
            "relation": "relation",
            "geometrySource": "procedural_closed_primitive_meshes",
            "primitiveTypes": list(_PRIMITIVE_TYPES),
        },
    )
    output_manifest = {
        "schema": SCENE_OUTPUT_SCHEMA,
        "version": 1,
        "experiment": EXPERIMENT_NAME,
        "scene": dict(scene_entry),
        "sceneId": scene.scene_id,
        "seedSplit": str(scene_entry["split"]),
        "unitCount": scene.unit_count,
        "streamingGranularityKiB": int(scene_entry["streamingGranularityKiB"]),
        "geometry": {
            "primitiveTypes": list(_PRIMITIVE_TYPES),
            "asset": "compiled/primitive_units.json",
            "bounds": {"min": scene.bounds[0].tolist(), "max": scene.bounds[1].tolist()},
        },
        "viewCells": {
            "count": len(viewcells),
            "subposesPerViewcell": SUBPOSES_PER_VIEWCELL,
            "shapes": sorted({cell.region_type for cell in viewcells}),
            "renderWidth": render_width,
            "renderHeight": render_height,
            "modelFovYDeg": MODEL_FOV_Y_DEG,
            "renderFovYDeg": RENDER_FOV_Y_DEG,
            "regions": [cell.region_manifest() for cell in viewcells],
        },
        "assets": {
            "runtimeMeta": "runtimeVisibilityMeta.json",
            "glbIndex": "glbIndex.json",
            "poseCsr": "pose_csr",
            "compiled": "compiled",
            "surface": "compiled/surface",
            "relation": "compiled/relation",
        },
        "poseCsr": {
            "schema": pose_meta["schema"],
            "poseCount": pose_meta["poseCount"],
            "splitCounts": pose_meta["splitCounts"],
            "candidateCount": pose_meta["candidateCount"],
            "visibleCount": pose_meta["visibleCount"],
        },
        "determinism": {
            "generatorSeed": int(scene_entry["generatorSeed"]),
            "device": str(device),
            "fixedPrimitiveMeshBank": True,
        },
    }
    _json_dump(scene_root / "scene_manifest.json", output_manifest)
    return output_manifest


def rebuild_compiled_scene_assets(
    scene_entry: Mapping[str, Any],
    output_root: str | Path,
    *,
    device: str = "cpu",
    allow_small_scene: bool = False,
) -> dict[str, Any]:
    """Rebuild deterministic geometry assets without touching frozen pose GT."""

    scene = build_primitive_scene(scene_entry, allow_small=allow_small_scene)
    scene_root = Path(output_root) / scene.scene_id
    manifest_path = scene_root / "scene_manifest.json"
    required_frozen = (
        scene_root / "runtimeVisibilityMeta.json",
        scene_root / "glbIndex.json",
        scene_root / "pose_csr/dataset_meta.json",
        manifest_path,
    )
    missing = [str(path) for path in required_frozen if not path.is_file()]
    if missing:
        raise ValueError(
            f"cannot rebuild compiled assets for {scene.scene_id}; missing frozen inputs: "
            + ", ".join(missing)
        )
    output_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        output_manifest.get("sceneId") != scene.scene_id
        or int(output_manifest.get("unitCount", -1)) != scene.unit_count
        or output_manifest.get("seedSplit") != str(scene_entry["split"])
    ):
        raise ValueError(f"existing synthetic scene identity disagrees for {scene.scene_id}")

    compiled_root = scene_root / "compiled"
    surface_root = compiled_root / "surface"
    relation_root = compiled_root / "relation"
    surface_asset = build_local_surface_asset(
        scene.surface_inputs(), sampling_seed=int(scene_entry["generatorSeed"]) + 17
    )
    write_local_surface_asset(surface_asset, surface_root)
    relation = build_proxy_relation_graph(scene.aabbs, unit_ids=scene.unit_ids.astype(np.uint64))
    write_proxy_relation_graph(relation, relation_root)

    _json_dump(
        compiled_root / "primitive_units.json",
        {
            "schema": "gcof-pvs-v5-procedural-primitive-units-v1",
            "sceneId": scene.scene_id,
            "unitCount": scene.unit_count,
            "primitiveTypes": list(_PRIMITIVE_TYPES),
            "units": [unit.mapping() for unit in scene.units],
        },
    )
    _json_dump(
        compiled_root / "compiled_manifest.json",
        {
            "schema": COMPILED_SCHEMA,
            "version": 1,
            "sceneId": scene.scene_id,
            "numUnits": scene.unit_count,
            "surface": "surface",
            "relation": "relation",
            "geometrySource": "procedural_closed_primitive_meshes",
            "primitiveTypes": list(_PRIMITIVE_TYPES),
        },
    )
    output_manifest.setdefault("determinism", {})["compiledAssetDevice"] = str(device)
    _json_dump(manifest_path, output_manifest)
    return dict(output_manifest)


def load_complete_scene_output(
    scene_entry: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any] | None:
    """Return a complete matching scene output, or ``None`` when absent."""

    scene_id = str(scene_entry["sceneId"])
    scene_root = Path(output_root) / scene_id
    manifest_path = scene_root / "scene_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"existing synthetic scene manifest is invalid: {manifest_path}") from exc
    expected = {
        "schema": SCENE_OUTPUT_SCHEMA,
        "version": 1,
        "sceneId": scene_id,
        "seedSplit": str(scene_entry["split"]),
        "unitCount": int(scene_entry["unitCount"]),
        "streamingGranularityKiB": int(scene_entry["streamingGranularityKiB"]),
    }
    if not isinstance(manifest, Mapping) or any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"existing synthetic scene identity disagrees with catalog: {manifest_path}")
    required = [
        scene_root / "runtimeVisibilityMeta.json",
        scene_root / "glbIndex.json",
        scene_root / "pose_csr/dataset_meta.json",
        scene_root / "compiled/surface/surface_manifest.json",
        scene_root / "compiled/relation/relation_manifest.json",
        scene_root / "compiled/compiled_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            f"existing synthetic scene {scene_id} is incomplete despite its final manifest: "
            + ", ".join(missing)
        )
    relation_manifest = json.loads(required[4].read_text(encoding="utf-8"))
    validate_relation_manifest(relation_manifest)
    return dict(manifest)


def build_dataset_manifest(
    catalog: Mapping[str, Any],
    *,
    generated_scene_ids: Sequence[str],
    device: str,
    views_per_scene: int,
    render_width: int,
    render_height: int,
    output_root: Path,
) -> dict[str, Any]:
    """Create the root manifest with all formal defaults explicitly recorded."""

    return {
        "schema": DATASET_SCHEMA,
        "version": 1,
        "experiment": EXPERIMENT_NAME,
        "catalogSchema": catalog["schema"],
        "method": "GCOF-PVS-V5",
        "baseSeed": int(catalog["baseSeed"]),
        "sceneCount": int(catalog["sceneCount"]),
        "splitCounts": dict(catalog["splitCounts"]),
        "structureFamilies": list(catalog["structureFamilies"]),
        "formalDefaults": {
            "viewsPerScene": int(views_per_scene),
            "subposesPerViewcell": SUBPOSES_PER_VIEWCELL,
            "renderWidth": int(render_width),
            "renderHeight": int(render_height),
            "modelFovYDeg": MODEL_FOV_Y_DEG,
            "renderFovYDeg": RENDER_FOV_Y_DEG,
            "near": DEFAULT_NEAR,
            "far": DEFAULT_FAR,
            "device": str(device),
        },
        "outputRoot": output_root.as_posix(),
        "sceneOutputs": {scene_id: f"{scene_id}/scene_manifest.json" for scene_id in sorted(generated_scene_ids)},
        "splitProtocol": catalog["splitProtocol"],
        "assets": {
            "poseCsr": "<scene>/pose_csr",
            "compiled": "<scene>/compiled/{surface,relation}",
            "runtimeMeta": "<scene>/runtimeVisibilityMeta.json",
        },
    }
