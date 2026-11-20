"""Generate the procedural training, validation, and diagnostic scenes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from gcof_pvs.io import write_json as _json_dump
from .build import build_dataset_manifest, generate_scene_dataset, load_complete_scene_output, rebuild_compiled_scene_assets
from .catalog import generate_synthetic_scene_manifest
from .protocol import DATASET_SCHEMA, DEFAULT_BASE_SEED, DEFAULT_OUTPUT_ROOT, DEFAULT_RENDER_HEIGHT, DEFAULT_RENDER_WIDTH, DEFAULT_VIEWS_PER_SCENE


def _resolve_device(value: str) -> str:
    resolved = str(value).lower()
    if resolved == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if resolved == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        except ImportError as exc:
            raise RuntimeError("CUDA generation requires PyTorch") from exc
    if resolved not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda, or auto")
    return resolved


def select_catalog_scene_ids(
    entries: Mapping[str, Any],
    *,
    all_scenes: bool,
    requested_scene_ids: Sequence[str] | None,
    shard_index: int | None,
    shard_count: int | None,
) -> list[str]:
    """Select one deterministic, disjoint scene shard from the frozen catalog."""

    if (shard_index is None) != (shard_count is None):
        raise ValueError("scene-shard-index and scene-shard-count must be provided together")
    if shard_index is not None:
        if not all_scenes:
            raise ValueError("scene sharding is supported only with --all")
        if shard_count is None or shard_count <= 1 or not 0 <= shard_index < shard_count:
            raise ValueError("scene shard requires count > 1 and 0 <= index < count")
    selected = (
        sorted(str(scene_id) for scene_id in entries)
        if all_scenes
        else sorted(set(str(value) for value in (requested_scene_ids or ())))
    )
    unknown = sorted(set(selected) - set(entries))
    if unknown:
        raise ValueError(f"unknown synthetic scene IDs: {', '.join(unknown)}")
    if shard_index is not None and shard_count is not None:
        selected = selected[shard_index::shard_count]
    return selected


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic procedural synthetic V5 datasets.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene-id", action="append", help="Generate one catalog scene; repeat for several scenes.")
    selection.add_argument("--all", action="store_true", help="Generate all 120 catalog scenes.")
    selection.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Validate all 120 existing scene outputs and write the canonical root manifest.",
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--views-per-scene", type=int, default=DEFAULT_VIEWS_PER_SCENE)
    parser.add_argument("--render-width", type=int, default=DEFAULT_RENDER_WIDTH)
    parser.add_argument("--render-height", type=int, default=DEFAULT_RENDER_HEIGHT)
    parser.add_argument("--scene-shard-index", type=int)
    parser.add_argument("--scene-shard-count", type=int)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="validate and skip scenes that already have a complete matching final manifest",
    )
    parser.add_argument(
        "--rebuild-compiled",
        action="store_true",
        help="rebuild deterministic surface and relation assets while preserving frozen PoseCSR",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.views_per_scene <= 0 or args.render_width <= 0 or args.render_height <= 0:
        raise SystemExit("views-per-scene and render dimensions must be positive")
    device = _resolve_device(args.device)
    catalog = generate_synthetic_scene_manifest(int(args.base_seed))
    entries = {str(entry["sceneId"]): entry for entry in catalog["scenes"]}
    if args.finalize_existing:
        selected = sorted(entries)
        missing = [scene_id for scene_id in selected if load_complete_scene_output(entries[scene_id], args.output_root) is None]
        if missing:
            raise SystemExit(
                f"cannot finalize synthetic dataset; missing {len(missing)} scene manifests"
            )
        root_manifest = build_dataset_manifest(
            catalog,
            generated_scene_ids=selected,
            device=device,
            views_per_scene=int(args.views_per_scene),
            render_width=int(args.render_width),
            render_height=int(args.render_height),
            output_root=args.output_root,
        )
        _json_dump(args.output_root / "synthetic_dataset_manifest.json", root_manifest)
        print(json.dumps({
            "schema": DATASET_SCHEMA,
            "generatedSceneCount": len(selected),
            "finalized": True,
        }), flush=True)
        return
    try:
        selected = select_catalog_scene_ids(
            entries,
            all_scenes=bool(args.all),
            requested_scene_ids=args.scene_id,
            shard_index=args.scene_shard_index,
            shard_count=args.scene_shard_count,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.output_root.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    for scene_id in selected:
        if args.rebuild_compiled:
            print(f"rebuild compiled {scene_id} device={device}", flush=True)
            rebuild_compiled_scene_assets(entries[scene_id], args.output_root, device=device)
            generated.append(scene_id)
            continue
        if args.skip_existing and load_complete_scene_output(entries[scene_id], args.output_root) is not None:
            print(f"skip complete {scene_id}", flush=True)
            generated.append(scene_id)
            continue
        print(f"generate {scene_id} device={device}", flush=True)
        generate_scene_dataset(
            entries[scene_id],
            args.output_root,
            device=device,
            views_per_scene=int(args.views_per_scene),
            render_width=int(args.render_width),
            render_height=int(args.render_height),
        )
        generated.append(scene_id)
    root_manifest = build_dataset_manifest(
        catalog,
        generated_scene_ids=generated,
        device=device,
        views_per_scene=int(args.views_per_scene),
        render_width=int(args.render_width),
        render_height=int(args.render_height),
        output_root=args.output_root,
    )
    manifest_name = (
        "synthetic_dataset_manifest.json"
        if args.scene_shard_index is None
        else f"synthetic_dataset_manifest_shard_{args.scene_shard_index:02d}_of_{args.scene_shard_count:02d}.json"
    )
    _json_dump(args.output_root / manifest_name, root_manifest)
    print(json.dumps({"schema": DATASET_SCHEMA, "generatedSceneCount": len(generated), "scenes": generated}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
