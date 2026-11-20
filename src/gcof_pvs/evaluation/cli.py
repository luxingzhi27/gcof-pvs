"""V5 columnar score export and set evaluation command line."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gcof_pvs.data.registry import DEFAULT_DATA_ROOT, DEFAULT_REGISTRY

from .contracts import V5Run
from .evaluation import evaluate_loso_matrix, evaluate_shared_fixed_threshold, evaluate_shared_matrix
from .inference import export_score_bundle
from .summary import summarize_loso, summarize_shared, write_summary


def _add_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("evaluate", help="calibrate and evaluate columnar score manifests")
    parser.add_argument("--protocol", choices=("shared", "loso"), required=True)
    parser.add_argument("--bundle", type=Path, action="append", required=True, help="V5 manifest.json; repeat for matrix members")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--final-test", action="store_true", help="permit a final frozen test replay")
    parser.add_argument("--fixed-zero", action="store_true", help="evaluate a shared model at the pre-registered logit-zero boundary")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--evaluation-bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)


def _add_export_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("export", help="compile checkpoint geometry and export columnar scores")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument("--protocol", choices=("shared", "loso"))
    parser.add_argument("--held-out-scene")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--geometry-chunk-size", type=int, default=512)
    parser.add_argument("--pose-chunk-size", type=int, default=4096)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_evaluate_parser(subparsers)
    _add_export_parser(subparsers)
    return parser.parse_args(argv)


def _evaluate(args: argparse.Namespace) -> None:
    if args.split == "test" and not args.final_test:
        raise PermissionError("--split test requires --final-test")
    runs = [V5Run.from_manifest(path, allow_test_read=bool(args.final_test)) for path in args.bundle]
    if args.fixed_zero:
        if args.protocol != "shared":
            raise ValueError("--fixed-zero currently requires --protocol shared")
        rows = [
            row
            for run in runs
            for row in evaluate_shared_fixed_threshold(
                run,
                threshold=0.0,
                evaluation_split=args.split,
                evaluation_bootstrap_replicates=args.evaluation_bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
                final_test=bool(args.final_test),
            )
        ]
        payload = {
            "schema": "pvs-v5-shared-fixed-zero-evaluation-v1",
            "protocol": "shared",
            "split": args.split,
            "testRead": bool(args.final_test),
            "threshold": 0.0,
            "selectionSplit": "none",
            "rows": rows,
        }
        write_summary(args.output, payload)
        print(json.dumps({"output": str(Path(args.output).resolve()), "rows": len(rows), "threshold": 0.0, "testRead": bool(args.final_test)}))
        return
    common = {
        "evaluation_split": args.split,
        "calibration_bootstrap_replicates": args.calibration_bootstrap_replicates,
        "evaluation_bootstrap_replicates": args.evaluation_bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "final_test": bool(args.final_test),
    }
    if args.protocol == "shared":
        rows = evaluate_shared_matrix(runs, **common)
        payload = summarize_shared(rows)
    else:
        rows = evaluate_loso_matrix(runs, **common)
        payload = summarize_loso(rows)
    write_summary(args.output, payload)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "protocol": args.protocol,
        "rows": len(rows),
        "split": args.split,
        "selectionSplit": "calibration",
        "testRead": bool(args.final_test),
    }, ensure_ascii=False))


def _export(args: argparse.Namespace) -> None:
    manifest = export_score_bundle(
        args.checkpoint,
        args.output_dir,
        registry_path=args.registry,
        data_root=args.data_root,
        scene_ids=args.scene,
        protocol=args.protocol,
        held_out_scene=args.held_out_scene,
        device=args.device,
        geometry_chunk_size=args.geometry_chunk_size,
        pose_chunk_size=args.pose_chunk_size,
        final_test=bool(args.final_test),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps({"manifest": str(manifest), "testRead": bool(args.final_test)}, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "evaluate":
        _evaluate(args)
    elif args.command == "export":
        _export(args)


if __name__ == "__main__":
    main()
