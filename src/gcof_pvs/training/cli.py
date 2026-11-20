"""Check source assets or train a GCOF-PVS model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gcof_pvs.data.registry import DEFAULT_DATA_ROOT, DEFAULT_REGISTRY
from gcof_pvs.protocol import TRAINING_VARIANTS
from .experiment import train_run
from .scenes import DEFAULT_SYNTHETIC_BASE_SEED, preflight_training_assets


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--synthetic-root", type=Path, help="override DATA_ROOT/synthetic")
    parser.add_argument("--synthetic-base-seed", type=int, default=DEFAULT_SYNTHETIC_BASE_SEED)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train one shared or LOSO member")
    train_parser.add_argument("--protocol", choices=("shared", "loso"), required=True)
    train_parser.add_argument("--variant", choices=TRAINING_VARIANTS, required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--held-out-scene")
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--model-lr", type=float, default=2e-4)
    train_parser.add_argument("--dual-lr", type=float, default=3e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    budget = train_parser.add_mutually_exclusive_group()
    budget.add_argument("--real-updates-per-scene", type=int)
    budget.add_argument("--total-updates", type=int)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--checkpoint-every", type=int, default=500)
    train_parser.add_argument("--log-every", type=int, default=100)
    train_parser.add_argument("--pose-count", type=int, default=4)
    train_parser.add_argument("--geometry-chunk-size", type=int, default=512)
    _add_data_arguments(train_parser)

    preflight_parser = subparsers.add_parser("preflight", help="validate every source train asset")
    preflight_parser.add_argument("--protocol", choices=("shared", "loso"), required=True)
    preflight_parser.add_argument("--held-out-scene")
    preflight_parser.add_argument("--output", type=Path, required=True)
    preflight_parser.add_argument("--seed", type=int, default=20260918)
    _add_data_arguments(preflight_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        payload = preflight_training_assets(
            protocol=args.protocol,
            output=args.output,
            registry_path=args.registry,
            data_root=args.data_root,
            synthetic_root=args.synthetic_root,
            synthetic_base_seed=args.synthetic_base_seed,
            held_out_scene=args.held_out_scene,
            seed=args.seed,
        )
        print(json.dumps({
            "output": str(args.output.resolve()),
            "sourceSceneCount": payload["sourceSceneCount"],
            "testRead": False,
        }))
        return
    if args.command == "train":
        train_run(
            protocol=args.protocol,
            variant=args.variant,
            seed=args.seed,
            output_dir=args.output_dir,
            registry_path=args.registry,
            data_root=args.data_root,
            synthetic_root=args.synthetic_root,
            synthetic_base_seed=args.synthetic_base_seed,
            held_out_scene=args.held_out_scene,
            model_learning_rate=args.model_lr,
            dual_learning_rate=args.dual_lr,
            weight_decay=args.weight_decay,
            total_updates=args.total_updates,
            real_updates_per_scene=args.real_updates_per_scene,
            device=args.device,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            log_every=args.log_every,
            pose_count=args.pose_count,
            geometry_chunk_size=args.geometry_chunk_size,
        )
        return


if __name__ == "__main__":
    main()
