"""Colab-friendly end-to-end game-review pipeline.

This wrapper does the Kaggle-only path:

    Kaggle download -> prepare per-game CSVs -> sentence split -> embed ->
    copy embedding_h5.h5 to Drive -> train VICReg

It keeps the heavy intermediate build in a local workspace for speed, then
persists the final embedding H5 and all training outputs under a Drive run
directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_SCRIPT = PROJECT_ROOT / "game_review_data" / "build.py"
TRAIN_SCRIPT = PROJECT_ROOT / "VICReg_review" / "train_vicreg_review_h5.py"


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def default_workspace_dir() -> Path:
    if in_colab():
        return Path("/content/studable_query_latent_work")
    return PROJECT_ROOT / "_colab_work"


def default_drive_root() -> Path:
    if in_colab():
        return Path("/content/drive/MyDrive/studable_query_latent_runs")
    return PROJECT_ROOT / "_colab_drive_runs"


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_optional_path(path: Path | None, *candidates: Path) -> Path | None:
    if path is not None:
        return Path(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def atomic_json_write(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_copy(source: Path, target: Path) -> None:
    source = Path(source)
    target = Path(target)
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        shutil.copy2(source, tmp_path)
        tmp_path.replace(target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def run_command(label: str, command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"\n--- {label} ---", flush=True)
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def mount_drive(mount_point: Path) -> None:
    if not in_colab():
        return
    from google.colab import drive  # type: ignore

    mount_point = Path(mount_point)
    mount_point.mkdir(parents=True, exist_ok=True)
    drive.mount(str(mount_point))


def prepare_kaggle_credentials(kaggle_json: Path | None) -> Path | None:
    if kaggle_json is None:
        return None

    source = Path(kaggle_json)
    if source.is_dir():
        source = source / "kaggle.json"
    if not source.exists():
        raise FileNotFoundError(f"Kaggle credentials file not found: {source}")

    target = Path.home() / ".kaggle" / "kaggle.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    try:
        target.chmod(0o600)
    except OSError:
        pass
    os.environ["KAGGLE_CONFIG_DIR"] = str(target.parent)
    return target


def build_command(
    build_dir: Path,
    backend: str,
    split_device: str,
    embed_device: str | None,
    token_file: Path | None,
    overwrite: bool,
    min_length: int,
    min_count: int,
    split_chunk_size: int,
    embed_batch_size: int,
    embed_concurrency: int,
    embed_max_in_flight: int | None,
) -> list[str]:
    command = [
        str(sys.executable),
        str(BUILD_SCRIPT),
        "--data-dir",
        str(build_dir),
        "--skip-source1",
        "--skip-enrich",
        "--backend",
        backend,
        "--split-device",
        split_device,
        "--chunk-size",
        str(split_chunk_size),
        "--min-length",
        str(min_length),
        "--min-count",
        str(min_count),
        "--batch-size",
        str(embed_batch_size),
        "--concurrency",
        str(embed_concurrency),
        "--python",
        str(sys.executable),
    ]
    if embed_device:
        command.extend(["--embed-device", embed_device])
    if token_file:
        command.extend(["--token-file", str(token_file)])
    if embed_max_in_flight is not None:
        command.extend(["--max-in-flight", str(embed_max_in_flight)])
    if overwrite:
        command.append("--overwrite")
    return command


def training_command(
    input_h5: Path,
    train_dir: Path,
    device: str,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    sample_fraction: float,
    backward_mode: str,
    vicreg_scope: str,
    output_dim: int,
    reduce_hidden: str,
    expander_dim: int,
    expander_hidden: str,
    compact_variance_weight: float,
    compact_covariance_weight: float,
    probe_every: int,
    probe_start_epoch: int,
    extra_args: list[str],
) -> list[str]:
    command = [
        str(sys.executable),
        str(TRAIN_SCRIPT),
        "--input-h5",
        str(input_h5),
        "--checkpoint-out",
        str(train_dir / "vicreg_review_h5_latest.pt"),
        "--best-checkpoint-out",
        str(train_dir / "vicreg_review_h5_best.pt"),
        "--history-tsv",
        str(train_dir / "vicreg_review_h5_history.tsv"),
        "--manifest-json",
        str(train_dir / "vicreg_review_h5_manifest.json"),
        "--probe-history-tsv",
        str(train_dir / "dual_probe_history.tsv"),
        "--device",
        device,
        "--amp",
        "--cache-mode",
        "queue",
        "--backward-mode",
        backward_mode,
        "--vicreg-scope",
        vicreg_scope,
        "--output-dim",
        str(output_dim),
        "--reduce-hidden",
        reduce_hidden,
        "--expander-dim",
        str(expander_dim),
        "--expander-hidden",
        expander_hidden,
        "--compact-variance-weight",
        str(compact_variance_weight),
        "--compact-covariance-weight",
        str(compact_covariance_weight),
        "--max-view-sentences",
        "0",
        "--epochs",
        str(epochs),
        "--steps-per-epoch",
        str(steps_per_epoch),
        "--batch-size",
        str(batch_size),
        "--sample-fraction",
        str(sample_fraction),
        "--probe-every",
        str(probe_every),
        "--probe-start-epoch",
        str(probe_start_epoch),
    ]
    if device == "cuda":
        command.append("--pin-cache")
    command.extend(extra_args)
    return command


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mount-drive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drive-mount-point", type=Path, default=Path("/content/drive"))
    parser.add_argument("--workspace-dir", type=Path, default=default_workspace_dir())
    parser.add_argument("--drive-root", type=Path, default=default_drive_root())
    parser.add_argument("--run-name", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--kaggle-json", type=Path, default=None)
    parser.add_argument("--token-file", type=Path, default=None)
    parser.add_argument("--backend", choices=["local", "cloud"], default="local")
    parser.add_argument("--split-device", default="cpu")
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--split-chunk-size", type=int, default=2000)
    parser.add_argument("--min-length", type=int, default=300)
    parser.add_argument("--min-count", type=int, default=500)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--embed-concurrency", type=int, default=256)
    parser.add_argument("--embed-max-in-flight", type=int, default=None)
    parser.add_argument("--build-overwrite", action="store_true")
    parser.add_argument("--copy-text-h5", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-fraction", type=float, default=0.6)
    parser.add_argument("--backward-mode", choices=["recompute", "split_recompute", "standard"], default="split_recompute")
    parser.add_argument("--vicreg-scope", choices=["game", "slot"], default="game")
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--reduce-hidden", default="128")
    parser.add_argument("--expander-dim", type=int, default=512)
    parser.add_argument("--expander-hidden", default="256,512")
    parser.add_argument("--compact-variance-weight", type=float, default=25.0)
    parser.add_argument("--compact-covariance-weight", type=float, default=25.0)
    parser.add_argument("--probe-every", type=int, default=0)
    parser.add_argument("--probe-start-epoch", type=int, default=3)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.train_args and args.train_args[0] == "--":
        args.train_args = args.train_args[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.mount_drive:
        mount_drive(args.drive_mount_point)

    workspace_dir = Path(args.workspace_dir).expanduser().resolve()
    build_dir = workspace_dir / "game_review_data"
    drive_root = Path(args.drive_root).expanduser().resolve()
    run_dir = drive_root / args.run_name
    train_dir = run_dir / "train"
    build_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    kaggle_json = resolve_optional_path(
        args.kaggle_json,
        drive_root / "kaggle.json",
        Path("/content/drive/MyDrive/kaggle.json"),
    )
    token_file = resolve_optional_path(
        args.token_file,
        drive_root / "tokenAPI.txt",
        Path("/content/drive/MyDrive/tokenAPI.txt"),
        PROJECT_ROOT / "tokenAPI.txt",
    )

    embed_device = args.embed_device or default_device()
    train_device = default_device()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    pipeline_manifest = run_dir / "pipeline_manifest.json"
    state = {
        "status": "starting",
        "stage": "bootstrap",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "workspace_dir": str(workspace_dir),
        "build_dir": str(build_dir),
        "drive_root": str(drive_root),
        "run_dir": str(run_dir),
        "train_dir": str(train_dir),
        "backend": args.backend,
        "split_device": args.split_device,
        "embed_device": embed_device,
        "train_device": train_device,
        "kaggle_json": "" if kaggle_json is None else str(kaggle_json),
        "token_file": "" if token_file is None else str(token_file),
    }
    atomic_json_write(state, pipeline_manifest)

    try:
        if kaggle_json is not None:
            prepare_kaggle_credentials(kaggle_json)

        build_cmd = build_command(
            build_dir=build_dir,
            backend=args.backend,
            split_device=args.split_device,
            embed_device=embed_device if args.backend == "local" else args.embed_device,
            token_file=token_file if args.backend == "cloud" else None,
            overwrite=args.build_overwrite,
            min_length=args.min_length,
            min_count=args.min_count,
            split_chunk_size=args.split_chunk_size,
            embed_batch_size=args.embed_batch_size,
            embed_concurrency=args.embed_concurrency,
            embed_max_in_flight=args.embed_max_in_flight,
        )
        state.update(
            {
                "status": "running",
                "stage": "build",
                "build_command": build_cmd,
            }
        )
        atomic_json_write(state, pipeline_manifest)
        run_command("build", build_cmd, cwd=PROJECT_ROOT, env=env)

        embedding_h5 = build_dir / "embedding_h5.h5"
        if not embedding_h5.exists():
            raise FileNotFoundError(embedding_h5)

        drive_embedding = run_dir / "embedding_h5.h5"
        atomic_copy(embedding_h5, drive_embedding)
        if args.copy_text_h5:
            text_h5 = build_dir / "text_h5.h5"
            if text_h5.exists():
                atomic_copy(text_h5, run_dir / "text_h5.h5")
            text_manifest = build_dir / "text_h5.h5.manifest.json"
            if text_manifest.exists():
                atomic_copy(text_manifest, run_dir / "text_h5.h5.manifest.json")

        state.update(
            {
                "status": "running",
                "stage": "copy",
                "embedding_h5_local": str(embedding_h5),
                "embedding_h5_drive": str(drive_embedding),
            }
        )
        atomic_json_write(state, pipeline_manifest)

        train_cmd = training_command(
            input_h5=embedding_h5,
            train_dir=train_dir,
            device=train_device,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            sample_fraction=args.sample_fraction,
            backward_mode=args.backward_mode,
            vicreg_scope=args.vicreg_scope,
            output_dim=args.output_dim,
            reduce_hidden=args.reduce_hidden,
            expander_dim=args.expander_dim,
            expander_hidden=args.expander_hidden,
            compact_variance_weight=args.compact_variance_weight,
            compact_covariance_weight=args.compact_covariance_weight,
            probe_every=args.probe_every,
            probe_start_epoch=args.probe_start_epoch,
            extra_args=args.train_args,
        )
        state.update(
            {
                "status": "running",
                "stage": "train",
                "train_command": train_cmd,
            }
        )
        atomic_json_write(state, pipeline_manifest)
        run_command("train", train_cmd, cwd=PROJECT_ROOT, env=env)

        state.update(
            {
                "status": "done",
                "stage": "complete",
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        atomic_json_write(state, pipeline_manifest)
        print("\nDone.", flush=True)
        print(f"Embedding copy: {drive_embedding}", flush=True)
        print(f"Training dir   : {train_dir}", flush=True)
        return 0
    except KeyboardInterrupt:
        state.update(
            {
                "status": "interrupted",
                "stage": state.get("stage", "unknown"),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        atomic_json_write(state, pipeline_manifest)
        raise
    except BaseException as exc:
        state.update(
            {
                "status": "error",
                "stage": state.get("stage", "unknown"),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_json_write(state, pipeline_manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
