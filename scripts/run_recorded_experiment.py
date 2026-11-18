#!/usr/bin/env python3
"""Run one authorized full-data attempt with shared records and stage acceptance."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_SOURCE = ROOT.parent / "research-workspace" / "src"
sys.path.insert(0, str(WORKSPACE_SOURCE))

from research_workspace import Recorder, source_snapshot
from research_workspace.records import export_csv
from reproduction_recording import Observation, sha256


class Tee:
    def __init__(self, original, output):
        self.original, self.output = original, output

    def write(self, value):
        self.original.write(value)
        self.output.write(value)
        self.output.flush()
        return len(value)

    def flush(self):
        self.original.flush()
        self.output.flush()

    def isatty(self):
        return False


def arguments(mode, seed, gpu, attempt):
    specification = json.loads((ROOT / "configs/cifar100-reproduction.json").read_text())
    batch = specification["batches"][mode]
    if (seed, gpu) not in zip(batch["seeds"], batch["gpus"]):
        raise ValueError("Seed/GPU pair is outside the authorized batch")
    config = {**specification["common"],
              **{key: value for key, value in batch.items() if key not in {"seeds", "gpus"}},
              "random_seed": seed, "gpu_id": gpu,
              "torch_data_dir": str(ROOT / "data/datasets"),
              "ae_weights_dir": str(attempt / "checkpoints/ae"),
              "model_weights_dir": str(attempt / "checkpoints"),
        "ae_specific_weights_path": str(attempt / "checkpoints/ae/no-such-ae.pth"),
              "experiments_dir": str(attempt / "native"),
              "experiment_name": f"{mode}-seed{seed}"}
    argv = []
    for key, value in config.items():
        argv.extend([f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)])
    return specification, argv


def accept(record, observer, attempt, mode):
    import torch

    outputs = [p for p in (attempt / "native").glob("*/logs/*.json")
               if not p.name.endswith(".status.json")]
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one native result, found {outputs}")
    result = json.loads(outputs[0].read_text())
    ga, dft = result["ae_pretrain_results"], result["finetune_results"]
    requested_ga, requested_dft = (1, 1) if mode == "trial" else (30, 10)
    if len(ga["epochs"]) != requested_ga or len(dft["epochs"]) != requested_dft:
        raise RuntimeError("Native result does not contain every requested GA/DFT epoch")
    if not observer.ga_audited or not observer.dft_audited or observer.csa_count != 1:
        raise RuntimeError("Runtime training/freezing audits did not complete")
    if not result["completion"]["state"] == "completed":
        raise RuntimeError("Native result did not mark experiment complete")
    metrics = dft["test_metric"]
    if len(metrics) != requested_dft or not all(0 <= x <= 1 for x in metrics):
        raise RuntimeError("Missing or invalid native test accuracies")
    checkpoint_paths = [outputs[0].with_suffix(".pth"), *list((attempt / "checkpoints/ae").rglob("ae_model.pth"))]
    if len(checkpoint_paths) < 2:
        raise RuntimeError("Missing final AE/DFT checkpoints")
    fingerprints = []
    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from tensors(child)
    for path in checkpoint_paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        values = list(tensors(state))
        if not values or any(not torch.isfinite(value).all() for value in values):
            raise RuntimeError(f"Invalid checkpoint tensors: {path}")
        fingerprints.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    record.event("checkpoint_validation", checkpoints=fingerprints)
    best_index = max(range(len(metrics)), key=lambda i: metrics[i])
    return {"ga_epochs": requested_ga, "csa_epochs": observer.csa_count, "dft_epochs": requested_dft,
            "best_test_accuracy": metrics[best_index], "best_epoch": best_index + 1,
            "final_test_accuracy": dft["final_test_accuracy"], "final_epoch": requested_dft,
            "accuracy_unit": "fraction", "best_tie_rule": "first maximum test accuracy; reporting only",
            "final_test_loss": dft["final_test_loss"], "native_result": str(outputs[0]),
            "checkpoints": fingerprints}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["trial", "formal"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt = (args.attempt or ROOT / ("tmp" if args.mode == "trial" else "results")
               / f"cifar100-{args.mode}" / f"seed{args.seed}-{stamp}").resolve()
    if not attempt.is_relative_to(ROOT / ("tmp" if args.mode == "trial" else "results")):
        parser.error("Attempt must live under the authorized tmp/results location")
    specification, argv = arguments(args.mode, args.seed, args.gpu, attempt)
    os.environ["TORCH_HOME"] = str(ROOT / "data/torch")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    from ae_psl.experiment import parse_experiment
    from ae_psl.orchestrator import run_orchestrator
    from ae_psl.utils.orchestrator_argument_utils import namespace_to_global_args_and_search_space
    parsed = parse_experiment(argv)
    del parsed.dry_run
    effective, sweep = namespace_to_global_args_and_search_space(parsed)
    if args.dry_run:
        print(json.dumps({"attempt": str(attempt), "argv": argv, "config": effective,
                          "sweep": sweep}, indent=2))
        return 0
    os.chdir(ROOT)
    source_files = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"),
                           ROOT / "configs/cifar100-reproduction.json", ROOT / "requirements.txt",
                           ROOT / "scripts/install_domestic.sh"])
    record = Recorder(attempt, project="ae-psl", name=f"cifar100-{args.mode}-seed{args.seed}",
                      group=f"cifar100-{args.mode}", method="ae-psl",
                      protocol=specification["protocol"], issue_url=specification["issue_url"],
                      config={"effective": effective, "sweep": sweep, "mode": args.mode},
                      command=[sys.executable, *sys.argv],
                      source=source_snapshot(ROOT, [str(p.relative_to(ROOT)) for p in source_files]))
    observer = Observation(record)
    output = (attempt / "console.log").open("x", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(sys.stdout, output), Tee(sys.stderr, output)
    try:
        for path in ("data/datasets/cifar100/manifest.json",
                     "data/provenance/imagenet100-cmc/validation.json",
                     "data/torch/hub/checkpoints/vit_b_32-d86f8d99.metadata.json"):
            if not (ROOT / path).is_file():
                raise FileNotFoundError(f"Required validated input missing: {path}")
        record.update_data({path: {"path": str(ROOT / path), "sha256": sha256(ROOT / path)}
                            for path in ("data/datasets/cifar100/manifest.json",
                                         "data/provenance/imagenet100-cmc/validation.json",
                                         "data/provenance/imagenet100-cmc/images.jsonl")})
        packages = {entry.metadata["Name"]: entry.version for entry in importlib.metadata.distributions()}
        (attempt / "environment.json").write_text(json.dumps({
            "python": sys.version, "executable": sys.executable, "packages": packages,
            "TORCH_HOME": os.environ["TORCH_HOME"], "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
            "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"]}, indent=2) + "\n")
        (attempt / "command.json").write_text(json.dumps(
            {"launcher": [sys.executable, *sys.argv], "native_argv": argv}, indent=2) + "\n")
        (attempt / "source.diff").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT))
        gpu = subprocess.run(["nvidia-smi", "-q"], text=True, capture_output=True, check=True)
        (attempt / "gpu.txt").write_text(gpu.stdout)
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        observer.install()
        record.event("start", native_argv=argv, gpu=args.gpu, seed=args.seed, cloud_enabled=False)
        run_orchestrator(parsed)
        summary = accept(record, observer, attempt, args.mode)
        record.finish("success", reason="max_epochs", summary=summary)
        export_csv(attempt, attempt / "metrics.csv")
        print(json.dumps({"attempt": str(attempt), "status": "success", **summary}, indent=2), flush=True)
        return 0
    except BaseException as error:
        traceback.print_exc()
        interrupted_run = isinstance(error, KeyboardInterrupt)
        record.event("exception", exception_type=type(error).__name__, error=str(error),
                     traceback=traceback.format_exc())
        record.finish("interrupted" if interrupted_run else "failed", reason=str(error),
                      exit_code=130 if interrupted_run else 1)
        return 130 if interrupted_run else 1
    finally:
        observer.restore()
        sys.stdout, sys.stderr = original_stdout, original_stderr
        output.close()


if __name__ == "__main__":
    raise SystemExit(main())
