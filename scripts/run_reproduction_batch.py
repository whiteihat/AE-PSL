#!/usr/bin/env python3
"""Run the approved trial, then three independent seeds after acceptance passes."""

from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def gpu_check(devices, minimum_mib=12288):
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,memory.used,uuid",
        "--format=csv,noheader,nounits"], text=True)
    rows = [line.split(", ") for line in output.strip().splitlines()]
    available = {int(row[0]): int(row[1]) for row in rows}
    for device in devices:
        if available.get(device, 0) < minimum_mib:
            raise RuntimeError(f"GPU {device} has {available.get(device, 0)} MiB free; need {minimum_mib}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continue-formal-from", type=Path,
                        help="Continue an already successful full-data seed-42 trial without rerunning it")
    options = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = ROOT / "tmp/cifar100-supervisor" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    queue_path = directory / "queue.json"
    queue = {"batch": stamp, "started_at": now(), "state": "running", "attempts": [],
             "policy": "stop on experiment failure; no automatic source repair or training retry"}
    if options.continue_formal_from:
        prior = options.continue_formal_from.resolve()
        prior_status = json.loads((prior / "status.json").read_text())
        if prior_status.get("state") != "success":
            parser.error("--continue-formal-from requires a successful trial attempt")
        queue["accepted_prior_trial"] = {"attempt": str(prior), "status": prior_status}
    active = []
    handles = []
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    def launch(mode, seed, gpu):
        attempt = ROOT / ("tmp" if mode == "trial" else "results") / f"cifar100-{mode}" / stamp / f"seed{seed}-attempt01"
        command = [str(PYTHON), "-u", str(ROOT / "scripts/run_recorded_experiment.py"),
                   "--mode", mode, "--seed", str(seed), "--gpu", str(gpu), "--attempt", str(attempt)]
        log_path = directory / f"{mode}-seed{seed}.log"
        log = log_path.open("x")
        handles.append(log)
        environment = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                       "TORCH_HOME": str(ROOT / "data/torch"), "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        item = {"mode": mode, "seed": seed, "gpu": gpu, "attempt": str(attempt),
                "command": command, "pid": process.pid, "started_at": now(), "state": "running",
                "launch_log": str(log_path)}
        queue["attempts"].append(item)
        active.append((process, item))
        save(queue_path, queue)
        print(f"Started {mode} seed={seed} GPU={gpu}: {attempt}", flush=True)
        return process, item
    def collect(process, item):
        item.update(exit_code=process.returncode, finished_at=now())
        path = Path(item["attempt"]) / "status.json"
        item["state"] = json.loads(path.read_text())["state"] if path.exists() else "failed"
        save(queue_path, queue)
        active.remove((process, item))
        if process.returncode != 0 or item["state"] != "success":
            raise RuntimeError(f"Experiment failed; source repair requires user confirmation: {item['attempt']}")
    try:
        if options.continue_formal_from is None:
            queue["trial_gpu"] = gpu_check([0])
            save(queue_path, queue)
            process, item = launch("trial", 42, 0)
            while process.poll() is None:
                time.sleep(5)
            collect(process, item)
            queue["trial_accepted_at"] = now()
        # Budget three seed caches plus checkpoints and a 10 GiB margin.
        existing = list((ROOT / "data/activations").glob("*.npy"))
        cached = sum(path.stat().st_size for path in existing)
        feature_bytes = (126689 + 5000) * 50 * 768 * 4
        required = max(0, 3 * feature_bytes - min(cached, feature_bytes)) + 10 * 2**30
        free = shutil.disk_usage(ROOT).free
        queue["formal_space_check"] = {"free_bytes": free, "required_bytes": required,
                                       "cache_credit_bytes": min(cached, feature_bytes)}
        if free < required:
            raise RuntimeError(f"Insufficient free disk before formal batch: {free} < {required}")
        queue["formal_gpus"] = gpu_check([0, 1, 2])
        save(queue_path, queue)
        for seed, gpu in zip((42, 43, 44), (0, 1, 2)):
            launch("formal", seed, gpu)
        while active:
            for process, item in list(active):
                if process.poll() is not None:
                    collect(process, item)
            if active:
                time.sleep(5)
        rows = []
        for item in queue["attempts"]:
            if item["mode"] == "formal":
                status = json.loads((Path(item["attempt"]) / "status.json").read_text())
                rows.append({"seed": item["seed"], "gpu": item["gpu"], "attempt": item["attempt"],
                             **status["summary"], "duration_seconds": status["training_duration_seconds"]})
        values = [row["final_test_accuracy"] for row in rows]
        summary = {"protocol": "cifar100-cmc-vitb32-split5-ae64-v1", "accuracy_unit": "fraction",
                   "seeds": rows, "final_accuracy_mean": statistics.mean(values),
                   "final_accuracy_sample_std": statistics.stdev(values)}
        summary_path = ROOT / "results/cifar100-formal" / stamp / "summary.json"
        save(summary_path, summary)
        queue.update(state="success", finished_at=now(), summary=str(summary_path))
        save(queue_path, queue)
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    except BaseException as error:
        for process, item in list(active):
            os.killpg(process.pid, signal.SIGTERM)
        for process, item in list(active):
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            item.update(state="interrupted", exit_code=process.returncode, finished_at=now())
        queue.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     finished_at=now(), error=f"{type(error).__name__}: {error}")
        save(queue_path, queue)
        print(queue["error"], file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
