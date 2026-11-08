"""Bounded-process frozen 96-fit matrix, optional 12 auxiliary fits, then locked 2025.

No remote/paid resources are launched. Run this with the validated CUDA Python.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from run_controlled import digest, json_write
from process_scheduler import ProcessRegistry, schedule
from provenance import QUEUE_FILES, source_hashes


def jobs(include_aux):
    specifications = []
    for lead in (1, 2, 3, 4):
        for rmw in ("on", "off"):
            for head in ("direct", "residual", "fixed", "adaptive"):
                for seed in (42, 123, 2024):
                    specifications.append({"lead": lead, "head": head, "rmw": rmw,
                                           "seed": seed, "aux": False})
    if include_aux:
        for lead in (1, 2, 3, 4):
            for seed in (42, 123, 2024):
                specifications.append({"lead": lead, "head": "adaptive", "rmw": "on",
                                       "seed": seed, "aux": True})
    return specifications


def job_name(job):
    return f"lead{job['lead']}_{job['head']}_rmw_{job['rmw']}_seed{job['seed']}" + ("_aux" if job["aux"] else "")


def execute(command, log_path, state_path, state, registry=None):
    begin = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND " + json.dumps(command) + "\n")
        log.flush()
        process = None
        try:
            starter = registry.start if registry is not None else subprocess.Popen
            process = starter(command, stdout=log, stderr=subprocess.STDOUT)
            json_write(state_path, {**state, "child_pid": process.pid, "log": str(log_path), "started_unix": begin})
            code = process.wait()
        except KeyboardInterrupt:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            json_write(state_path, {**state, "status": "interrupted", "resumable": True})
            raise
        except Exception as error:
            json_write(state_path, {**state, "status": "interrupted" if registry and registry.cancelled.is_set() else "failed",
                                    "error": str(error), "started_unix": begin})
            raise
        finally:
            if registry is not None and process is not None:
                registry.unregister(process)
        if code:
            json_write(state_path, {**state, "status": "interrupted" if registry and registry.cancelled.is_set() else "failed", "exit_code": code,
                                    "log": str(log_path), "resumable_from_last_completed_epoch": True})
            raise RuntimeError(f"Run failed with exit code {code}; inspect {log_path}")
        json_write(state_path, {**state, "status": "completed", "execution_phase": state["status"],
                                "child_pid": process.pid, "started_unix": begin,
                                "finished_unix": time.time(), "elapsed_seconds": time.time() - begin})


def seal_selection(output, specifications):
    selected, checkpoint_hashes = {}, {}
    for lead in (1, 2, 3, 4):
        candidates = {}
        for job in specifications:
            if job["lead"] != lead:
                continue
            run = output / "runs" / job_name(job)
            status = json.loads((run / "status.json").read_text(encoding="utf-8"))
            val = json.loads((run / "validation.json").read_text(encoding="utf-8"))
            checkpoint_hashes[job_name(job)] = {"sha256": digest(run / "best.pt"), "selected_epoch": status["selected_epoch"]}
            if not job["aux"]:
                candidates.setdefault(job["head"] + "_rmw_" + job["rmw"], []).append(float(np.mean(val["window_mean_km"])))
        scores = {name: float(np.mean(values)) for name, values in candidates.items()}
        selected[str(lead * 6)] = {"selected_configuration": min(scores, key=lambda k: (scores[k], k)),
                                   "validation_seed_mean_km": scores}
    lock = {"selection_rule": "For each independent lead task, select the main-matrix head/RMW configuration with lowest three-seed validation window-mean haversine km. Auxiliary experiments are excluded from primary selection. Exact ties use alphabetical configuration names.",
            "selected_per_lead_hours": selected, "all_checkpoint_hashes": checkpoint_hashes,
            "2025_policy": "All predeclared fits are complete and checkpoint hashes sealed before any 2025 evaluation. Report all predeclared comparisons, without reselection from 2025 errors."}
    path = output / "holdout_selection_lock.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != lock:
        raise ValueError("Existing holdout selection lock differs; no checkpoint reselection permitted")
    if not path.exists():
        json_write(path, lock)


def comparisons(output, specifications, here, heldout=False):
    base = output / ("holdout2025" if heldout else "runs")
    label = "2025" if heldout else "2022_2024"
    for lead in (1, 2, 3, 4):
        groups = {}
        for job in specifications:
            if job["lead"] == lead:
                name = job["head"] + "_rmw_" + job["rmw"] + ("_aux" if job["aux"] else "")
                groups.setdefault(name, []).append(str(base / job_name(job)))
        destination = output / f"comparisons_{label}_lead{lead}.json"
        if destination.exists():
            continue
        command = [sys.executable, str(here / "compare_storms.py"), "--output", str(destination)]
        for name, paths in groups.items():
            command += ["--group", name + "=" + ",".join(paths)]
        execute(command, output / "logs" / f"comparisons_{label}_lead{lead}.log", output / "queue_status.json",
                {"status": "storm_bootstrap", "period": label, "lead": lead})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--include-aux", action="store_true")
    p.add_argument("--evaluate-2025", action="store_true")
    p.add_argument("--max-parallel", type=int, default=1, help="Independent child processes on the selected device; throughput setting only")
    p.add_argument("--stop-after-runs", type=int, default=0, help="Pause entire queue after this many successful new/continued fits")
    args = p.parse_args()
    if args.max_parallel < 1 or args.stop_after_runs < 0:
        raise ValueError("Positive max-parallel and nonnegative stop-after-runs required")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    root, output = args.data_root.resolve(), args.output.resolve()
    here = Path(__file__).resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    (output / "job_status").mkdir(exist_ok=True)
    specifications = jobs(args.include_aux)
    plan = {"data_root": str(root), "device": args.device, "jobs": specifications,
            "epochs": 40, "batch_size": 32, "learning_rate": 0.0001,
            "code_sha256": source_hashes(QUEUE_FILES, here),
            "metadata_sha256": {f"lead{lead}": digest(root / f"controlled_12to1_lead{lead}" / "metadata.json") for lead in (1, 2, 3, 4)},
            "auxiliary_weights": {"direct": 0.25, "residual": 0.25, "gate_toward_half": 0.03},
            "primary_selection": "per-lead three-seed validation window-mean km; primary candidates are the 8 main head/RMW configurations; auxiliary configurations excluded"}
    plan_path, state_path = output / "queue_plan.json", output / "queue_status.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Frozen queue plan/code/metadata changed")
    else:
        json_write(plan_path, plan)
    execution_path = output / "queue_execution_context.json"
    invocations = json.loads(execution_path.read_text(encoding="utf-8")) if execution_path.exists() else []
    timing_note = "Concurrent child processes share one device. Recorded fit/epoch seconds include contention and must not be used as isolated model speed rankings. Epochs, batch size, seeds and model-selection budgets are unchanged."
    invocations.append({"pid": os.getpid(), "started_unix": time.time(), "max_parallel": args.max_parallel,
                        "device": args.device, "timing_interpretation": timing_note})
    json_write(execution_path, invocations)
    finished, pending = 0, []
    for job in specifications:
        name = job_name(job)
        run = output / "runs" / name
        status_path = run / "status.json"
        if status_path.exists():
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            if previous["status"] == "completed" and previous["test_evaluated"]:
                finished += 1
                continue
        command = [sys.executable, str(here / "run_controlled.py"), "--data-dir", str(root / f"controlled_12to1_lead{job['lead']}"),
                   "--output", str(run), "--head", job["head"], "--rmw", job["rmw"], "--seed", str(job["seed"]),
                   "--device", args.device, "--epochs", "40", "--batch-size", "32", "--evaluate-test"]
        if job["aux"]:
            command += ["--aux-direct", "0.25", "--aux-residual", "0.25", "--aux-gate", "0.03"]
        if (run / "frozen_config.json").exists():
            command.append("--resume")
        pending.append((name, command))
    registry = ProcessRegistry()
    def worker(name, command):
        # Each worker owns one unique JSON/tmp path; no worker writes queue_status.
        state = {"status": "training", "job": name, "max_parallel": args.max_parallel,
                 "timing_interpretation": timing_note}
        execute(command, output / "logs" / (name + ".log"), output / "job_status" / (name + ".json"), state, registry)
    def progress(state):
        # schedule invokes this callback only on the main thread.
        json_write(state_path, {**state, "completed_fits": finished + len(state["completed_jobs"]),
                                "total_fits": len(specifications), "max_parallel": args.max_parallel})
    selected_pending = pending[:args.stop_after_runs] if args.stop_after_runs else pending
    result = schedule(selected_pending, worker, args.max_parallel, progress, registry)
    finished += len(result["completed"])
    if result["failed"]:
        raise RuntimeError("Training queue failed; no new jobs dispatched after detected failure; already-running jobs were allowed to finish. Inspect queue_status and per-job logs.")
    if finished < len(specifications):
        json_write(state_path, {"status": "paused", "completed_fits": finished, "total_fits": len(specifications)})
        return
    seal_selection(output, specifications)
    comparisons(output, specifications, here)
    if args.evaluate_2025:
        for index, job in enumerate(specifications):
            name = job_name(job)
            destination = output / "holdout2025" / name
            if (destination / "status.json").exists():
                saved = json.loads((destination / "status.json").read_text(encoding="utf-8"))
                if saved["status"] == "completed" and saved["test_evaluated"]:
                    continue
            command = [sys.executable, str(here / "evaluate_locked.py"), "--run", str(output / "runs" / name),
                       "--data-dir", str(root / f"extension2025_lead{job['lead']}"),
                       "--output", str(destination), "--device", args.device]
            if destination.exists():
                command.append("--resume")
            execute(command, output / "logs" / (name + "_2025.log"), state_path,
                    {"status": "evaluating_frozen_2025", "job": name, "completed_fits": finished,
                     "holdout_index": index + 1, "total_fits": len(specifications)})
        comparisons(output, specifications, here, heldout=True)
    json_write(state_path, {"status": "completed", "completed_fits": finished,
                           "total_fits": len(specifications), "holdout2025_evaluated": args.evaluate_2025})


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
