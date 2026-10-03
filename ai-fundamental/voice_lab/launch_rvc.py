"""Run this ONLY in the AutoDL container. Includes install/download in time cap."""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def gpu_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError("Cannot inspect GPU occupancy; do not launch unattended training")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def shutdown_command(script):
    # AutoDL's wrapper may have no shebang, so it needs an explicit shell.
    # Keep the billing-stop action while preserving the user's recoverable Trash.
    if "supervisord" not in script:
        raise RuntimeError("Unrecognized AutoDL shutdown wrapper; stop via the instance page")
    lines = [line for line in script.splitlines()
             if line.strip() != "rm -rf /root/.local/share/Trash"]
    return ["bash", "-c", "\n".join(lines)]


def allowance(budget, rate, already_spent, maximum_hours=3):
    if not all(math.isfinite(x) for x in (budget, rate, already_spent, maximum_hours)):
        raise ValueError("Budget values must be finite")
    if rate <= 0 or budget <= 0 or already_spent < 0 or maximum_hours <= 0:
        raise ValueError("Invalid budget/rate")
    reserve = 5.0
    available = budget - already_spent - reserve
    if available <= 0:
        raise ValueError("No budget left after 5 yuan reserve")
    seconds = min(maximum_hours * 3600, available / rate * 3600) - 120
    if seconds < 1800:
        raise ValueError("Less than 30 minutes available; do not start this experiment")
    return seconds, reserve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hourly-rate", required=True, type=float, help="Current instance price, yuan/hour; read AutoDL instance details")
    p.add_argument("--spent-so-far", type=float, default=0, help="Prior expense or a documented conservative protection baseline; not a platform bill")
    p.add_argument("--budget", type=float, default=50)
    p.add_argument("--max-hours", type=float, default=3)
    p.add_argument("--out", default="runs/self_rvc_compare")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--failure-log-grace-seconds", type=float, default=600,
                   help="Keep failed logs accessible briefly, within the original time cap")
    a = p.parse_args()
    if not math.isfinite(a.failure_log_grace_seconds) or a.failure_log_grace_seconds < 0:
        raise ValueError("Failure log grace must be finite and nonnegative")
    if sys.platform != "linux" or not Path("/root/autodl-tmp").exists() or not Path("/usr/bin/shutdown").exists():
        raise RuntimeError("AutoDL container required. This launcher invokes the platform shutdown command.")
    seconds, reserve = allowance(a.budget, a.hourly_rate, a.spent_so_far, a.max_hours)
    output = (ROOT / a.out).resolve()
    if not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Output must be inside voice_lab/runs")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output already exists: choose a new --out folder")
    occupied = gpu_processes()
    if occupied:
        raise RuntimeError("GPU already in use. Existing work remains running; start voice experiment after it finishes. " + repr(occupied))
    os.chdir(ROOT)
    audit = ROOT / "runs" / (output.name + "_billing.json")
    audit.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {"utc_started": datetime.now(timezone.utc).isoformat(), "budget_yuan": a.budget,
              "hourly_rate_yuan": a.hourly_rate, "spent_before_launch_yuan": a.spent_so_far,
              "cap_seconds": seconds, "reserve_yuan": reserve,
              "billing_note": "Estimate from supplied rate; excludes unreported prior charges/storage/fees. Confirm platform bill and shutdown status."}
    proc = None
    interrupted = False
    print(json.dumps(result, ensure_ascii=False), flush=True)
    try:
        # start_new_session lets a deadline signal reach training and workers together.
        proc = subprocess.Popen(["bash", "cloud_rvc_pipeline.sh", "--out", str(output), "--steps", str(a.steps)],
                                start_new_session=True)
        try:
            code = proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            interrupted = True
            os.killpg(proc.pid, signal.SIGINT)
            try:
                code = proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                code = proc.wait()
        result.update(exit_code=code, deadline_reached=interrupted)
    except KeyboardInterrupt:
        result.update(exit_code=130, stop_reason="user_interrupt")
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        if result.get("exit_code") and not interrupted:
            # Failed jobs have released the GPU. Allow inspection before shutdown,
            # without extending the time or spending limits of this launch.
            remaining = max(0.0, seconds - (time.monotonic() - started))
            grace = min(a.failure_log_grace_seconds, remaining)
            if grace:
                result["failure_log_grace_seconds"] = grace
                print(f"Pipeline failed; logs available for {grace:.0f} seconds within the existing cap.", flush=True)
                try:
                    time.sleep(grace)
                except KeyboardInterrupt:
                    result["grace_interrupted"] = True
        elapsed = time.monotonic() - started
        result.update(wall_seconds=elapsed, estimated_total_yuan=a.spent_so_far + elapsed / 3600 * a.hourly_rate)
        audit.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if output.exists():
            (output / "billing.json").write_text(audit.read_text(encoding="utf-8"), encoding="utf-8")
            # stdlib-only packing also works if environment install failed.
            import zipfile
            archive = output / "results.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
                for path in output.rglob("*"):
                    if path.is_file() and path != archive and path.suffix != ".zip":
                        z.write(path, path.relative_to(output))
        if result.get("exit_code") == 0:
            # Leave a bounded window for SSH retrieval of the finalized archive.
            # The archive contains the pre-download estimate; the separate audit
            # is updated after this window and remains available after shutdown.
            grace = min(120.0, max(0.0, seconds - (time.monotonic() - started)))
            if grace:
                result["result_download_grace_seconds"] = grace
                result["estimated_total_yuan_upper_through_download"] = (
                    a.spent_so_far + (time.monotonic() - started + grace) / 3600 * a.hourly_rate
                )
                audit.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"RESULTS_READY {output / 'results.zip'}; download window {grace:.0f}s within existing cap.", flush=True)
                try:
                    time.sleep(grace)
                except KeyboardInterrupt:
                    result["download_grace_interrupted"] = True
                elapsed = time.monotonic() - started
                result.update(wall_seconds=elapsed, estimated_total_yuan=a.spent_so_far + elapsed / 3600 * a.hourly_rate)
                audit.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                if output.exists():
                    (output / "billing.json").write_text(audit.read_text(encoding="utf-8"), encoding="utf-8")
        print("Records saved. Requesting AutoDL shutdown.", flush=True)
        # Official AutoDL docs recommend this exact command, without OS-specific args.
        try:
            occupied = gpu_processes()
        except Exception:
            occupied = ["GPU status unavailable"]
        if occupied:
            result["shutdown_skipped"] = "Another GPU process exists or occupancy is unknown"
            audit.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print("Another GPU job is active or occupancy cannot be checked; leaving instance running. Check the AutoDL bill.", flush=True)
        else:
            try:
                command = shutdown_command(Path("/usr/bin/shutdown").read_text(encoding="utf-8"))
                shutdown = subprocess.run(command, check=False)
                if shutdown.returncode:
                    print("Shutdown request failed. Use the AutoDL instance page to stop billing.", flush=True)
            except Exception as error:
                result["shutdown_error"] = repr(error)
                audit.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"Shutdown request failed: {error}. Stop billing via the AutoDL instance page.", flush=True)
    if result.get("exit_code"):
        raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
