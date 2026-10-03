"""Record real pretrained conversion evidence without training or waveform mixing."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from engine import convert_recording

ROOT = Path(__file__).resolve().parent


def stats(path: Path, original: Path) -> dict:
    y, rate = sf.read(path, dtype="float64")
    x, xrate = sf.read(original, dtype="float64")
    from scipy.signal import resample_poly
    from math import gcd
    if xrate != rate:
        div = gcd(xrate, rate)
        x = resample_poly(x, rate // div, xrate // div)
    block = max(1, int(rate * 0.02))
    n = min(len(x), len(y)) // block
    rx = np.sqrt(np.mean(x[:n * block].reshape(n, block) ** 2, axis=1) + 1e-20)
    ry = np.sqrt(np.mean(y[:n * block].reshape(n, block) ** 2, axis=1) + 1e-20)
    quiet = rx < np.quantile(rx, .95) * 10 ** (-30 / 20)
    quiet_rms = float(np.sqrt(np.mean(ry[quiet] ** 2))) if quiet.any() else None
    return {
        "sample_rate": rate, "seconds": len(y) / rate,
        "finite": bool(np.isfinite(y).all()), "peak": float(np.max(np.abs(y))),
        "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)),
        "clip_fraction": float(np.mean(np.abs(y) >= .99)),
        "source_quiet_frames": int(quiet.sum()),
        "quiet_rms_dbfs": None if quiet_rms is None else float(20 * np.log10(quiet_rms + 1e-12)),
        "envelope_correlation": float(np.corrcoef(rx, ry)[0, 1]),
        "caveat": "Unaligned 20-ms source quiet-frame proxy, not naturalness, timbre similarity or listening approval.",
    }


def main():
    folder = ROOT / "benchmarks"
    source = folder / "self_singing_input_8s.wav"
    self_ref = folder / "self_singing_reference_10s.wav"
    target = ROOT / "assets" / "ljspeech_reference.wav"
    trials = [("model_self_reconstruction", 0.0, True)]
    trials.extend((f"experimental_alpha_{int(a * 100):03d}", a, False)
                  for a in (.25, .5, .75, 1.0))
    receipt = {
        "scope": "Local CPU pretrained OpenVoice V2; singing is an unvalidated extrapolation from a speech model.",
        "target": "Linda Johnson, LJ Speech Public Domain. Not Adele.",
        "source": source.name, "self_reference": self_ref.name,
        "no_paid_gpu_task": True, "new_training_updates": 0,
        "seed": 1234, "interpolation": "lerp in speaker embedding, not waveform addition",
        "quality_approved": False, "listening_scores": None, "trials": [],
    }
    for name, alpha, diagnostic in trials:
        output = folder / f"{name}.wav"
        start = time.perf_counter()
        print(f"Starting {name}", flush=True)
        try:
            meta = convert_recording(source, target, self_ref, output, alpha,
                                     "singing", "lerp", diagnostic)
            trial = {"name": name, "status": "complete", "metadata": meta,
                     "wall_seconds": time.perf_counter() - start,
                     "stats": stats(output, source), "output": output.name}
        except Exception as error:
            trial = {"name": name, "status": "failed", "error": str(error),
                     "wall_seconds": time.perf_counter() - start}
        receipt["trials"].append(trial)
        (folder / "BENCHMARK_RECEIPT.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in trial.items() if k != "metadata"}), flush=True)
        if trial["status"] == "failed":
            break


if __name__ == "__main__":
    main()
