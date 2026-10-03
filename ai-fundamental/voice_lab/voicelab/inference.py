import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from .audio import audio_stats, load_audio
from .core import ROOT, run_upstream, sha256, write_json


def mix_weights(alpha, speakers):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if speakers < 2:
        if alpha != 0:
            raise ValueError("A one-speaker model has no second timbre; use alpha=0")
        return None
    return {1: 1 - alpha, 2: alpha}


def convert(source, checkpoint, alphas, output, steps=10, seed=42):
    source, checkpoint, output = Path(source).resolve(), Path(checkpoint).resolve(), Path(output).resolve()
    if steps < 1:
        raise ValueError("inference steps must be positive")
    config = yaml.safe_load((checkpoint.parent / "config.yaml").read_text(encoding="utf-8"))
    speakers = config["model"]["n_spk"]
    weights = [mix_weights(a, speakers) for a in alphas]
    x, _ = load_audio(source)
    if len(x) / 44100 > 60:
        raise ValueError("Use a <=60s excerpt for the first inference experiment")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "inference.json").exists():
        raise ValueError("Inference output already exists; use a new output folder")
    normalized_source = output / "input.wav"
    x = x * min(1.0, 0.99 / max(float(np.max(np.abs(x))), 1e-8))
    sf.write(normalized_source, x, 44100, subtype="PCM_16")
    results = []
    for alpha, mix in zip(alphas, weights):
        name = f"alpha_{alpha:.3f}"
        target = output / (name + ".wav")
        if target.exists():
            raise ValueError("Duplicate alpha or existing WAV; choose distinct alphas/new output")
        argv = [str(ROOT / "voicelab/vendor_entry.py"), "--seed", str(seed), "--script", "main_reflow.py",
                "-i", str(normalized_source), "-m", str(checkpoint), "-o", str(target), "-id", "1", "-k", "0",
                "-step", str(steps), "-method", "euler", "-ts", "0"]
        if mix is not None:
            argv += ["-mix", repr(mix)]
        started = time.perf_counter()
        run_upstream(argv, output / (name + ".log"))
        wall = time.perf_counter() - started
        y, _ = load_audio(target)
        results.append({"alpha": alpha, "speaker_weights": mix, "path": target.name,
                        "sha256": sha256(target), "process_wall_s": wall,
                        "cold_rtf": wall / (len(x) / 44100), "output_stats": audio_stats(y, 44100)})
    result = {"source_sha256": sha256(source), "checkpoint_sha256": sha256(checkpoint),
              "normalized_source_sha256": sha256(normalized_source),
              "config_sha256": sha256(checkpoint.parent / "config.yaml"), "seed": seed,
              "inference_steps": steps, "source_seconds": len(x) / 44100, "outputs": results,
              "timing_note": "cold_rtf includes Python startup, model loading and preprocessing; not streaming latency",
              "mix_note": "Linear interpolation of trained speaker embeddings; alpha is a model weight, not a calibrated perceptual percentage. alpha=0 is resynthesized, not the original waveform."}
    write_json(output / "inference.json", result)
    return result


def pitch_metrics(a, b):
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n]), np.asarray(b[:n])
    va, vb = np.isfinite(a) & (a > 0), np.isfinite(b) & (b > 0)
    joint = va & vb
    cents = 1200 * np.abs(np.log2(b[joint] / a[joint]))
    return {"frames_compared": n, "joint_voiced_frames": int(joint.sum()),
            "voiced_unvoiced_disagreement": float(np.mean(va != vb)) if n else None,
            "median_abs_pitch_cents": float(np.median(cents)) if len(cents) else None,
            "pitch_error_over_50_cents_fraction": float(np.mean(cents > 50)) if len(cents) else None}


def evaluate(source, converted, output):
    import librosa
    x, _ = load_audio(source, sample_rate=22050)
    y, _ = load_audio(converted, sample_rate=22050)
    f0 = []
    for signal in (x, y):
        values, _, _ = librosa.pyin(signal, fmin=50, fmax=1100, sr=22050,
                                    frame_length=2048, hop_length=256)
        f0.append(values)
    result = {"source_sha256": sha256(source), "output_sha256": sha256(converted),
              "estimator": "librosa.pyin", "sr": 22050, "hop_length": 256,
              "source_stats": audio_stats(x, 22050), "converted_stats": audio_stats(y, 22050),
              "duration_ratio": len(y) / len(x), **pitch_metrics(*f0),
              "limitations": "Assumes matching time alignment and dry monophonic vocals; shared voiced frames only. F0 preservation does not measure timbre similarity or naturalness."}
    write_json(output, result)
    return result
