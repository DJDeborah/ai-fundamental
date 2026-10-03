"""Audio quality checks and recording-level splits, before GPU expenditure."""
import csv
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .core import sha256, write_json

EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".aiff"}


def load_audio(path, sample_rate=44100):
    path = Path(path)
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError):
        if not shutil.which("ffmpeg"):
            raise RuntimeError(f"Cannot decode {path.name}. Convert to WAV or install FFmpeg.")
        with tempfile.TemporaryDirectory() as tmp:
            decoded = Path(tmp) / "decoded.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path),
                            "-ac", "1", "-ar", str(sample_rate), str(decoded)], check=True)
            x, sr = sf.read(decoded, dtype="float32", always_2d=True)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    channels = x.shape[1]
    x = x.mean(axis=1)
    if sr != sample_rate:
        divisor = math.gcd(sr, sample_rate)
        x = resample_poly(x, sample_rate // divisor, sr // divisor).astype(np.float32)
    return x, {"original_sample_rate": sr, "original_channels": channels}


def audio_stats(x, sr):
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return {"seconds": len(x) / sr, "peak": peak,
            "rms_dbfs": 20 * math.log10(max(rms, 1e-12)),
            "clipped_fraction": float(np.mean(np.abs(x) >= 0.999)),
            "near_silent_fraction": float(np.mean(np.abs(x) < 0.001))}


def inventory(directory, output):
    directory = Path(directory)
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() in EXTENSIONS:
            try:
                x, meta = load_audio(path)
                rows.append({"path": str(path.relative_to(directory)), "sha256": sha256(path),
                             **meta, **audio_stats(x, 44100)})
            except Exception as e:
                rows.append({"path": str(path.relative_to(directory)), "error": str(e)})
    if not rows:
        raise ValueError(f"No recordings in {directory}; place your recordings there first.")
    write_json(output, {"files": rows, "quality_note": "These stats cannot identify accompaniment or reverb. Listen manually."})
    return {"files": len(rows), "errors": sum("error" in r for r in rows), "output": str(output)}


def read_manifest(path, recordings):
    """A group is a whole song/session, never a randomly split audio chunk."""
    root = Path(recordings).resolve()
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Manifest is empty")
    groups, duplicates = {}, {}
    speakers = set()
    for row in rows:
        for key in ("speaker_id", "group", "split", "path", "consent", "quality"):
            if not row.get(key, "").strip():
                raise ValueError(f"Missing {key}: {row}")
        spk = int(row["speaker_id"])
        if spk < 1 or row["split"] not in {"train", "val", "test", "stress"}:
            raise ValueError(f"Invalid speaker/split: {row}")
        if row["consent"] not in {"self", "authorized"}:
            raise ValueError("consent must be self or authorized")
        if row["split"] in {"train", "val"} and row["quality"] != "clean":
            raise ValueError("First-run train and val recordings must be marked clean")
        source = (root / row["path"]).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"Missing file or path outside recordings: {row['path']}")
        digest = sha256(source)
        if row["group"] in groups and groups[row["group"]] != row["split"]:
            raise ValueError(f"Recording/song group leaks across splits: {row['group']}")
        if digest in duplicates:
            raise ValueError(f"Duplicate source recording: {row['path']}")
        groups[row["group"]] = row["split"]
        duplicates[digest] = row["split"]
        row.update(speaker_id=spk, source_sha256=digest, source_path=source)
        if row["split"] in {"train", "val", "test"}:
            speakers.add(spk)
    if speakers != set(range(1, max(speakers, default=0) + 1)) or not speakers:
        raise ValueError("Speaker IDs must be consecutive starting at 1")
    for spk in speakers:
        for split in ("train", "val", "test"):
            if not any(r["speaker_id"] == spk and r["split"] == split for r in rows):
                raise ValueError(f"Speaker {spk} needs a separate {split} recording")
    return rows


def prepare(manifest, recordings, output, chunk_seconds=8.0, min_seconds=3.0):
    rows = read_manifest(manifest, recordings)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty: {output}. Use a new output directory.")
    if not 2 < min_seconds <= chunk_seconds <= 30:
        raise ValueError("Require 2 < min_seconds <= chunk_seconds <= 30")
    sr = 44100
    chunks, sources = [], []
    for row in rows:
        x, meta = load_audio(row["source_path"], sr)
        source_stats = audio_stats(x, sr)
        # Only reduce illegal float peaks. Never amplify noise or auto-tune training data.
        scale = min(1.0, 0.99 / max(source_stats["peak"], 1e-8))
        x = x * scale
        accepted = 0
        width = round(chunk_seconds * sr)
        for start in range(0, len(x), width):
            clip = x[start:start + width]
            if len(clip) < round(min_seconds * sr) or audio_stats(clip, sr)["rms_dbfs"] < -50:
                continue
            rel = Path(row["split"]) / "audio" / str(row["speaker_id"]) / f"{row['source_sha256'][:16]}_{start:010d}.wav"
            target = output / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target, clip, sr, subtype="PCM_16")
            chunks.append({"path": rel.as_posix(), "speaker_id": row["speaker_id"],
                           "group": row["group"], "split": row["split"],
                           "source_sha256": row["source_sha256"], "sha256": sha256(target),
                           "start_seconds": start / sr, "seconds": len(clip) / sr,
                           "consent": row["consent"], "quality": row["quality"]})
            accepted += 1
        if not accepted:
            raise ValueError(f"No non-silent >= {min_seconds}s clips: {row['path']}")
        sources.append({k: v for k, v in row.items() if k != "source_path"} |
                       meta | source_stats | {"peak_scale": scale, "accepted_chunks": accepted})
    totals = {}
    for c in chunks:
        key = f"{c['split']}/speaker_{c['speaker_id']}"
        totals.setdefault(key, {"clips": 0, "seconds": 0})
        totals[key]["clips"] += 1
        totals[key]["seconds"] += c["seconds"]
    result = {"manifest_sha256": sha256(manifest), "sample_rate": sr, "channels": 1,
              "chunk_seconds": chunk_seconds, "min_seconds": min_seconds,
              "n_speakers": max(r["speaker_id"] for r in rows if r["split"] in {"train", "val", "test"}), "totals": totals,
              "sources": sources, "chunks": chunks,
              "policy": "Split by declared song/session group before slicing; resample/downmix; no denoising, pitch correction or gain amplification."}
    write_json(output / "manifest.json", result)
    return {"totals": totals, "manifest": str(output / "manifest.json")}
