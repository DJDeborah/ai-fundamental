"""Prepare two pinned, private NPR Adele speech-reference candidates.

Simple use from this directory:
    python prepare_npr_adele_reference.py

The public original-publisher MP3 is downloaded with ordinary urllib HTTPS,
then verified against the SHA256 of the source used in the earlier experiment.
No cookies, login bypass, ASR installation, model download or paid GPU is used.
The copyrighted source and crops remain in ignored local assets. This script
does not provide an open license or permission to redistribute those media.

To verify preparation without changing the reference catalog or old references:
    python prepare_npr_adele_reference.py --input-mp3 PATH \
        --out-dir assets/npr_reproduction_check --skip-register
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf

from prepare_adele_reference import ASSET_ROOT, register_reference

ROOT = Path(__file__).resolve().parent
SOURCE_PAGE = "https://www.nprillinois.org/2015-11-24/you-cant-prepare-yourself-a-conversation-with-adele"
MEDIA_URL = "https://ondemand.npr.org/anon.npr-mp3/npr/atc/2015/11/20151124_atc_adele_-_25.mp3"
SOURCE_NAME = "npr_adele_20151124_edited.mp3"
SOURCE_SHA256 = "b194b7691d0fff3a8a45c5e753241b406631b29c0900d6721a7468cf6cf5dbdb"
LICENSE_TEXT = "Copyrighted public NPR interview; not an openly licensed voice library; private local reference experiment, no media redistribution"
RATE = 22050
MAX_SOURCE_BYTES = 64 * 1024 * 1024
INTERVALS = (
    {
        "id": "adele_npr_emotion",
        "display_name": "Adele · NPR 讲话候选（音乐与听众情绪）",
        "file": "npr_adele_speech_emotion_20s.wav",
        "start_s": 78.30,
        "end_s": 98.30,
        "expected_sha256": "5f705eb96f8ae5ee9c915c7880591c1e7f5c34fc8fe48e9ad2870fbdafdc8cb1",
        "selection": "Prior real ASR locates the interviewer ending at77.90s and the Adele guest answer beginning at78.40s; the crop remains within her answer.",
    },
    {
        "id": "adele_npr_parenthood",
        "display_name": "Adele · NPR 讲话候选（母亲身份与孩子）",
        "file": "npr_adele_speech_parenthood_19s.wav",
        "start_s": 211.70,
        "end_s": 230.97,
        "expected_sha256": "bb711312662e4424501e0eb4f5ed41c58b22ebdfe8582a299b6a653c21b57da5",
        "selection": "Prior real ASR locates a continuous Adele guest sentence beginning at211.80s and the sentence-final word ending at230.94s.",
    },
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_source(path):
    if not path.is_file():
        raise FileNotFoundError("Source MP3 was not found: " + str(path))
    if sha256(path) != SOURCE_SHA256:
        raise RuntimeError("NPR source SHA256 differs from the pinned interview. No crops or catalog entries were created.")
    return path


def download_source(folder):
    """Use the publisher's ordinary public media endpoint; verify before rename."""
    destination = folder / SOURCE_NAME
    if destination.exists():
        return require_source(destination)
    temporary = None
    try:
        request = Request(MEDIA_URL, headers={"User-Agent": "NPRReferencePreparation/1.0"})
        with urlopen(request, timeout=45) as response:
            if not response.geturl().startswith("https://"):
                raise RuntimeError("Publisher download redirected away from HTTPS")
            with tempfile.NamedTemporaryFile(dir=folder, prefix=".npr-source-", suffix=".mp3", delete=False) as stream:
                temporary = Path(stream.name)
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_SOURCE_BYTES:
                        raise RuntimeError("Publisher download exceeded the bounded source size")
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        require_source(temporary)
        if destination.exists():
            return require_source(destination)
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def decode_once(source):
    """Exact crops share one decoded timeline, avoiding independent MP3 seeks."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Install FFmpeg and make ffmpeg available on PATH, then run this command again")
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(source),
         "-ac", "1", "-ar", str(RATE), "-f", "f32le", "pipe:1"],
        capture_output=True, check=True, timeout=180,
    )
    waveform = np.frombuffer(result.stdout, dtype="<f4")
    if not len(waveform) or not np.isfinite(waveform).all():
        raise RuntimeError("Decoded source has no finite audio")
    return waveform


def save_candidate(specification, waveform, folder):
    start = round(specification["start_s"] * RATE)
    end = round(specification["end_s"] * RATE)
    if not 0 <= start < end <= len(waveform):
        raise RuntimeError("Pinned interval is outside the actual decoded waveform")
    crop = waveform[start:end]
    peak = float(np.max(np.abs(crop)))
    gain = min(1.0, 0.95 / max(peak, 1e-12))
    destination = folder / specification["file"]
    temporary = None
    reused = destination.exists()
    try:
        if reused:
            if sha256(destination) != specification["expected_sha256"]:
                raise RuntimeError("Existing reference differs from the expected WAV; choose an empty --out-dir. Existing files were not overwritten.")
        else:
            with tempfile.NamedTemporaryFile(dir=folder, prefix=".npr-crop-", suffix=".wav", delete=False) as stream:
                temporary = Path(stream.name)
            sf.write(temporary, crop * gain, RATE, subtype="PCM_16", format="WAV")
            actual = sha256(temporary)
            if actual != specification["expected_sha256"]:
                raise RuntimeError("Prepared WAV SHA256 differs from the measured candidate. Check FFmpeg/NumPy/SoundFile versions; no old reference was overwritten.")
            if destination.exists():
                raise RuntimeError("A reference appeared during preparation; rerun to verify it without overwrite")
            os.replace(temporary, destination)
        stored, rate = sf.read(destination, dtype="float32")
        if rate != RATE or stored.ndim != 1 or not len(stored) or not np.isfinite(stored).all():
            raise RuntimeError("Prepared WAV failed format/finite-sample checks")
        if float(np.max(np.abs(stored))) > 0.951 or np.any(np.abs(stored) >= 0.999):
            raise RuntimeError("Prepared WAV failed headroom check")
        receipt = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "catalog_id": specification["id"],
            "display_name": specification["display_name"],
            "speaker": "Adele (official guest attribution and prior ASR matching; this script performs no speaker verification)",
            "publisher": "NPR",
            "publication_date": "2015-11-24",
            "source_url": SOURCE_PAGE,
            "source_media_url": MEDIA_URL,
            "source_file": SOURCE_NAME,
            "source_sha256": SOURCE_SHA256,
            "source_start_seconds": specification["start_s"],
            "source_end_seconds": specification["end_s"],
            "decoded_source_duration_seconds": len(waveform) / RATE,
            "container_duration_ffprobe_seconds_from_prior_record": 493.83575,
            "timeline_note": "Pinned crops use actual decoded sample indices. Earlier FFprobe metadata and decoded sample duration differed by about0.74s; cause was not established.",
            "kind": "speech",
            "selection_basis": specification["selection"],
            "selection_provenance": "Fixed intervals established on2026-10-03 by actual CPU tiny.en ASR plus official NPR interviewer/guest-role matching. This script replays those intervals; it does not install or rerun ASR, diarization or training.",
            "selection_caveat": "Guest-role attribution is not listening approval or proof of clean background, no overlap or no background music.",
            "sample_rate": rate,
            "seconds": len(stored) / rate,
            "sha256": sha256(destination),
            "expected_sha256": specification["expected_sha256"],
            "hash_matches_measured_candidate": True,
            "decoded_source_crop_peak_abs": peak,
            "static_headroom_gain": gain,
            "static_headroom_gain_db": 20 * math.log10(gain),
            "peak": float(np.max(np.abs(stored))),
            "rms_dbfs": float(20 * np.log10(max(np.sqrt(np.mean(stored.astype(np.float64) ** 2)), 1e-12))),
            "digital_clip_fraction": float(np.mean(np.abs(stored) >= 0.999)),
            "finite_samples": True,
            "processing": "One full FFmpeg22050-Hz mono float32 decode, exact sample-index crop, one static gain to maximum peak0.95, PCM16 WAV. No vocal separation, denoising, waveform mixing or dynamic compression.",
            "license": LICENSE_TEXT,
            "quality_approved": False,
            "background_quality_approved": False,
            "separation_applied": False,
            "publicly_reuploaded": False,
            "purpose": "Local reference-conditioned experiment requested by user",
            "existing_wav_reused": reused,
        }
        sidecar = destination.with_suffix(".json")
        if not sidecar.exists():
            sidecar.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return destination, receipt
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-mp3", type=Path, help="Use an existing local MP3 with the pinned SHA256; omit to download normally from NPR")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "assets" / "adele_private_reference", help="Private output folder; existing references are verified and never overwritten")
    parser.add_argument("--skip-register", action="store_true", help="Prepare/verify WAVs without changing the reference catalog")
    args = parser.parse_args()
    folder = args.out_dir.resolve()
    if not args.skip_register:
        try:
            folder.relative_to(ASSET_ROOT)
        except ValueError as exc:
            raise ValueError("Registered references must be inside workbench/assets; use --skip-register for a separate verification folder") from exc
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required on PATH")
    folder.mkdir(parents=True, exist_ok=True)
    source = require_source(args.input_mp3.resolve()) if args.input_mp3 else download_source(folder)
    waveform = decode_once(source)
    prepared = [save_candidate(specification, waveform, folder) for specification in INTERVALS]
    if not args.skip_register:
        for specification, (destination, receipt) in zip(INTERVALS, prepared):
            register_reference(
                specification["id"], destination, "speech", LICENSE_TEXT,
                source_url=SOURCE_PAGE, display_name=specification["display_name"],
            )
    print(json.dumps({
        "source_sha256": SOURCE_SHA256,
        "catalog_registered": not args.skip_register,
        "quality_approved": False,
        "references": [{
            "id": specification["id"], "path": str(destination), "seconds": receipt["seconds"],
            "sha256": receipt["sha256"], "hash_matches_measured_candidate": True,
            "existing_wav_reused": receipt["existing_wav_reused"],
        } for specification, (destination, receipt) in zip(INTERVALS, prepared)],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
