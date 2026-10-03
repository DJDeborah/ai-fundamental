"""CPU audio-to-audio OpenVoice V2 with an experimental latent mixer.

The source waveform is never added to the converted waveform. Two reference
embeddings are interpolated before the official inverse flow. Singing is an
unvalidated use case. Product alpha=0 bypass belongs to the interface layer.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream" / "OpenVoice"
MODEL = ROOT / "assets" / "openvoice-v2" / "converter"
DEFAULT_REFERENCE = ROOT / "assets" / "ljspeech_reference.wav"
CODE_REVISION = "74a1d147b17a8c3092dd5430504bd83ef6c7eb23"
MODEL_REVISION = "f36e7edfe1684461a8343844af60babc2efbb727"
CHECKPOINT_SHA256 = "9652c27e92b6b2a91632590ac9962ef7ae2b712e5c5b7f4c34ec55ee2b37ab9e"
SEED = 1234
LOCK = threading.RLock()
_converter = None
_load_seconds = 0.0
_embeddings = {}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_status() -> dict:
    config = MODEL / "config.json"
    sr = 22050
    if config.exists():
        sr = int(json.loads(config.read_text(encoding="utf-8"))["data"]["sampling_rate"])
    missing = [name for name in ["torch", "librosa", "torchaudio", "soundfile", "scipy"] if importlib.util.find_spec(name) is None]
    available = config.exists() and (MODEL / "checkpoint.pth").exists() and (UPSTREAM / "openvoice" / "api.py").exists()
    return {
        "ready": available and not missing, "model_available": available,
        "model_loaded": _converter is not None, "device": "cpu",
        "engine": "OpenVoice V2", "modelpath": str(MODEL / "checkpoint.pth"),
        "model_path": str(MODEL / "checkpoint.pth"),
        "code_revision": CODE_REVISION, "model_revision": MODEL_REVISION,
        "sample_rates": {"speech": sr, "singing": sr},
        "missing_dependencies": missing,
        "default_reference": {
            "path": str(DEFAULT_REFERENCE), "name": "Linda Johnson / LJ Speech",
            "license": "Public Domain", "source_url": "https://keithito.com/LJ-Speech-Dataset/",
            "available": DEFAULT_REFERENCE.exists(),
        },
        "limitations": ["Embedding interpolation is experimental and not a calibrated voice percentage.", "Singing voice conversion is unvalidated for this speech model.", "CPU inference converts a recorded clip, not a live microphone stream."],
    }


def _load():
    global _converter, _load_seconds
    if _converter is not None:
        return _converter
    status = get_status()
    if not status["ready"]:
        raise RuntimeError("Model is not prepared in this Python environment. Run prepare_openvoice.py and install requirements-model.txt in the workbench .venv.")
    if _sha(MODEL / "checkpoint.pth") != CHECKPOINT_SHA256:
        raise RuntimeError("Official OpenVoice checkpoint SHA256 does not match the pinned model")
    receipt = ROOT / "assets" / "openvoice-v2" / "PREPARATION_RECEIPT.json"
    if not receipt.exists():
        raise RuntimeError("Missing preparation receipt")
    record = json.loads(receipt.read_text(encoding="utf-8"))
    if record["code_revision"] != CODE_REVISION or record["model_revision"] != MODEL_REVISION:
        raise RuntimeError("Preparation revisions differ from the pinned adapter")
    for name, item in record["files"].items():
        if name.startswith("code/"):
            if _sha(UPSTREAM / name.removeprefix("code/")) != item["sha256"]:
                raise RuntimeError(f"Official source changed: {name}")
    # Load only this pinned upstream package. It is isolated from old RVC.
    upstream_path = str(UPSTREAM)
    if upstream_path not in sys.path:
        sys.path.insert(0, upstream_path)
    from openvoice.api import OpenVoiceBaseClass, ToneColorConverter
    import torch
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))

    class ConverterWithoutWatermark(ToneColorConverter):
        def __init__(self, config_path: str):
            # Official ToneColorConverter always loads an additional wavmark
            # model. Its enable_watermark kwarg leaks to a base class which
            # does not accept it. Initialize that same base explicitly.
            OpenVoiceBaseClass.__init__(self, config_path, device="cpu")
            self.watermark_model = None
            self.version = getattr(self.hps, "_version_", "v1")

    started = time.perf_counter()
    _converter = ConverterWithoutWatermark(str(MODEL / "config.json"))
    _converter.load_ckpt(str(MODEL / "checkpoint.pth"))
    _load_seconds = time.perf_counter() - started
    return _converter


def _read_audio(path: Path):
    import numpy as np
    import soundfile as sf
    try:
        wave, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise ValueError("This recording format needs ffmpeg. Upload WAV, FLAC, or OGG instead.")
        with tempfile.TemporaryDirectory(prefix="decode-", dir=str(ROOT / "outputs")) as folder:
            decoded = Path(folder) / "decoded.wav"
            subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-y", str(decoded)], check=True, capture_output=True, timeout=60)
            wave, sr = sf.read(str(decoded), dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Audio must be non-empty and contain finite samples")
    if np.max(np.abs(wave)) < 1e-5:
        raise ValueError("Audio is silent")
    duration = len(wave) / sr
    if not 0.5 <= duration <= 35:
        raise ValueError("Use a recording/reference between 0.5 and 35 seconds for this learning demo")
    return wave, int(sr)


def _prepare(path: Path, destination: Path, sr: int) -> dict:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    wave, original_sr = _read_audio(path)
    if original_sr != sr:
        divisor = math.gcd(original_sr, sr)
        wave = resample_poly(wave, sr // divisor, original_sr // divisor).astype(np.float32)
    peak = float(np.max(np.abs(wave)))
    if peak > 1:
        wave /= peak  # prevent clipping; no loudness enhancement/effects
    sf.write(str(destination), wave, sr, subtype="PCM_16")
    return {"raw_sha256": _sha(path), "prepared_sha256": _sha(destination), "original_sr": original_sr, "model_sr": sr, "seconds": len(wave) / sr, "peak_before_clip_guard": peak}


def _embedding(converter, path: Path, key: str):
    if key not in _embeddings:
        import torch
        with torch.inference_mode():
            _embeddings[key] = converter.extract_se([str(path)]).detach()
    return _embeddings[key]


def _mix(a, b, alpha: float, method: str):
    import torch
    if method == "lerp":
        return (1 - alpha) * a + alpha * b
    if method != "slerp":
        raise ValueError("method must be lerp or slerp")
    if alpha == 0:
        return a.clone()
    if alpha == 1:
        return b.clone()
    # Interpolate directions on the sphere, and norms linearly. This retains
    # both endpoints exactly without silently normalizing model embeddings.
    a_norm = torch.linalg.vector_norm(a).clamp_min(1e-8)
    b_norm = torch.linalg.vector_norm(b).clamp_min(1e-8)
    dot = ((a / a_norm) * (b / b_norm)).sum().clamp(-1 + 1e-7, 1 - 1e-7)
    if abs(float(dot)) > 0.9995:
        return (1 - alpha) * a + alpha * b
    angle = torch.acos(dot)
    direction = (torch.sin((1 - alpha) * angle) * a / a_norm + torch.sin(alpha * angle) * b / b_norm) / torch.sin(angle)
    return direction * ((1 - alpha) * a_norm + alpha * b_norm)


def convert_recording(source: Path, target: Path | None, self_ref: Path | None, output: Path, alpha: float, mode: str, method: str, diagnostic_self: bool) -> dict:
    """Convert one clip; metadata is also written beside the returned WAV."""
    if mode not in {"speech", "singing"}:
        raise ValueError("mode must be speech or singing")
    if method not in {"lerp", "slerp"} or not 0 <= float(alpha) <= 1:
        raise ValueError("Select lerp/slerp and an alpha between 0 and 1")
    source, output = Path(source), Path(output)
    target = Path(target) if target else DEFAULT_REFERENCE
    self_ref = Path(self_ref) if self_ref else source
    for path in [source, self_ref] + ([] if diagnostic_self else [target]):
        if not path.is_file():
            raise FileNotFoundError(f"Required audio file is unavailable: {path.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with LOCK:
        import numpy as np
        import soundfile as sf
        import torch
        loaded_before = _converter is not None
        converter = _load()
        sr = int(converter.hps.data.sampling_rate)
        with tempfile.TemporaryDirectory(prefix="prepared-", dir=str(ROOT / "outputs")) as folder:
            folder = Path(folder)
            source_wav, self_wav, target_wav = [folder / (name + ".wav") for name in ("source", "self", "target")]
            source_info = _prepare(source, source_wav, sr)
            self_info = _prepare(self_ref, self_wav, sr)
            # src_se describes this particular recording; self_se is the
            # reusable personal identity chosen by the user for mixing.
            src_se = _embedding(converter, source_wav, source_info["prepared_sha256"])
            self_se = _embedding(converter, self_wav, self_info["prepared_sha256"])
            if diagnostic_self:
                target_info = self_info
                effective_alpha = 0.0
                mixed_se = self_se
            else:
                target_info = _prepare(target, target_wav, sr)
                target_se = _embedding(converter, target_wav, target_info["prepared_sha256"])
                effective_alpha = float(alpha)
                mixed_se = _mix(self_se, target_se, effective_alpha, method)
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            model_started = time.perf_counter()
            with torch.inference_mode():
                wave = converter.convert(str(source_wav), src_se=src_se, tgt_se=mixed_se, output_path=None, tau=0.3, message=None)
            model_seconds = time.perf_counter() - model_started
            if not np.isfinite(wave).all() or not len(wave):
                raise RuntimeError("Official converter produced invalid audio")
            peak = float(np.max(np.abs(wave)))
            # Save the real model result without wet/dry mixing or effects.
            sf.write(str(output), wave, sr, subtype="FLOAT")
            metadata = {
                "engine": "OpenVoice V2", "code_revision": CODE_REVISION, "model_revision": MODEL_REVISION,
                "checkpoint_sha256": CHECKPOINT_SHA256, "config_sha256": _sha(MODEL / "config.json"),
                "adapter_sha256": _sha(Path(__file__)),
                "adapter_changes": ["Official OpenVoice base initialization; watermark_model=None; no upstream source mutation", "Direct extract_se on clean reference; no ASR/VAD models", "Reference embedding lerp/slerp before official inverse flow; no waveform sum"],
                "torch_version": torch.__version__, "device": "cpu", "seed": SEED, "threads": torch.get_num_threads(),
                "sample_rate": sr, "output_seconds": len(wave) / sr,
                "total_seconds": time.perf_counter() - started, "model_seconds": model_seconds,
                "model_load_seconds": 0 if loaded_before else _load_seconds,
                "rtf_model_only": model_seconds / (len(wave) / sr),
                "alpha_requested": float(alpha), "alpha_effective": effective_alpha,
                "interpolation": method, "diagnostic_self": bool(diagnostic_self),
                "mode": mode, "singing_validated": False,
                "source": source_info, "self_reference": self_info, "target_reference": target_info,
                "source_filename": source.name, "self_reference_filename": self_ref.name,
                "target_reference_filename": self_ref.name if diagnostic_self else target.name,
                "output_sha256": _sha(output), "output_peak": peak, "samples_outside_unit_interval": int((np.abs(wave) > 1).sum()),
                "waveform_mixing": False, "postprocessing": "none; FLOAT WAV preserves model output",
                "watermark": "disabled; adapter does not load an additional watermark model",
                "warning": "Experimental latent interpolation; alpha is not a calibrated percentage of two people. Singing is unvalidated. This is recorded-clip CPU inference, not a live streaming engine.",
            }
    metadata_file = output.with_suffix(".json")
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(output), "audio_path": str(output), "metadata_path": str(metadata_file), "metadata": metadata, "sample_rate": sr, "duration": metadata["output_seconds"], "elapsed_s": metadata["total_seconds"], "warning": metadata["warning"]}
