"""Read-only numerical/code audit of the completed RVC learning experiment.

No inference or training is run, and no original audio/checkpoint is modified.
Run from the project root with .venv/Scripts/python.exe -X utf8 <this file>.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/new4090d_ab_retry1"
OUT = Path(__file__).resolve().parent
RVC = ROOT / "upstream/RVC"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def power_fraction(f, p, lo, hi):
    return float(p[(f >= lo) & (f < hi)].sum() / max(p.sum(), 1e-24))


def audio_audit():
    audio = {}
    rows = {}
    for name, rel in {"input": "listening/input.wav", "scratch": "listening/scratch.wav",
                      "finetune": "listening/finetune.wav", "X": "blind/X.wav", "Y": "blind/Y.wav"}.items():
        p = SOURCE / rel
        y, sr = sf.read(p, dtype="float64", always_2d=True)
        info = sf.info(p)
        assert sr == 40000 and y.shape[1] == 1
        y = y[:, 0]
        audio[name] = y
        f, psd = signal.welch(y, sr, nperseg=8192)
        rows[name] = {
            "sha256": sha(p), "channels": info.channels, "samplerate": sr,
            "subtype": info.subtype, "samples": len(y), "seconds": len(y) / sr,
            "peak": float(np.max(np.abs(y))),
            "rms_dbfs": float(20 * np.log10(max(np.sqrt(np.mean(y*y)), 1e-12))),
            "fraction_abs_ge_0_99": float(np.mean(np.abs(y) >= .99)),
            "dc_mean": float(np.mean(y)),
            "power_fraction_0_80_hz": power_fraction(f, psd, 0, 80),
            "power_fraction_80_500_hz": power_fraction(f, psd, 80, 500),
            "power_fraction_500_4000_hz": power_fraction(f, psd, 500, 4000),
            "power_fraction_4000_16000_hz": power_fraction(f, psd, 4000, 16000),
            "power_fraction_16000_20001_hz": power_fraction(f, psd, 16000, 20001),
        }
    for label, arm in (("X", "scratch"), ("Y", "finetune")):
        y, x = audio[label], audio[arm]
        gain = float(np.dot(x, y) / np.dot(x, x))
        residual = y - gain*x
        rows[label]["scaled_raw_comparison"] = {
            "raw_arm": arm, "least_squares_gain": gain,
            "gain_db": float(20*np.log10(abs(gain))),
            "correlation": float(np.corrcoef(x, y)[0, 1]),
            "residual_rms_dbfs": float(20*np.log10(max(np.sqrt(np.mean(residual*residual)), 1e-12))),
            "max_abs_residual": float(np.max(np.abs(residual))),
            "pcm16_quantization_step": 1/32768,
        }
    block = 800  # 20 ms, nonoverlapping; preserve exact source frame alignment.
    n = min(len(audio[k]) for k in ("input", "scratch", "finetune")) // block
    env = {k: np.sqrt(np.mean(audio[k][:n*block].reshape(n, block)**2, axis=1))
           for k in ("input", "scratch", "finetune")}
    # A diagnostic proxy for quiet source frames, not a trained voice activity detector.
    threshold = float(np.percentile(env["input"], 95) * 10**(-30/20))
    quiet = env["input"] < threshold
    for k in env:
        active = ~quiet
        rows[k]["source_quiet_frame_test"] = {
            "block_ms": 20, "criterion": "input RMS < input p95 RMS - 30 dB",
            "input_rms_threshold": threshold, "quiet_frame_count": int(quiet.sum()),
            "total_frame_count": n,
            "quiet_rms_dbfs": float(20*np.log10(max(np.sqrt(np.mean(env[k][quiet]**2)), 1e-12))),
            "active_rms_dbfs": float(20*np.log10(max(np.sqrt(np.mean(env[k][active]**2)), 1e-12))),
            "envelope_correlation_with_input": float(np.corrcoef(env[k], env["input"])[0, 1]),
            "caveat": "Aligned quiet-frame proxy; phoneme and synthesis lag can affect this statistic.",
        }

    fig, axs = plt.subplots(3, 2, figsize=(14, 9), constrained_layout=True)
    for i, k in enumerate(("input", "scratch", "finetune")):
        y = audio[k]
        f, t, z = signal.stft(y, 40000, nperseg=2048, noverlap=1536, boundary=None)
        keep = f <= 8000
        db = 20*np.log10(np.maximum(np.abs(z[keep]), 1e-7))
        m = axs[i, 0].pcolormesh(t, f[keep], db, shading="auto", cmap="magma", vmin=-85, vmax=-25)
        axs[i, 0].set(title=k + " | identical absolute dBFS scale", ylabel="Hz", xlabel="seconds")
        tt = np.arange(n)*.02
        axs[i, 1].plot(tt, 20*np.log10(np.maximum(env[k], 1e-7)), label=k)
        axs[i, 1].plot(tt, 20*np.log10(np.maximum(env["input"], 1e-7)), alpha=.45, label="input")
        axs[i, 1].set(title="20 ms RMS envelope", ylim=(-100, -5), xlabel="seconds", ylabel="dBFS")
        axs[i, 1].legend()
    fig.colorbar(m, ax=axs[:, 0], label="STFT magnitude dBFS")
    fig.savefig(OUT / "spectrogram_envelope.png", dpi=160)
    plt.close(fig)
    return rows


def weight_audit():
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(RVC))
    from infer.module.models import SynthesizerTrnMs768NSFsid
    result = {}
    for arm in ("scratch", "finetune"):
        p = SOURCE / arm / "model.pth"
        c = torch.load(p, map_location="cpu", weights_only=False)
        config = list(c["config"])
        config[-3] = c["weight"]["emb_g.weight"].shape[0]
        net = SynthesizerTrnMs768NSFsid(*config, is_half=False)
        del net.enc_q
        loaded = net.load_state_dict(c["weight"], strict=False)
        # Recorded upstream uses strict=False; explicitly check its returned compatibility.
        result[arm] = {
            "sha256": sha(p), "version": c.get("version"), "f0": c.get("f0"),
            "sr_metadata": c.get("sr"), "sample_rate_config": config[-1],
            "speaker_embedding_shape": list(c["weight"]["emb_g.weight"].shape),
            "tensor_count": len(c["weight"]),
            "missing_inference_keys": list(loaded.missing_keys),
            "unexpected_inference_keys": list(loaded.unexpected_keys),
            "all_finite": all(bool(torch.isfinite(x).all()) for x in c["weight"].values()),
            "dtypes": sorted(set(str(x.dtype) for x in c["weight"].values())),
        }
        start = torch.load(SOURCE / arm / "checkpoints/step0.pth", map_location="cpu", weights_only=False)
        result[arm]["changed_tensors_since_step0"] = sum(
            not torch.equal(x, start["weight"][k]) for k, x in c["weight"].items())
        del net, c, start
    official = RVC / "assets/pretrained_v2/f0G40k.pth"
    if official.is_file():
        official_c = torch.load(official, map_location="cpu", weights_only=False)["model"]
        start = torch.load(SOURCE / "finetune/checkpoints/step0.pth", map_location="cpu", weights_only=False)["weight"]
        matched, mismatch, absent = [], [], []
        for k, v in start.items():
            if k not in official_c:
                absent.append(k)
            elif torch.equal(v, official_c[k].half()):
                matched.append(k)
            else:
                mismatch.append(k)
        result["finetune"]["step0_vs_official_pretrained"] = {
            "official_sha256": sha(official), "matched_exported_half_tensor_count": len(matched),
            "mismatched_keys": mismatch, "absent_official_keys": absent,
            "note": "Training export intentionally omits posterior encoder enc_q; compares exported inference tensors only.",
        }
    return result


def static_evidence():
    return {
        "blind_postprocessing": "rvc_compare.py:249-256 only multiplies each raw output by one gain then PCM16 quantizes; no second track added.",
        "rms_mix_rate": "Recorded --rms-mix-rate 1 skips change_rms. That function changes loudness envelope and is not timbre or track mixing.",
        "chunking": "Recorded 20 s input; fp16 4090 D config x_max=65 s. No multi-chunk loop in this sample. Output chunks would be concatenated, not summed.",
        "f0_zero_interpolation": "Both train/dataset/extract_f0.py and infer/vc/pipeline.py interpolate ALL f0==0 positions from voiced positions.",
        "nsf_voicing": "infer/module/models.py SineGen._f02uv uses f0 > voiced_threshold with threshold 0; thus interpolated positive F0 makes originally unvoiced positions harmonic-source eligible.",
        "protect": "Inference index_rate=0 means retrieval blend is absent; protect blends identical HuBERT feature streams. Additionally F0 zero interpolation removes the low-F0 mask. No claim that protect=.33 fixes artifacts is warranted here.",
        "precision": "Training batch math was FP32; upstream exports small .pth tensors as FP16; recorded inference is FP16. 'FP32 experiment' should not imply FP32 checkpoint storage or inference.",
        "speaker_count": "Only speaker ID 0 has user training samples. 109 embedding rows are architecture metadata, not 109 trained target singers and contain no Adele training.",
    }


if __name__ == "__main__":
    started = time.monotonic()
    record = {"scope": "Local code, WAV and checkpoint audit; no listening claim and no model rerun.",
              "audio": audio_audit(), "weights": weight_audit(), "static_evidence": static_evidence(),
              "limitations": ["Mono files can still contain summed tracks; mono alone does not rule out polyphony.",
                              "Code absence of waveform addition rules out the wrapper summing tracks, not neural artifacts.",
                              "F0 interpolation is a testable cause candidate, not an experimentally isolated root cause.",
                              "Spectral power and pitch estimators are diagnostics, not naturalness or identity scores."]}
    record["audit_wall_seconds"] = time.monotonic() - started
    (OUT / "AUDIT.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"written": ["AUDIT.json", "spectrogram_envelope.png"],
                      "weights": record["weights"], "blind_scaling": {k: record["audio"][k]["scaled_raw_comparison"] for k in ("X", "Y")}}, ensure_ascii=False, indent=2))
