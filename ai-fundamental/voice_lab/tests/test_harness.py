import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voicelab.audio import prepare, read_manifest
from voicelab.assets import safe_extract
from voicelab.inference import mix_weights, pitch_metrics
from voicelab.training import make_config


def fixture_manifest(tmp_path):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    rows = []
    for i, split in enumerate(("train", "val", "test")):
        t = np.arange(6 * 16000) / 16000
        audio = 0.2 * np.sin(2 * np.pi * (200 + i * 40) * t)
        sf.write(recordings / f"{split}.wav", audio, 16000)
        rows.append(dict(speaker_id=1, group=f"song_{i}", split=split, path=f"{split}.wav",
                         consent="self", quality="clean"))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return recordings, manifest, rows


def test_prepare_resample_split_and_config(tmp_path):
    recordings, manifest, _ = fixture_manifest(tmp_path)
    out = tmp_path / "prepared"
    prepare(manifest, recordings, out)
    data = json.loads((out / "manifest.json").read_text())
    assert len(data["chunks"]) == 3
    for c in data["chunks"]:
        signal, sr = sf.read(out / c["path"])
        assert sr == 44100 and signal.ndim == 1 and len(signal) == 6 * 44100
    config = tmp_path / "config.yaml"
    result = make_config(out, config, "unit_test", profile="learning")
    assert result["speakers"] == 1
    assert config.with_suffix(".metadata.json").is_file()
    with pytest.raises(ValueError, match="not empty"):
        prepare(manifest, recordings, out)


def test_reject_group_leakage(tmp_path):
    recordings, manifest, rows = fixture_manifest(tmp_path)
    text = manifest.read_text()
    manifest.write_text(text.replace("song_1", "song_0"))
    with pytest.raises(ValueError, match="leaks"):
        read_manifest(manifest, recordings)


def test_reject_duplicate_source(tmp_path):
    recordings, manifest, _ = fixture_manifest(tmp_path)
    (recordings / "val.wav").write_bytes((recordings / "train.wav").read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(manifest, recordings)


def test_reject_noisy_training_and_path_escape(tmp_path):
    recordings, manifest, _ = fixture_manifest(tmp_path)
    original = manifest.read_text()
    manifest.write_text(original.replace("self,clean", "self,noisy", 1))
    with pytest.raises(ValueError, match="marked clean"):
        read_manifest(manifest, recordings)
    manifest.write_text(original.replace("train.wav", "../outside.wav"))
    with pytest.raises(ValueError, match="outside"):
        read_manifest(manifest, recordings)


def test_mix_endpoints_and_single_speaker():
    assert mix_weights(0, 2) == {1: 1, 2: 0}
    assert mix_weights(1, 2) == {1: 0, 2: 1}
    assert mix_weights(0.25, 2) == {1: 0.75, 2: 0.25}
    assert mix_weights(0, 1) is None
    with pytest.raises(ValueError):
        mix_weights(0.5, 1)
    with pytest.raises(ValueError):
        mix_weights(2, 2)


def test_pitch_cents_and_voicing():
    values = pitch_metrics(np.array([440, 440, np.nan]), np.array([880, 440, 440]))
    assert values["median_abs_pitch_cents"] == 600
    assert values["voiced_unvoiced_disagreement"] == pytest.approx(1 / 3)
    assert pitch_metrics([np.nan], [np.nan])["median_abs_pitch_cents"] is None


def test_zip_traversal_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, tmp_path / "extracted")
