"""Validate source, assets, imports and real CPU preprocessing before training."""
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RVC = ROOT / "upstream/RVC"


def main():
    os.environ["RVC_AUDIO_FORCE_CPU"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(ROOT))
    from voicelab.core import sha256
    from voicelab.rvc_compare import verify_corpus
    from voicelab.rvc_assets import ASSETS, destination

    manifest = verify_corpus(ROOT / "data/self_rvc_v1")
    snapshot = json.loads((ROOT / "RVC_SNAPSHOT.json").read_text(encoding="utf-8"))
    for rel, expected in snapshot["files"].items():
        if sha256(RVC / rel) != expected:
            raise ValueError(f"Changed upstream source: {rel}")
    for name, expected in ASSETS.items():
        path = destination(name)
        if not path.is_file() or (expected and sha256(path) != expected):
            raise ValueError(f"Missing or changed asset: {name}")
    for module in ("torch", "torchaudio", "librosa", "parselmouth", "transformers", "av"):
        importlib.import_module(module)
    env = os.environ | {"PYTHONPATH": str(RVC), "OPENBLAS_NUM_THREADS": "1"}
    subprocess.run([sys.executable, "-c", "from train import utils; import infer.hubert; import infer.rmvpe"],
                   cwd=RVC, env=env, check=True)
    recording = min((row for row in manifest["records"] if row["split"] == "train"),
                    key=lambda row: row["seconds"])
    with tempfile.TemporaryDirectory(prefix="voice_preflight_", dir=ROOT / "runs") as temp:
        import shutil
        temp = Path(temp)
        inputs, output = temp / "input", temp / "features"
        inputs.mkdir()
        output.mkdir()  # Upstream opens its log before creating folders.
        shutil.copy2(ROOT / "data/self_rvc_v1" / recording["path"], inputs / "sample.wav")
        subprocess.run([sys.executable, "-m", "train.preprocess", str(inputs), "40000", "1",
                        str(output), "True", "3.7"], cwd=RVC, env=env, check=True)
        wavs = list((output / "0_gt_wavs").glob("*.wav"))
        if not wavs:
            raise RuntimeError("Preprocessing completed without output WAVs")
        print(json.dumps({"status": "preflight_passed", "source_verified": len(snapshot["files"]),
                          "assets_verified": len(ASSETS), "real_recording_wav_chunks": len(wavs),
                          "training_updates": 0}), flush=True)


if __name__ == "__main__":
    (ROOT / "runs").mkdir(exist_ok=True)
    main()
