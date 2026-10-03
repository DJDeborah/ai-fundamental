"""Only the six assets needed for this pinned RVC comparison, with hashes."""
import json
import time
import urllib.request
from pathlib import Path

from .core import ROOT, sha256, write_json

RVC = ROOT / "upstream/RVC"
REVISION = "e6d0c1a17da07c33557852f9dfa2bd44cc75737d"
REPOSITORY = "lj1995/VoiceConversionWebUI"
ASSETS = {
    "hubert_base/config.json": None,
    "hubert_base/preprocessor_config.json": None,
    "hubert_base/pytorch_model.bin": "cc8c20f4b90a520757260197a3ff2505705a7adbd20ad9eeaa4e1a9b38442ef5",
    "pretrained_v2/f0G40k.pth": "3b2c44035e782c4b14ddc0bede9e2f4a724d025cd073f736d4f43708453adfcb",
    "pretrained_v2/f0D40k.pth": "6b6ab091e70801b28e3f41f335f2fc5f3f35c75b39ae2628d419644ec2b0fa09",
    "rmvpe.pt": "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193",
}


def destination(name):
    return RVC / "assets" / ("rmvpe/rmvpe.pt" if name == "rmvpe.pt" else name)


def download(output):
    # Operator-controlled upload gate for hosts without access to Hugging Face.
    pending = ROOT / "ASSETS_UPLOAD_PENDING"
    deadline = time.monotonic() + 900
    if pending.exists():
        print("Waiting for verified asset upload (at most 15 minutes)", flush=True)
    while pending.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Verified asset upload was not completed within 15 minutes")
        time.sleep(5)
    records = []
    for name, expected in ASSETS.items():
        target = destination(name)
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}?download=true"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            print(f"Downloading {name}", flush=True)
            temporary = target.with_suffix(target.suffix + ".partial")
            req = urllib.request.Request(url, headers={"User-Agent": "voice-lab/1.0"})
            with urllib.request.urlopen(req, timeout=90) as src, temporary.open("wb") as dst:
                while block := src.read(1024 * 1024):
                    dst.write(block)
            temporary.replace(target)
        digest = sha256(target)
        if expected and digest != expected:
            raise ValueError(f"Wrong asset SHA256: {target}")
        if name.endswith(".json"):
            json.loads(target.read_text(encoding="utf-8"))
        records.append({"path": name, "url": url, "bytes": target.stat().st_size,
                        "sha256": digest, "expected_sha256": expected})
    write_json(Path(output), {"repository": REPOSITORY, "revision": REVISION,
                             "assets": records, "license_note": "Repository/model card labels are recorded sources, not a complete commercial clearance."})
    return records
