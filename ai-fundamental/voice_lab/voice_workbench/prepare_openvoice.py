"""Fetch the fixed official OpenVoice V2 converter and only its Python code.

This does not download ASR, TTS, datasets, or the watermark model. Run this
inside the workbench environment after installing requirements-model.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE_REVISION = "74a1d147b17a8c3092dd5430504bd83ef6c7eb23"
MODEL_REVISION = "f36e7edfe1684461a8343844af60babc2efbb727"
CHECKPOINT_SHA256 = "9652c27e92b6b2a91632590ac9962ef7ae2b712e5c5b7f4c34ec55ee2b37ab9e"
CHECKPOINT_BYTES = 131320490


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(url: str, destination: Path, expected: str | None = None) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected and sha256(destination) == expected:
        return {"sha256": expected, "bytes": destination.stat().st_size, "url": url}
    partial = destination.with_name(destination.name + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "voice-workbench-prepare/1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    digest = sha256(partial)
    if expected and digest != expected:
        raise RuntimeError(f"SHA256 mismatch for {destination.name}; incomplete file preserved")
    partial.replace(destination)
    return {"sha256": digest, "bytes": destination.stat().st_size, "url": url}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-only", action="store_true")
    args = parser.parse_args()
    upstream = ROOT / "upstream" / "OpenVoice"
    model = ROOT / "assets" / "openvoice-v2" / "converter"
    receipt_path = ROOT / "assets" / "openvoice-v2" / "PREPARATION_RECEIPT.json"
    started = time.time()
    tree_url = f"https://api.github.com/repos/myshell-ai/OpenVoice/git/trees/{CODE_REVISION}?recursive=1"
    request = urllib.request.Request(tree_url, headers={"User-Agent": "voice-workbench-prepare/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        tree = json.load(response)
    if tree.get("truncated"):
        raise RuntimeError("Official code tree is truncated")
    files = {}
    for entry in tree["tree"]:
        name = entry["path"]
        if entry["type"] != "blob" or not (name.startswith("openvoice/") and name.endswith(".py") or name in {"LICENSE", "README.md"}):
            continue
        url = f"https://raw.githubusercontent.com/myshell-ai/OpenVoice/{CODE_REVISION}/{name}"
        files["code/" + name] = download(url, upstream / name)
    if not args.code_only:
        base_url = f"https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/{MODEL_REVISION}/converter"
        files["model/config.json"] = download(base_url + "/config.json", model / "config.json")
        files["model/checkpoint.pth"] = download(base_url + "/checkpoint.pth", model / "checkpoint.pth", CHECKPOINT_SHA256)
        if files["model/checkpoint.pth"]["bytes"] != CHECKPOINT_BYTES:
            raise RuntimeError("Checkpoint size mismatch")
        files["model/README.md"] = download(
            f"https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/{MODEL_REVISION}/README.md",
            model.parent / "MODEL_CARD.md",
        )
    receipt = {
        "engine": "OpenVoice V2 official tone color converter",
        "code_revision": CODE_REVISION, "model_revision": MODEL_REVISION,
        "code_license": "MIT", "weights_license": "MIT", "files": files,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": time.time() - started,
        "scope": "Converter only. No datasets, ASR, MeloTTS, watermark weights, or GPU work.",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(receipt_path), "files": len(files), "elapsed_s": receipt["elapsed_s"]}))


if __name__ == "__main__":
    main()
