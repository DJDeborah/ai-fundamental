"""Verify an uploaded asset archive before releasing the cloud upload gate."""
import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    if sha256(args.archive) != args.sha256:
        raise ValueError("Uploaded archive is incomplete or its SHA256 is wrong")
    with zipfile.ZipFile(args.archive) as bundle:
        manifest = json.loads(bundle.read("ASSETS_UPLOAD_MANIFEST.json"))
        if len(manifest["files"]) != 6:
            raise ValueError("Expected exactly six assets")
        for name in bundle.namelist():
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Archive path escapes the project")
        for name, expected in manifest["files"].items():
            target = (root / name).resolve()
            if not name.startswith("upstream/RVC/assets/") or not target.is_relative_to(root):
                raise ValueError("Unexpected asset destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha256(target) != expected:
                    raise ValueError(f"Existing asset differs: {name}")
            else:
                temporary = target.with_suffix(target.suffix + ".uploading")
                with bundle.open(name) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if sha256(temporary) != expected:
                    raise ValueError(f"Asset SHA256 mismatch: {name}")
                temporary.replace(target)
    receipt = {**manifest, "archive_sha256": args.sha256, "verified_files": 6,
               "utc_unix": time.time()}
    (root / "ASSETS_UPLOAD_MANIFEST.json").write_text(json.dumps(receipt, indent=2))
    output = root / "runs/self_rvc_compare"
    if output.exists():
        (output / "asset_upload.json").write_text(json.dumps(receipt, indent=2))
        patch = root / "ASSETS_UPLOAD_PATCH.json"
        if patch.exists():
            shutil.copy2(patch, output / patch.name)
    (root / "ASSETS_UPLOAD_PENDING").unlink(missing_ok=True)
    print("VOICE_ASSET_UPLOAD_VERIFIED", json.dumps({"verified_files": 6,
          "archive_sha256": args.sha256, "gate_released": True}), flush=True)


if __name__ == "__main__":
    main()
