"""Package verified public RVC assets for a cloud host without HF access."""
import json
import zipfile
from pathlib import Path

from voicelab.core import ROOT, sha256
from voicelab.rvc_assets import ASSETS, REPOSITORY, REVISION, destination


def main():
    archive = ROOT.parent / "dist/voice_self_rvc_assets.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for name, expected in ASSETS.items():
            source = destination(name)
            digest = sha256(source)
            if expected and digest != expected:
                raise ValueError(f"Asset checksum mismatch: {name}")
            if name.endswith(".json"):
                json.loads(source.read_text(encoding="utf-8"))
            relative = source.relative_to(ROOT).as_posix()
            files[relative] = digest
            bundle.write(source, relative)
        bundle.writestr("ASSETS_UPLOAD_MANIFEST.json", json.dumps({
            "repository": REPOSITORY, "revision": REVISION, "files": files,
            "transport": "Verified local download followed by upload",
        }, ensure_ascii=False, indent=2))
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": sha256(archive), "asset_files": len(files)}))


if __name__ == "__main__":
    main()
