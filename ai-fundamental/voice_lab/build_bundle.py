"""Package a source-only starter. Never package personal recordings or trained weights."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT.parent / "dist"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    upstream = ROOT / "upstream/DDSP-SVC"
    vendor_files = [p for p in upstream.rglob("*") if p.is_file() and
                    not any(part in {".git", "__pycache__"} for part in p.relative_to(upstream).parts) and
                    p.suffix.lower() not in {".pt", ".pth", ".bin", ".wav", ".npy", ".pyc", ".zip"}]
    snapshot = {"sha": "d6dd52fcdbce07fbe4c4d120c911b0d63d53e2d2",
                "files": {p.relative_to(upstream).as_posix(): digest(p) for p in sorted(vendor_files)}}
    (ROOT / "UPSTREAM_SNAPSHOT.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    files = [ROOT / name for name in ("README.md", "TUTORIAL.md", "SOURCES_AND_ROUTE.md",
              "RESEARCH_REPORT.md", "sources.json", "UPSTREAM_SNAPSHOT.json", "manifest.example.csv",
              "listening_scores.example.csv", "requirements-data.txt", "requirements-cloud.txt",
              "setup_cloud.sh", "listen.ipynb", "build_bundle.py", "runs/local_validation.json")]
    files += list((ROOT / "voicelab").glob("*.py"))
    files += list((ROOT / "tests").glob("test_*.py"))
    files += vendor_files
    files = sorted(set(files))
    manifest = {"files": {p.relative_to(ROOT).as_posix(): digest(p) for p in files},
                "personal_recordings_included": False, "pretrained_weights_included": False,
                "gpu_training_verified": False}
    DIST.mkdir(parents=True, exist_ok=True)
    bundle = DIST / "voice_lab_autodl_starter.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2))
        for folder in ("recordings/1/", "recordings/2/", "recordings/stress/", "configs/"):
            z.writestr(folder, "")
    info = {"path": str(bundle), "bytes": bundle.stat().st_size, "sha256": digest(bundle), "files": len(files)}
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
