"""Build a minimal, verified ZIP for the first AutoDL session."""
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "ai_fundamental_autodl_starter.zip"


def files_to_include():
    fixed = [
        "AUTODL_START_HERE.md", "BEGINNER_TUTORIAL.md", "DATA_PILOT_README.md", "LESSONS.md", "README.md",
        "requirements.txt", "requirements-data.txt", "requirements-report.txt",
        "tokenizer.json", "raw/demo.jsonl",
        "runs/chat_toy.pt", "runs/smoke/base.pt",
    ]
    paths = [ROOT / relative for relative in fixed]
    paths += sorted((ROOT / "fundamental").glob("*.py"))
    paths += sorted((ROOT / "tests").glob("*.py"))
    paths += sorted((ROOT / "scripts").glob("*.py"))
    paths += sorted((ROOT / "report").glob("*.md"))
    paths += sorted((ROOT / "report" / "data").glob("*.jsonl"))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing bundle inputs: " + ", ".join(missing))
    return paths


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files_to_include():
            name = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            archive.writestr(name, data)
            manifest[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(OUT) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC check failed: {bad}")
        assert set(archive.namelist()) == set(manifest) | {"BUNDLE_MANIFEST.json"}
    print(json.dumps({"zip": str(OUT), "bytes": OUT.stat().st_size, "files": len(manifest)}))


if __name__ == "__main__":
    main()
