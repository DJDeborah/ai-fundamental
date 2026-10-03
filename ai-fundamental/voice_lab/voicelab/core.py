import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "DDSP-SVC"
UPSTREAM_SHA = "d6dd52fcdbce07fbe4c4d120c911b0d63d53e2d2"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def environment():
    packages = {}
    for name in ("torch", "torchaudio", "numpy", "soundfile", "librosa", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "upstream_sha": UPSTREAM_SHA}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_upstream(argv, log_path):
    """Keep the upstream working directory explicit; preserve complete console output."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen([sys.executable, "-u", *map(str, argv)], cwd=UPSTREAM,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            code = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    if code:
        raise RuntimeError(f"Upstream exited {code}; see {log_path}")
