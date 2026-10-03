"""Download only the three explicitly named upstream assets; keep provenance."""
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .core import ROOT, UPSTREAM, sha256, utc_now, write_json

VOCODER_URL = "https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip"
RMVPE_URL = "https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip"


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temp = target.with_suffix(target.suffix + ".part")
        print(f"Downloading {target.name} from official source", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as f:
            shutil.copyfileobj(response, f, length=1024 * 1024)
        temp.replace(target)
    if target.stat().st_size < 100:
        raise ValueError(f"Invalid download: {target}")
    return {"url": url, "path": str(target.relative_to(ROOT)),
            "bytes": target.stat().st_size, "sha256": sha256(target)}


def safe_extract(archive, directory):
    root = Path(directory).resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            path = (root / info.filename).resolve()
            if not path.is_relative_to(root) or info.external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe archive member: {info.filename}")
        z.extractall(root)


def fetch_assets():
    specs = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    revision = specs["contentvec_revision"]
    if revision == "main":
        raise ValueError("ContentVec revision must be pinned in sources.json")
    records = []
    records.append(download(f"https://huggingface.co/lengyue233/content-vec-best/resolve/{revision}/pytorch_model.bin",
                            UPSTREAM / "pretrain/contentvec/pytorch_model.bin"))
    for name, url in (("vocoder", VOCODER_URL), ("rmvpe", RMVPE_URL)):
        archive = ROOT / "downloads" / (name + ".zip")
        records.append(download(url, archive))
        destination = ROOT / "downloads" / name
        safe_extract(archive, destination)
        if name == "vocoder":
            configs = list(destination.rglob("config.json"))
            if len(configs) != 1:
                raise ValueError(f"Expected one vocoder config.json, found {configs}")
            candidates = [p for p in configs[0].parent.iterdir() if p.is_file() and
                          (p.name == "model" or p.suffix in {".pt", ".pth", ".ckpt"})]
            if len(candidates) != 1:
                raise ValueError(f"Inspect vocoder archive: expected one checkpoint, found {candidates}")
            target = UPSTREAM / "pretrain/nsf_hifigan"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(configs[0], target / "config.json")
            shutil.copy2(candidates[0], target / "model")
        else:
            candidates = [p for p in destination.rglob("*.pt") if p.name in {"model.pt", "rmvpe.pt"}]
            if len(candidates) != 1:
                raise ValueError(f"Inspect RMVPE archive: {candidates}")
            target = UPSTREAM / "pretrain/rmvpe/model.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidates[0], target)
    installed = {}
    for rel in ("contentvec/pytorch_model.bin", "nsf_hifigan/model", "nsf_hifigan/config.json", "rmvpe/model.pt"):
        path = UPSTREAM / "pretrain" / rel
        installed[rel] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    write_json(ROOT / "runs/assets.json", {"downloaded_utc": utc_now(), "downloads": records,
                                          "installed": installed, "license_notes": specs["licenses"]})
    return {"status": "downloaded", "record": "runs/assets.json",
            "commercial_status": "Default vocoder weights are CC BY-NC-SA 4.0; learning use only."}
