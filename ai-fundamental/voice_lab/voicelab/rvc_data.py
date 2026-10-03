"""Read-only preparation of this user's six recordings; split BEFORE slicing."""
import argparse
import json
from pathlib import Path

import soundfile as sf

from .audio import audio_stats, load_audio
from .core import sha256, write_json

# Provisional recording-level split. Song identity must be confirmed by the singer.
SPLITS = {"1234": "train", "19": "train", "57": "train", "63": "val", "92": "test", "93": "train"}


def prepare(source, output, groups=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty destination; original recordings are never changed")
    records, seen_groups = [], {}
    files = sorted(source.glob("*.m4a"))
    if len(files) != 6:
        raise ValueError("This manifest expects exactly the six inventoried M4A recordings")
    for index, path in enumerate(files, 1):
        suffix = path.stem.split()[-1]
        split = SPLITS[suffix]
        group = (groups or {}).get(suffix, f"recording_{suffix}")
        if group in seen_groups and seen_groups[group] != split:
            raise ValueError(f"Song group {group} crosses splits; revise SPLITS before preparing")
        seen_groups[group] = split
        samples, metadata = load_audio(path, sample_rate=40000)
        rel = Path(split) / f"r{index:03d}.wav"
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dest, samples, 40000, subtype="PCM_16")
        records.append({"alias": f"r{index:03d}", "original_suffix": suffix, "split": split,
                        "group": group, "path": rel.as_posix(), "source_sha256": sha256(path),
                        "sha256": sha256(dest), "decoded_sample_rate": 40000,
                        **audio_stats(samples, 40000)})
    totals = {s: sum(r["seconds"] for r in records if r["split"] == s) for s in ("train", "val", "test")}
    result = {"sample_rate": 40000, "channels": 1, "records": records, "seconds": totals,
              "speaker_id": 0, "grouping_confirmed": groups is not None,
              "quality_source": "User describes a cappella; clipping inspected numerically, accompaniment/reverb not verified by listening.",
              "limitations": "One singer, ~6 minutes, one validation and one test recording. Recording-level grouping does not establish unseen-song generalization.",
              "policy": "Decode, mono resample, PCM16. No denoising, pitch correction or volume amplification. Only train/ enters RVC preprocessing."}
    write_json(output / "manifest.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--groups", help="JSON mapping of recording suffix to song ID")
    a = p.parse_args()
    groups = json.loads(Path(a.groups).read_text(encoding="utf-8")) if a.groups else None
    print(json.dumps(prepare(a.source, a.out, groups), ensure_ascii=False, indent=2))
