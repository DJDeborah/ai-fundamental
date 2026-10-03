import argparse
import json
import sys
import time
from pathlib import Path

from .core import ROOT, UPSTREAM, append_json, environment, run_upstream, sha256, utc_now, write_json


def doctor(require_gpu=False):
    import torch
    info = environment()
    info.update(cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    if torch.cuda.is_available():
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    try:
        import torchaudio
        info["torchaudio_import"] = "OK"
    except Exception as e:
        info["torchaudio_import"] = str(e)
    info["missing_assets"] = [p for p in ("contentvec/pytorch_model.bin", "nsf_hifigan/model",
                                              "nsf_hifigan/config.json", "rmvpe/model.pt")
                              if not (UPSTREAM / "pretrain" / p).is_file()]
    snapshot = ROOT / "UPSTREAM_SNAPSHOT.json"
    if snapshot.exists():
        expected = json.loads(snapshot.read_text(encoding="utf-8"))["files"]
        changed = [p for p, digest in expected.items()
                   if not (UPSTREAM / p).is_file() or sha256(UPSTREAM / p) != digest]
        info["upstream_snapshot_matches"] = not changed
        info["changed_upstream_files"] = changed
    write_json(ROOT / "runs/doctor.json", info)
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; inspect runs/doctor.json")
    return info


def report(output, dataset=None):
    sources = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    summaries = []
    for path in sorted((ROOT / "runs").rglob("summary_*.json")):
        summaries.append({"file": str(path.relative_to(ROOT)), **json.loads(path.read_text(encoding="utf-8"))})
    body = ["# 歌声音色转换实验记录", "", "本文件由实际保存的记录生成。未执行的训练与试听没有结果。", "",
            "## 实现与来源", "", f"DDSP-SVC 固定版本 `{sources['ddsp_sha']}`。", "",
            "默认 PC-NSF-HiFiGAN 权重采用 CC BY-NC-SA 4.0，本组合用于非商业学习实验。", "",
            "## 数据", ""]
    if dataset:
        manifest = Path(dataset) / "manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        body += [f"数据 manifest SHA256 `{sha256(manifest)}`。", "", "```json",
                 json.dumps(data["totals"], ensure_ascii=False, indent=2), "```", ""]
    else:
        body += ["尚未指定真实录音数据集。", ""]
    body += ["## 云端训练", ""]
    if not summaries:
        body += ["未发现云端训练 summary；不能声称模型已训练或声音质量已改善。", ""]
    for summary in summaries:
        body += [f"### {summary['file']}", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""]
    body += ["## 试听与客观指标", "", "F0 误差只衡量音高保持。音色相似度、自然度和比例单调性需盲听对照；RMS 不是 LUFS。", ""]
    for glob in ("inference.json", "*_evaluation.json"):
        for path in sorted((ROOT / "runs").rglob(glob)):
            body += [f"### {path.relative_to(ROOT)}", "", "```json", path.read_text(encoding="utf-8").strip(), "```", ""]
    body += ["## 下一轮对照设计", "", "在固定测试录音、模型、推理步数和随机种子下比较 checkpoint 0 与训练后模型。",
             "双歌手时扫描 alpha=0,0.25,0.5,0.75,1；检查音高、歌词、自然度与身份变化。",
             "修音 beta、任意用户零样本音色插值、伴奏分离和混音尚未接入此训练实验。", ""]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(body), encoding="utf-8")
    return {"report": str(output), "training_summaries": len(summaries)}


def main():
    parser = argparse.ArgumentParser(description="A step-by-step singing voice conversion lab")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor")
    p.add_argument("--require-gpu", action="store_true")
    p = sub.add_parser("inventory")
    p.add_argument("--recordings", type=Path, default=ROOT / "recordings")
    p.add_argument("--out", type=Path, default=ROOT / "runs/inventory.json")
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--recordings", type=Path, default=ROOT / "recordings")
    p.add_argument("--out", type=Path, default=ROOT / "data/self_v1")
    p.add_argument("--chunk-seconds", type=float, default=8)
    p = sub.add_parser("config")
    p.add_argument("--data", type=Path, default=ROOT / "data/self_v1")
    p.add_argument("--out", type=Path, default=ROOT / "configs/self_v1.yaml")
    p.add_argument("--run-name", default="self_v1")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--profile", choices=["learning", "upstream"], default="learning")
    sub.add_parser("assets")
    p = sub.add_parser("preprocess")
    p.add_argument("--config", type=Path, default=ROOT / "configs/self_v1.yaml")
    p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("train")
    p.add_argument("--config", type=Path, default=ROOT / "configs/self_v1.yaml")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--minutes", type=float, default=15)
    p.add_argument("--resume", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-clips", type=int, default=3)
    p = sub.add_parser("infer")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("--data", type=Path)
    p.add_argument("--out", type=Path, default=ROOT / "runs/REPORT.md")
    args = parser.parse_args()
    # Resolve input paths before upstream code changes the working directory.
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    record = {"started_utc": utc_now(), "command": args.command,
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "environment": environment()}
    started = time.perf_counter()
    try:
        if args.command == "doctor":
            result = doctor(args.require_gpu)
        elif args.command == "inventory":
            from .audio import inventory
            result = inventory(args.recordings, args.out)
        elif args.command == "prepare":
            from .audio import prepare
            result = prepare(args.manifest, args.recordings, args.out, args.chunk_seconds)
        elif args.command == "config":
            from .training import make_config
            result = make_config(args.data, args.out, args.run_name, args.batch_size, args.profile)
        elif args.command == "assets":
            from .assets import fetch_assets
            result = fetch_assets()
        elif args.command == "preprocess":
            import yaml
            config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            dataset = Path(config["data"]["train_path"]).parent
            prepared = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
            for clip in prepared["chunks"]:
                if sha256(dataset / clip["path"]) != clip["sha256"]:
                    raise ValueError(f"Prepared audio changed: {clip['path']}")
            run_upstream([str(ROOT / "voicelab/vendor_entry.py"), "--script", "preprocess.py", "-c", args.config,
                          "-j", args.workers], ROOT / "runs/preprocess.log")
            files = [{"path": p.relative_to(dataset).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size}
                     for split in ("train", "val") for p in sorted((dataset / split).rglob("*.npy"))]
            if not files:
                raise ValueError("Preprocessing did not produce .npy features")
            write_json(dataset / "features.json", {"files": files, "config_sha256": sha256(args.config),
                       "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
                       "asset_manifest_sha256": sha256(ROOT / "runs/assets.json")})
            result = {"status": "preprocessed", "config_sha256": sha256(args.config),
                      "features_sha256": sha256(dataset / "features.json"), "feature_files": len(files)}
        elif args.command == "train":
            from .training import train
            result = train(args.config, args.steps, args.minutes, args.resume, args.seed, args.val_clips)
        elif args.command == "infer":
            from .inference import convert
            result = convert(args.input, args.checkpoint, args.alphas, args.out, args.steps, args.seed)
        elif args.command == "evaluate":
            from .inference import evaluate
            result = evaluate(args.input, args.converted, args.out)
        else:
            result = report(args.out, args.data)
        record.update(status="completed", result=result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record.update(wall_s=time.perf_counter() - started, finished_utc=utc_now())
        append_json(ROOT / "runs/commands.jsonl", record)
