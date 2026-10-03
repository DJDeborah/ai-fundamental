"""Shared feature extraction, two controlled arms, held-out conversion and records."""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import audio_stats, load_audio
from .core import ROOT, append_json, environment, sha256, utc_now, write_json
from .rvc_assets import RVC, download


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", RVC_CUDA_GRAPH="0", RVC_AUDIO_FORCE_CPU="1",
               rmvpe_root=str(RVC / "assets/rmvpe"), PYTHONPATH=str(RVC), OPENBLAS_NUM_THREADS="1")
    with log.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen([sys.executable, "-u", *map(str, command)], cwd=RVC,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=env, text=True, encoding="utf-8", errors="replace")
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                out.write(line)
                out.flush()
            code = proc.wait()
        except BaseException:
            # Launcher signals the entire process group; wait for boundary saves.
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=15)
            raise
    if code:
        raise RuntimeError(f"Stage exited {code}; see {log}")


def feature_filelist(shared):
    # Every preprocessed WAV must have all three corresponding feature files.
    rows = []
    wavs = sorted((shared / "0_gt_wavs").glob("*.wav"))
    if not wavs:
        raise ValueError("Preprocessing produced no training WAVs")
    for wav in wavs:
        paths = [wav, shared / "3_feature768" / f"{wav.stem}.npy",
                 shared / "2a_f0" / f"{wav.name}.npy", shared / "2b-f0nsf" / f"{wav.name}.npy"]
        for p in paths:
            if not p.is_file():
                raise ValueError(f"Missing feature for {wav.name}: {p}")
        units, f0, pitch = (np.load(p) for p in paths[1:])
        if units.ndim != 2 or units.shape[1] != 768 or len(units) == 0:
            raise ValueError(f"Invalid HuBERT features: {paths[1]}")
        # Silence slicing can leave short but valid tails. Upstream's common
        # bucket sampler determines which lengths enter training for both arms.
        if not all(np.isfinite(x).all() for x in (units, f0, pitch)) or min(len(f0), len(pitch)) == 0:
            raise ValueError(f"Invalid/nonfinite F0 features: {wav}")
        if any("|" in str(p) for p in paths):
            raise ValueError("RVC filelist paths cannot contain a pipe")
        rows.append("|".join(p.as_posix() for p in paths) + "|0")
    return "\n".join(rows) + "\n"


def verify_corpus(data):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    groups = {}
    for row in manifest["records"]:
        relative = Path(row["path"])
        path = (data / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(data.resolve()):
            raise ValueError("Corpus path escapes its root")
        if row["split"] not in {"train", "val", "test"} or relative.parts[0] != row["split"]:
            raise ValueError("Corpus split/path mismatch")
        if groups.setdefault(row["group"], row["split"]) != row["split"]:
            raise ValueError("Same group crosses splits")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Changed training data: {path}")
    declared = {r["path"] for r in manifest["records"] if r["split"] == "train"}
    actual = {p.relative_to(data).as_posix() for p in (data / "train").rglob("*") if p.is_file()}
    if actual != declared:
        raise ValueError("Training folder contains undeclared/missing files")
    return manifest


def export_results(output):
    target = output / "results.zip"
    files = [p for p in output.rglob("*") if p.is_file() and p != target and p.suffix != ".zip"]
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, p.relative_to(output))
    return target


def report(output):
    manifest = json.loads((output / "corpus.json").read_text(encoding="utf-8"))
    summaries = {}
    for arm in ("scratch", "finetune"):
        p = output / arm / "summary.json"
        summaries[arm] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    equal = (all(s and s["status"] == "complete" for s in summaries.values()) and
             summaries["scratch"]["steps"] == summaries["finetune"]["steps"])
    result = {"utc": utc_now(), "arms": summaries, "equal_steps_complete": bool(equal),
              "data_seconds": manifest["seconds"], "song_grouping_confirmed": manifest["grouping_confirmed"],
              "perceptual_winner": None, "limitations": ["One seed and one singer; learning pilot only",
              "Same-singer holdout reconstruction does not test conversion from another singer",
              "Recording-level split until song grouping is confirmed", "No blind listening scores supplied yet"]}
    result["objective_metrics"] = {arm: json.loads((output / "listening" / f"{arm}_metrics.json").read_text(encoding="utf-8"))
        for arm in ("scratch", "finetune") if (output / "listening" / f"{arm}_metrics.json").exists()}
    if manifest["grouping_confirmed"]:
        result["limitations"].remove("Recording-level split until song grouping is confirmed")
    write_json(output / "comparison.json", result)
    rows = "\n".join(f"| {a} | {s['steps'] if s else '待运行'} | {s['status'] if s else '待运行'} | {round(s['wall_s'],1) if s else '—'} |" for a, s in summaries.items())
    text = f"""# 本人清唱 RVC 对照实验记录

生成于 {utc_now()}。同一 RVC v2 40k 架构、同一训练数据、共享 HuBERT/RMVPE；A 随机初始化生成器与判别器，B 加载官方 f0G40k/f0D40k 预训练权重。

## 数据与控制变量

训练 {manifest['seconds']['train']:.2f} 秒；验证 {manifest['seconds']['val']:.2f} 秒；测试 {manifest['seconds']['test']:.2f} 秒。完整录音先划分，再仅对 train 做上游切片。歌曲分组确认：{manifest['grouping_confirmed']}。

共同设置：seed=1234，batch=4，FP32，AdamW，40kHz，v2，F0 enabled；检索索引比例=0，移调=0，推理 seed=1234。上游训练预处理还有 48Hz 高通、静音切片及幅度归一化；两组只预处理一次。

| 组别 | 更新步数 | 状态 | 包含初始化/保存的秒数 |
|---|---:|---|---:|
{rows}

完成相同步数：{bool(equal)}。若为 False，不能宣称已完成受控训练对照。

## 如何体验

打开 rvc_listen.ipynb，选择 RVC Comparison 内核。先听原声和 blind/ 中匿名的 X、Y，填写 listening_scores.csv，再查看 blind_key.json，之后可听两组未匹配响度的原始输出。评分 1–5：自然度、歌词可懂度、音色相似度、音高稳定性；每项应加备注。

原声转回本人用于检查重建保真。因为所有录音都来自本人，此实验还没有评估“另一人的声音转成你”；需要另一位授权歌者的未训练输入才能测这个能力。

## 结论状态

当前没有主观听感胜负结论。训练 loss 不是听感质量；两个初始化的 GAN 总 loss 也不能直接当成质量排名。metrics.jsonl 保存显示截断之前的原始 loss。自动音高评估保存在 listening/*_metrics.json 与 comparison.json，若对应文件缺失表示未完成；它假设输入输出时间对齐，不衡量身份相似度或自然度。

这是低预算学习实验，单个随机种子、约六分钟数据、各一段验证和测试，不能支撑正式论文的统计结论。预训练使用了额外数据，其规模/组成未在本实验复现，因此这里比较的是相同本轮适配成本下的效果，不能声称总训练算力相同。

## 来源

[RVC 官方 CLI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/docs/en/cli.md)；[官方权重仓库](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/e6d0c1a17da07c33557852f9dfa2bd44cc75737d)。上游快照、数据、特征、checkpoint 哈希随结果保存。商业许可需要分别核验代码、权重和数据。
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/self_rvc_v1")
    p.add_argument("--out", default="runs/self_rvc_compare")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--arm-minutes", type=float, default=45)
    a = p.parse_args()
    if a.steps < 1 or a.arm_minutes <= 0:
        raise ValueError("Positive training limits required")
    data, output = Path(a.data).resolve(), Path(a.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Comparison output already exists; use --out runs/new_name")
    manifest = verify_corpus(data)
    from launch_rvc import gpu_processes
    if gpu_processes():
        raise RuntimeError("GPU occupied by another experiment; do not start a second training job")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not start paid CPU training")
    torch.zeros(1, device="cuda").add_(1).cpu()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "corpus.json", manifest)
    write_json(output / "environment.json", environment() | {"rvc_sha": "81eed5e8f68b6bed1789f682fe78cdd324495afc",
        "device": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda})
    for audit_name in ("ASSETS_UPLOAD_MANIFEST.json", "ASSETS_UPLOAD_PATCH.json",
                       "PIPELINE_RUNTIME_PATCH.json", "RVC_NATIVE_STORAGE.json"):
        audit = ROOT / audit_name
        if audit.is_file():
            shutil.copy2(audit, output / audit_name)
    try:
        snapshot = json.loads((ROOT / "RVC_SNAPSHOT.json").read_text(encoding="utf-8"))
        for rel, expected in snapshot["files"].items():
            if sha256(RVC / rel) != expected:
                raise ValueError(f"Changed pinned RVC source: {rel}")
        download(output / "assets.json")
        run_id = "self_" + time.strftime("%Y%m%d_%H%M%S")
        shared = RVC / "logs" / (run_id + "_shared")
        # Upstream opens preprocess.log before PreProcess creates its folders.
        shared.mkdir(parents=True, exist_ok=False)
        # Module execution keeps RVC's root first on sys.path; direct execution
        # of train/preprocess.py makes train/train.py shadow the train package.
        run(["-m", "train.preprocess", data / "train", "40000", "4", shared, "False", "3.7"], output / "preprocess.log")
        run(["-m", "train.dataset.extract_f0", "cuda", "1", "0", "0", shared, "true"], output / "f0.log")
        run(["-m", "train.dataset.extract_hubert_feature", "cuda:0", "1", "0", "0", shared, "v2", "true"], output / "hubert.log")
        filelist = feature_filelist(shared)
        features = {p.relative_to(shared).as_posix(): sha256(p) for p in shared.rglob("*") if p.is_file() and p.suffix in {".wav", ".npy"}}
        write_json(output / "features.json", {"shared_dir": str(shared), "files": features})
        config = json.loads((RVC / "configs/v1/40k.json").read_text(encoding="utf-8"))
        config["train"].update(seed=1234, batch_size=4, epochs=1200, log_interval=200)
        for arm in ("scratch", "finetune"):
            name = run_id + "_" + arm
            native = RVC / "logs" / name
            native.mkdir(parents=True)
            write_json(native / "config.json", config)
            (native / "filelist.txt").write_text(filelist, encoding="utf-8")
            arm_dir = output / arm
            arm_dir.mkdir()
            shutil.copy2(native / "config.json", arm_dir / "config.json")
            shutil.copy2(native / "filelist.txt", arm_dir / "filelist.txt")
            args = ["-e", name, "-sr", "40k", "-f0", "1", "-bs", "4", "-g", "0", "-te", "1200", "-se", "1200", "-l", "1", "-c", "0", "-sw", "0", "-v", "v2"]
            if arm == "finetune":
                args += ["-pg", str(RVC / "assets/pretrained_v2/f0G40k.pth"), "-pd", str(RVC / "assets/pretrained_v2/f0D40k.pth")]
            command = [ROOT / "voicelab/rvc_bridge.py", "--output", arm_dir, "--steps", str(a.steps),
                       "--seconds", str(a.arm_minutes * 60), "--checkpoint-every", str(min(500, a.steps)), "--", *args]
            append_json(output / "commands.jsonl", {"utc": utc_now(), "arm": arm, "args": list(map(str, command))})
            run(command, arm_dir / "train.log")
            summary = json.loads((arm_dir / "summary.json").read_text(encoding="utf-8"))
            shutil.copy2(summary["model"], arm_dir / "model.pth")
        report(output)
        listening = output / "listening"
        listening.mkdir()
        test = next(r for r in manifest["records"] if r["split"] == "test")
        # Fixed first 20 seconds, declared before either model is heard.
        source, _ = load_audio(data / test["path"], sample_rate=40000)
        sf.write(listening / "input.wav", source[:800000], 40000, subtype="PCM_16")
        for arm in ("scratch", "finetune"):
            started = time.monotonic()
            wav = listening / f"{arm}.wav"
            run([ROOT / "voicelab/rvc_infer_entry.py", "--model", output / arm / "model.pth",
                 "--input", listening / "input.wav", "--output", wav, "--speaker-id", "0", "--pitch", "0",
                 "--index-rate", "0", "--rms-mix-rate", "1", "--protect", "0.33"], listening / f"{arm}.log")
            audio, _ = load_audio(wav, sample_rate=40000)
            append_json(output / "inference.jsonl", {"arm": arm, "sha256": sha256(wav),
                        "cold_process_seconds": time.monotonic() - started, "stats": audio_stats(audio, 40000),
                        "note": "Time includes model loading, not streaming latency"})
        from .inference import evaluate
        for arm in ("scratch", "finetune"):
            evaluate(listening / "input.wav", listening / f"{arm}.wav", listening / f"{arm}_metrics.json")
        blind = output / "blind"
        blind.mkdir()
        rng = np.random.default_rng(90517)
        order = list(rng.permutation(["scratch", "finetune"]))
        key = dict(zip(("X", "Y"), order))
        for label, arm in key.items():
            y, _ = load_audio(listening / f"{arm}.wav", sample_rate=40000)
            rms = np.sqrt(np.mean(y.astype(np.float64) ** 2))
            gain = min(0.08 / max(rms, 1e-8), 0.95 / max(float(np.max(np.abs(y))), 1e-8))
            sf.write(blind / f"{label}.wav", y * gain, 40000, subtype="PCM_16")
        write_json(output / "blind_key.json", {"labels": key, "note": "Keep hidden until scoring; blind versions only are RMS matched and peak limited."})
        with (output / "listening_scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["listener", "label", "naturalness_1_5", "intelligibility_1_5", "identity_1_5", "pitch_stability_1_5", "notes"])
            writer.writerows([["", label, "", "", "", "", ""] for label in ("X", "Y")])
        report(output)
    except BaseException as error:
        write_json(output / "pipeline_failure.json", {"utc": utc_now(), "error": repr(error)})
        report(output)
        raise
    finally:
        export_results(output)
    print(f"Download {output / 'results.zip'}", flush=True)


if __name__ == "__main__":
    main()
