"""Build a portable listening page from six explicitly allowed local recordings.

No inference, media download, credentials, reference audio or session directories
are involved. Output paths inside receipts are deliberately never followed.
Building uses the existing FFmpeg executable to reproduce app normalization;
the generated HTML/audio page has no runtime dependencies or external assets.
Run: python build_adele_demo.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
BENCHMARKS = ROOT / "benchmarks"
EXPLORATION = BENCHMARKS / "adele_reference_exploration"
VOGUE_URL = "https://www.vogue.com/video/watch/73-questions-with-adele"
NPR_URL = "https://www.nprillinois.org/2015-11-24/you-cant-prepare-yourself-a-conversation-with-adele"

# A fixed filename allowlist prevents accidental publication of interviews or
# private recordings referenced by receipt paths.
SAMPLES = (
    ("source", "本人原声", "输入：8 秒轻唱", 0.0, None),
    ("vogue_singing_alpha050", "清唱参考 · α 0.50", "Vogue 现场清唱参考", 0.5, VOGUE_URL),
    ("vogue_singing_alpha100", "清唱参考 · α 1.00", "Vogue 现场清唱参考", 1.0, VOGUE_URL),
    ("vogue_speech_alpha100", "讲话参考 · α 1.00", "Vogue 连续讲话参考", 1.0, VOGUE_URL),
    ("npr_speech_alpha050", "讲话参考 · α 0.50", "NPR 连续访谈参考", 0.5, NPR_URL),
    ("npr_speech_alpha100", "讲话参考 · α 1.00", "NPR 连续访谈参考", 1.0, NPR_URL),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wav_info(path: Path) -> dict:
    """Read PCM/FLOAT RIFF WAV duration using only the standard library."""
    size = path.stat().st_size
    fmt = None
    data_bytes = None
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError("不是受支持的 RIFF WAV")
        while stream.tell() + 8 <= size:
            chunk = stream.read(8)
            chunk_id, chunk_size = struct.unpack("<4sI", chunk)
            if stream.tell() + chunk_size > size:
                raise ValueError("WAV 数据不完整")
            if chunk_id == b"fmt ":
                if chunk_size < 16:
                    raise ValueError("WAV fmt 数据不完整")
                fmt = struct.unpack("<HHIIHH", stream.read(16))
                stream.seek(chunk_size - 16, 1)
            elif chunk_id == b"data":
                data_bytes = chunk_size
                stream.seek(chunk_size, 1)
            else:
                stream.seek(chunk_size, 1)
            if chunk_size & 1:
                stream.seek(1, 1)
    if fmt is None or data_bytes is None:
        raise ValueError("WAV 缺少 fmt/data")
    encoding, channels, rate, byte_rate, block_align, bits = fmt
    if encoding not in (1, 3, 65534) or min(channels, rate, byte_rate, block_align) <= 0:
        raise ValueError("WAV 格式无效")
    return {
        "duration_s": data_bytes / byte_rate,
        "sample_rate": rate,
        "channels": channels,
        "encoding": encoding,
        "bits_per_sample": bits,
    }


def finite_number(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def stamp(value) -> str | None:
    """Publish only a validated ISO date, not arbitrary receipt strings."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


def digest_value(value) -> str | None:
    if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value):
        return value
    return None


def normalized_source_hash(source: Path, out: Path) -> str | None:
    """Recreate the app's PCM16 22050Hz encoding for receipt source checks.

    This creates one temporary file, performs no inference, and copies no
    reference audio. The fixed original is still the playback source.
    """
    executable = shutil.which("ffmpeg")
    if not executable or not source.is_file():
        return None
    descriptor, filename = tempfile.mkstemp(prefix=".source-check-", suffix=".wav", dir=out)
    os.close(descriptor)
    temporary = Path(filename)
    try:
        subprocess.run(
            [executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
             "-map", "0:a:0", "-vn", "-t", "30.25", "-ac", "1", "-ar", "22050",
             "-c:a", "pcm_s16le", str(temporary)],
            capture_output=True, timeout=45, check=True,
        )
        return sha256(temporary)
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        temporary.unlink(missing_ok=True)


def collect_sample(spec: tuple, out: Path, source_hash: str | None, normalized_hash: str | None) -> dict:
    key, title, reference_label, intended_alpha, source_url = spec
    item = {
        "id": key,
        "title": title,
        "reference_label": reference_label,
        "source_url": source_url,
        "state": "missing",
        "message": "未生成",
        "audio": None,
    }
    source = BENCHMARKS / "self_singing_input_8s.wav" if key == "source" else EXPLORATION / (key + ".wav")
    destination = out / "listening" / (key + ".wav")
    receipt = None
    metadata = {}
    try:
        if not source.is_file():
            return item
        if key != "source":
            receipt_path = EXPLORATION / (key + ".receipt.json")
            if not receipt_path.is_file():
                return {**item, "message": "未生成：缺少完整推理收据"}
            receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
            if receipt.get("state") != "complete":
                return {**item, "message": "未生成：推理尚未完成"}
            result = receipt.get("result", {})
            metadata = result.get("metadata", {})
            alpha = finite_number(receipt.get("alpha"))
            if alpha is None or abs(alpha - intended_alpha) > 1e-8:
                raise ValueError("收据 alpha 与固定展示项不一致")
            if receipt.get("mode") != "singing":
                raise ValueError("收据输入类型不是当前轻唱实验")
            if metadata.get("waveform_mixing") is not False or result.get("ai_inference") is not True:
                raise ValueError("收据未确认真实模型推理及非波形叠加")
            if result.get("bypass") is not False:
                raise ValueError("输出为旁路而非模型输出")
            raw_source_hash = receipt.get("source", {}).get("sha256")
            if source_hash is None or normalized_hash is None or raw_source_hash != normalized_hash:
                raise ValueError("规范化输入哈希与当前固定片段不一致，或未能完成源校验")
        actual_hash = sha256(source)
        info = wav_info(source)
        if receipt is not None:
            declared_hash = receipt.get("output", {}).get("sha256")
            if declared_hash != actual_hash or metadata.get("output_sha256") != actual_hash:
                raise ValueError("输出 SHA256 与推理收据不一致")
            declared_duration = finite_number(receipt.get("output", {}).get("duration_s"))
            declared_sr = finite_number(receipt.get("output", {}).get("sample_rate"))
            if declared_duration is None or abs(info["duration_s"] - declared_duration) > 0.002:
                raise ValueError("输出时长与收据不一致")
            if declared_sr != info["sample_rate"]:
                raise ValueError("输出采样率与收据不一致")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256(destination) != actual_hash:
            raise ValueError("展示文件复制校验失败")
        item.update(
            state="complete", message="已生成" if receipt else "输入原声", audio=f"listening/{key}.wav",
            alpha=finite_number(receipt.get("alpha")) if receipt else 0.0,
            alpha_is_calibrated_identity_percentage=False,
            inference=receipt is not None, waveform_mixing=False,
            audio_sha256=actual_hash, duration_s=info["duration_s"],
            sample_rate=info["sample_rate"], channels=info["channels"],
        )
        if receipt:
            # Do not copy a receipt wholesale: it contains local/session paths
            # and private reference audio API URLs.
            item.update(
                finished_at=stamp(receipt.get("finished_at")),
                reference_duration_s=finite_number(receipt.get("target", {}).get("original", {}).get("duration_s")),
                engine="OpenVoice V2" if metadata.get("engine") == "OpenVoice V2" else "unknown",
                device="cpu" if metadata.get("device") == "cpu" else "unknown",
                seed=finite_number(metadata.get("seed")),
                method=receipt.get("method") if receipt.get("method") in ("lerp", "slerp") else "unknown",
                total_seconds=finite_number(metadata.get("total_seconds")),
                model_seconds=finite_number(metadata.get("model_seconds")),
                model_load_seconds=finite_number(metadata.get("model_load_seconds")),
                checkpoint_sha256=digest_value(metadata.get("checkpoint_sha256")),
                adapter_sha256=digest_value(metadata.get("adapter_sha256")),
                singing_validated=False,
                source_original_sha256=source_hash,
                source_normalized_sha256=normalized_hash,
                source_identity_check="same fixed original reproduced with the app FFmpeg normalization; SHA256 matched",
            )
        return item
    except (ValueError, OSError, TypeError, KeyError, struct.error) as error:
        # This message comes from our own checks; no raw receipt data is echoed.
        reason = str(error) if isinstance(error, ValueError) and not isinstance(error, json.JSONDecodeError) else type(error).__name__
        return {**item, "state": "invalid", "message": "未生成：校验未通过（" + reason + "）"}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def seconds(value: float | None) -> str:
    return "未记录" if value is None else f"{value:.2f} 秒"


def make_card(item: dict, index: int) -> str:
    ready = item["state"] == "complete"
    alpha = item.get("alpha")
    alpha_label = "原声" if item["id"] == "source" else (f"α = {alpha:.2f}" if alpha is not None else "待生成")
    duration = seconds(item.get("duration_s"))
    reference = f'<a href="{esc(item["source_url"])}" target="_blank" rel="noopener noreferrer">原发布者出处 ↗</a>' if item["source_url"] else '<span>本人录音输入</span>'
    player = f'<audio controls preload="metadata" aria-label="{esc(item["title"])}"><source src="{esc(item["audio"])}" type="audio/wav">浏览器不支持播放，请下载 WAV。</audio><a class="download" href="{esc(item["audio"])}" download>下载 WAV ↓</a>' if ready else f'<div class="empty" role="status">{esc(item["message"])}</div>'
    stats = f'<span>{duration}</span><span>{item.get("sample_rate", 0) / 1000:g} kHz</span>' if ready else '<span>等待完整音频与收据</span>'
    details = "未经过模型转换；与其他条目使用同一段轻唱。"
    if item["id"] != "source":
        details = (f'实际 α 来自推理收据；{esc(item.get("method", "待记录"))} 条件插值。'
                   f'参考片段 {seconds(item.get("reference_duration_s"))}。') if ready else "保留此固定位置；可用结果生成后再更新展示。"
    timing = ""
    if ready and item["id"] != "source":
        timing = f'<details><summary>查看本次运行记录</summary><dl><dt>设备</dt><dd>{esc(item.get("device"))}</dd><dt>本次总耗时</dt><dd>{seconds(item.get("total_seconds"))}</dd><dt>其中模型推理</dt><dd>{seconds(item.get("model_seconds"))}</dd><dt>本次模型加载</dt><dd>{seconds(item.get("model_load_seconds"))}</dd></dl><p>缓存、预热与输入长度影响耗时；这些记录不是实时性能基准。</p><p class="hash">输出 SHA256<br>{esc(item["audio_sha256"])}</p></details>'
    return f'<article class="card"><div class="card-top"><span class="number">{index:02d}</span><span class="pill">{esc(alpha_label)}</span></div><h2>{esc(item["title"])}</h2><p class="reference">{esc(item["reference_label"])}</p><div class="stats">{stats}</div>{player}<p class="detail">{details}</p><div class="source-link">{reference}</div>{timing}</article>'


def render(summary: dict) -> str:
    cards = "\n".join(make_card(item, i + 1) for i, item in enumerate(summary["samples"]))
    ready_count = sum(item["state"] == "complete" for item in summary["samples"])
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="本人轻唱的 AI 音色探索：Adele 来源参考条件与不同 alpha 的实际模型输出；尚未验收，非真人演唱。"><title>AI 音色探索 · 参考与插值试听</title>
<style>
:root{color-scheme:light;--ink:#183536;--muted:#58716f;--paper:#faf9f4;--line:#d5e0d8;--accent:#146359;--cream:#f1ead7}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.7 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}a{color:var(--accent);text-underline-offset:4px}main{max-width:1120px;margin:auto;padding:42px 24px 28px}.eyebrow{font-size:13px;letter-spacing:.12em;color:var(--accent);font-weight:700}h1{font-size:clamp(30px,5vw,54px);line-height:1.2;letter-spacing:-.035em;margin:15px 0}header>p{max-width:760px;color:var(--muted);margin:0 0 20px}.notice{padding:16px 20px;border:1px solid #d7c79c;background:var(--cream);border-radius:14px;font-size:15px}.notice strong{display:block}.flow{display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:24px 0;color:var(--muted);font-size:14px}.flow b{background:#e7efe9;padding:6px 13px;border-radius:100px;color:var(--ink)}.section-title{display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin:6px 0 18px}.section-title h2{font-size:22px;margin:0}.section-title span{font-size:13px;color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.card{min-width:0;padding:22px;background:white;border:1px solid var(--line);border-radius:18px;box-shadow:0 3px 18px #18353605}.card-top{display:flex;justify-content:space-between;align-items:center}.number{color:#7b9188;font-size:13px;font-variant-numeric:tabular-nums}.pill{background:#edf3ef;color:var(--accent);border-radius:100px;padding:3px 11px;font-size:12px;font-weight:650}.card h2{font-size:19px;margin:18px 0 3px}.reference{color:var(--muted);margin:0;font-size:14px}.stats{display:flex;gap:12px;font-size:12px;color:var(--muted);margin:16px 0 10px}audio{display:block;width:100%;height:44px}.download{display:inline-block;font-size:13px;margin-top:8px}.detail{font-size:13px;color:var(--muted);margin:16px 0 12px}.source-link{font-size:13px}.empty{min-height:77px;display:flex;align-items:center;font-size:13px;color:var(--muted);background:#f3f5f3;border-radius:9px;padding:12px}details{border-top:1px solid var(--line);padding-top:12px;margin-top:17px;font-size:12px;color:var(--muted)}summary{cursor:pointer;color:var(--ink)}dl{display:grid;grid-template-columns:1fr auto;gap:4px 12px;margin:10px 0}dt,dd{margin:0}.hash{overflow-wrap:anywhere;font-family:ui-monospace,monospace;font-size:10px}.notes{margin-top:26px;padding:22px 25px;border:1px solid var(--line);border-radius:16px;background:#eef2ec}.notes h2{font-size:18px;margin:0 0 10px}.notes ul{margin:0;padding-left:21px}.notes li{margin:7px 0;font-size:14px}footer{margin-top:24px;font-size:12px;color:var(--muted);display:flex;gap:14px;flex-wrap:wrap}#play-status{min-height:23px;margin:10px 0 16px;color:var(--muted);font-size:13px}@media(max-width:880px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:580px){main{padding:28px 16px}.grid{grid-template-columns:1fr}.card{padding:20px}.section-title{display:block}.section-title span{display:block;margin-top:4px}.notice{padding:14px}.flow{font-size:12px;gap:7px;padding:19px 0}.flow b{padding:5px 9px}}
</style></head><body><main><header><div class="eyebrow">VOICE WORKBENCH / REFERENCE EXPLORATION</div><h1>同一段轻唱，<br>探索不同音色条件。</h1><p>听本人原声，再比较清唱参考、讲话参考与不同 α 的实际输出。所有输出来自同一段本人录音，使用 OpenVoice V2 进行音色条件转换。</p><div class="notice"><strong>AI 音色探索，使用 Adele 来源参考；尚未验收，不是真人演唱。</strong>这是一组实验结果。Adele 来源访谈与清唱未声明开放授权；本页仅提供原发布者链接，不提供其原始录音。</div></header>
<div class="flow" aria-label="处理流程"><b>本人 8 秒轻唱</b><span>→</span><b>参考音色条件 / α</b><span>→</span><b>模型转换输出</b></div>
<div class="section-title"><h2>逐条试听</h2><span>READY_COUNT / 6 条可播放 · 切换播放会暂停上一条</span></div><div id="play-status" aria-live="polite">建议戴耳机，先听原声，再用相近音量比较。</div><section class="grid" aria-label="音频对照">CARDS</section>
<section class="notes"><h2>这次试听能说明什么</h2><ul><li><strong>讲话参考也作用于同一段轻唱输入。</strong>参考类型不同，不代表这里完成了讲话输入的质量验证。</li><li><strong>α 是音色条件插值参数。</strong>0.50 不代表经校准的“50% 本人 + 50% Adele”。生成音频没有把两条波形叠加。</li><li><strong>没有进行新训练；不是实时直播转换。</strong>本页播放已经生成的 WAV。运行记录受缓存、预热等影响，不能作为速度基准。</li><li><strong>OpenVoice V2 的歌唱质量尚未验收。</strong>请分别注意歌词清晰度、音高与节奏保留、滋滋声、复音感及目标音色；不能据几条样本宣布整体质量胜者。</li><li>参照片段来自公开版权节目，不是开源音色库。出处：<a href="VOGUE_URL" target="_blank" rel="noopener noreferrer">Vogue</a>、<a href="NPR_URL" target="_blank" rel="noopener noreferrer">NPR / NPR Illinois</a>。</li></ul></section>
<footer><a href="summary.json">查看脱敏实测摘要 JSON</a><span>页面及音频可下载后离线播放</span><span>生成于 BUILD_TIME</span></footer></main>
<script>
const players = Array.from(document.querySelectorAll('audio'));
const status = document.querySelector('#play-status');
for (const active of players) {
  active.addEventListener('play', () => {
    for (const other of players) if (other !== active) other.pause();
    status.textContent = '正在播放：' + active.getAttribute('aria-label');
  });
  active.addEventListener('ended', () => { status.textContent = '播放结束，可继续比较另一条。'; });
  active.addEventListener('error', () => { status.textContent = '浏览器未能播放该条音频，可使用卡片中的“下载 WAV”。'; });
}
</script></body></html>""".replace("CARDS", cards).replace("READY_COUNT", str(ready_count)).replace("VOGUE_URL", esc(VOGUE_URL)).replace("NPR_URL", esc(NPR_URL)).replace("BUILD_TIME", esc(summary["built_at"]))


def build(out: Path) -> dict:
    out = out.resolve()
    # Copy only the six filenames above. Never traverse raw references, assets,
    # local_sessions, weights, logs or arbitrary paths from receipts.
    if out == ROOT or out == BENCHMARKS or out == EXPLORATION:
        raise ValueError("请选择独立的静态展示输出目录")
    out.mkdir(parents=True, exist_ok=True)
    source_path = BENCHMARKS / "self_singing_input_8s.wav"
    source_hash = sha256(source_path) if source_path.is_file() else None
    normalized_hash = normalized_source_hash(source_path, out)
    samples = [collect_sample(spec, out, source_hash, normalized_hash) for spec in SAMPLES]
    summary = {
        "schema_version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "title": "本人轻唱 / Adele 来源参考的 AI 音色探索",
        "disclaimer": "AI 音色探索，使用 Adele 来源参考；尚未验收，不是真人演唱。",
        "input_type": "singing", "new_training": False, "live_streaming": False,
        "singing_validated": False, "quality_winner": None,
        "timings_are_per_run_records_not_speed_benchmark": True,
        "reference_audio_included": False, "receipts_copied_wholesale": False,
        "waveform_mixing": False, "samples": samples,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "index.html").write_text(render(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "static_adele_demo")
    args = parser.parse_args()
    summary = build(args.out)
    print(json.dumps({
        "output": str(args.out.resolve()),
        "available": sum(item["state"] == "complete" for item in summary["samples"]),
        "total": len(summary["samples"]),
        "states": {item["id"]: item["state"] for item in summary["samples"]},
        "reference_audio_included": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
