"""Local recording workbench. Run: python app.py (127.0.0.1:8872)."""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
ASSET_ROOT = (HERE / "assets").resolve()
REFERENCE_CATALOG = ASSET_ROOT / "reference_catalog.json"
SESSION_ROOT = Path(os.environ.get("VOICE_WORKBENCH_SESSIONS", str(HERE / "local_sessions"))).resolve()
SESSION_ROOT.mkdir(parents=True, exist_ok=True)
MAX_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 30.0
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".mp4"}
MODEL_SAMPLE_RATES = {"speech": 22050, "singing": 44100}
CONVERSION_LOCK = asyncio.Lock()
EXAMPLE_FILES = {
    "self_singing_input_8s": "本人清唱原声 · 8 秒",
    "model_self_reconstruction": "模型重建本人 · 强制模型推理",
    "experimental_alpha_025": "实验插值 α=0.25",
    "experimental_alpha_050": "实验插值 α=0.50",
    "experimental_alpha_075": "实验插值 α=0.75",
    "experimental_alpha_100": "完整参考转换 α=1.00 · Linda Johnson",
}

app = FastAPI(title="Voice Workbench", version="0.1.0")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_receipt(directory: Path, receipt: dict) -> None:
    temporary = directory / "receipt.json.tmp"
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(directory / "receipt.json")


def load_engine():
    module_name = f"{__package__}.engine" if __package__ else "engine"
    return importlib.import_module(module_name)


def engine_status() -> dict:
    try:
        engine = load_engine()
        get_status = getattr(engine, "get_status", None)
        if callable(get_status):
            result = get_status()
            if not isinstance(result, dict):
                raise TypeError("engine.get_status() must return a dictionary")
            return result
        return {"ready": False, "state": "unknown", "message": "引擎已接入，模型就绪状态尚未提供。"}
    except Exception as exc:
        return {"ready": False, "state": "unavailable", "message": f"推理引擎尚未就绪：{type(exc).__name__}: {exc}"}


def default_reference(status: dict) -> tuple[Path | None, dict | None]:
    reference = status.get("default_reference")
    if isinstance(reference, str):
        reference = {"path": reference, "name": "开放参考女声"}
    if not isinstance(reference, dict) or not reference.get("path"):
        return None, None
    path = Path(reference["path"]).expanduser()
    if not path.is_absolute():
        path = HERE / path
    path = path.resolve()
    if not path.is_file():
        return None, None
    public = {key: value for key, value in reference.items() if key != "path"}
    public.setdefault("name", "开放参考女声")
    public["audio_url"] = "/api/default-reference.wav"
    return path, public


def audio_probe(path: Path) -> dict:
    executable = shutil.which("ffprobe")
    if not executable:
        raise HTTPException(503, detail={"code": "ffmpeg_missing", "message": "未找到 ffprobe；请先安装 FFmpeg 并加入 PATH。"})
    try:
        result = subprocess.run(
            [executable, "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=sample_rate,channels,duration:format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=25, check=True,
        )
        metadata = json.loads(result.stdout)
        streams = metadata.get("streams", [])
        if not streams:
            raise ValueError("文件中没有音频轨道")
        stream = streams[0]
        value = stream.get("duration") or metadata.get("format", {}).get("duration")
        duration = float(value) if value not in (None, "N/A") else None
        if duration is not None and (not math.isfinite(duration) or duration <= 0):
            raise ValueError("无效的音频时长")
        return {"duration_s": duration, "sample_rate": int(stream["sample_rate"]), "channels": int(stream["channels"])}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, detail={"code": "invalid_audio", "message": f"无法读取音频：{type(exc).__name__}。请使用正常的 WAV/MP3/M4A/WebM 文件。"}) from exc


def reference_catalog() -> dict[str, tuple[Path, dict]]:
    """Read a deliberate local catalog, never discover arbitrary recordings."""
    if not REFERENCE_CATALOG.is_file():
        return {}
    try:
        payload = json.loads(REFERENCE_CATALOG.read_text(encoding="utf-8-sig"))
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise ValueError("items 必须是数组")
        accepted, seen = {}, set()
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("参考条目必须是对象")
            reference_id = item.get("id")
            if not isinstance(reference_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", reference_id):
                raise ValueError("参考 id 必须是 1–64 个小写字母、数字、下划线或短横线")
            if reference_id in seen:
                raise ValueError("参考 id 重复，必须严格唯一")
            seen.add(reference_id)
            for field in ("name", "path", "source_url", "license"):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    raise ValueError(f"参考缺少 {field}")
            parsed_url = urlparse(item["source_url"])
            if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
                raise ValueError("参考 source_url 必须是有效的 HTTP(S) 来源")
            if item.get("kind") not in ("speech", "singing"):
                raise ValueError("参考 kind 必须是 speech 或 singing")
            if not isinstance(item.get("quality_approved", False), bool):
                raise ValueError("quality_approved 必须是布尔值")
            relative = Path(item["path"])
            if relative.is_absolute():
                raise ValueError("参考 path 必须相对 assets")
            path = (ASSET_ROOT / relative).resolve()
            try:
                path.relative_to(ASSET_ROOT)
            except ValueError as exc:
                raise ValueError("参考 path 超出 assets") from exc
            if path.suffix.lower() != ".wav":
                raise ValueError("内置参考必须为 WAV 文件")
            if not path.is_file():
                continue
            public = {key: item[key] for key in ("id", "name", "kind", "source_url", "license")}
            public["quality_approved"] = item.get("quality_approved", False)
            public["audio_url"] = f"/api/references/{reference_id}.wav"
            public["notice"] = "公开访谈/清唱版权参考，供本机研究使用；非开放授权音库，尚未验收。"
            accepted[reference_id] = (path, public)
        return accepted
    except (ValueError, OSError, TypeError) as exc:
        raise HTTPException(503, detail={"code": "reference_catalog_invalid",
                                        "message": f"内置参考配置有误：{exc}"}) from exc


def normalize_audio(source: Path, destination: Path, sample_rate: int) -> dict:
    metadata = audio_probe(source)
    if metadata["duration_s"] is not None and metadata["duration_s"] > MAX_SECONDS + 0.15:
        raise HTTPException(422, detail={"code": "audio_too_long", "message": "每段音频最多 30 秒；请先裁剪后上传。"})
    executable = shutil.which("ffmpeg")
    if not executable:
        raise HTTPException(503, detail={"code": "ffmpeg_missing", "message": "未找到 ffmpeg；请先安装 FFmpeg 并加入 PATH。"})
    try:
        subprocess.run(
            [executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
             "-map", "0:a:0", "-vn", "-t", str(MAX_SECONDS + 0.25), "-ac", "1", "-ar", str(sample_rate),
             "-c:a", "pcm_s16le", str(destination)],
            capture_output=True, timeout=45, check=True,
        )
        normalized = audio_probe(destination)
        if normalized["duration_s"] is None or normalized["duration_s"] > MAX_SECONDS + 0.15:
            raise HTTPException(422, detail={"code": "audio_too_long", "message": "音频超过 30 秒，或无法可靠确定时长；请裁剪后上传。"})
        if normalized["duration_s"] < 0.1:
            raise HTTPException(422, detail={"code": "audio_too_short", "message": "音频不足 0.1 秒。"})
        return {"original": metadata, "normalized": normalized, "sha256": sha256(destination)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, detail={"code": "normalization_failed", "message": "FFmpeg 转码失败；请尝试导出为 WAV 后重新上传。"}) from exc


async def save_upload(upload: UploadFile, directory: Path, label: str) -> Path:
    suffix = Path(upload.filename or "recording.webm").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, detail={"code": "unsupported_file", "message": "支持 WAV、MP3、M4A、WebM、OGG、FLAC、MP4 音频文件。"})
    destination = directory / f"{label}_uploaded{suffix}"
    size = 0
    try:
        with destination.open("wb") as stream:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, detail={"code": "file_too_large", "message": "每个音频文件最多 20 MB。"})
                stream.write(chunk)
        if not size:
            raise HTTPException(422, detail={"code": "empty_file", "message": "上传的音频文件为空。"})
        return destination
    finally:
        await upload.close()


def session_directory(session_id: str) -> Path:
    try:
        if str(uuid.UUID(session_id)) != session_id:
            raise ValueError("not canonical")
    except ValueError as exc:
        raise HTTPException(404, "未找到该录音") from exc
    return SESSION_ROOT / session_id


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((HERE / "static" / "index.html").read_text(encoding="utf-8"))


@app.get("/api/status")
async def status():
    engine = await asyncio.to_thread(engine_status)
    _, reference = default_reference(engine)
    public_engine = {key: value for key, value in engine.items() if key != "default_reference"}
    return {
        "service": "Voice Workbench", "busy": CONVERSION_LOCK.locked(),
        "ffmpeg_ready": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
        "engine": public_engine, "max_seconds": MAX_SECONDS, "max_file_mb": 20,
        "sample_rates": engine.get("sample_rates", MODEL_SAMPLE_RATES), "default_reference": reference,
        "singing_verified": bool(engine.get("singing_verified", False)),
        "notice": "音色插值是实验参数，不是感知上的真人身份百分比。0% 原声旁路不运行 AI。",
    }


@app.get("/api/default-reference.wav")
async def reference_audio():
    path, _ = default_reference(await asyncio.to_thread(engine_status))
    if path is None:
        raise HTTPException(404, "当前没有可试听的默认参考")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.get("/api/examples")
async def examples():
    # Only these six deliberately published examples are listed; never enumerate sessions.
    return {"mode": "singing", "singing_verified": False,
            "description": "本人→Linda Johnson，非 Adele；OpenVoice 轻唱研究试转，尚未验证转换质量。",
            "items": [{"name": name, "label": label, "audio_url": f"/examples/{name}.wav",
                       "available": (HERE / "benchmarks" / f"{name}.wav").is_file()}
                      for name, label in EXAMPLE_FILES.items()]}


@app.get("/api/references")
async def references():
    catalog = await asyncio.to_thread(reference_catalog)
    items = []
    for path, public in catalog.values():
        metadata = await asyncio.to_thread(audio_probe, path)
        items.append({**public, "duration_s": metadata["duration_s"]})
    return {"items": items, "local_only": True,
            "notice": "这些来源明确的版权参考只供本机试听/研究，非开放授权音库。"}


@app.get("/api/references/{reference_id}.wav")
async def builtin_reference_audio(reference_id: str):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", reference_id):
        raise HTTPException(404, "未找到内置参考")
    entry = (await asyncio.to_thread(reference_catalog)).get(reference_id)
    if entry is None:
        raise HTTPException(404, "未找到已准备的内置参考")
    path, _ = entry
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.get("/examples/{name}.wav")
async def example_audio(name: str):
    if name not in EXAMPLE_FILES:
        raise HTTPException(404, "未找到固定示例")
    path = HERE / "benchmarks" / f"{name}.wav"
    if not path.is_file():
        raise HTTPException(404, "该示例尚未生成")
    return FileResponse(path, media_type="audio/wav", filename=f"{name}.wav")


@app.get("/outputs/{session_id}.wav")
async def output_audio(session_id: str):
    path = session_directory(session_id) / "output.wav"
    if not path.is_file():
        raise HTTPException(404, "尚未生成结果")
    return FileResponse(path, media_type="audio/wav", filename=f"voice-{session_id}.wav")


@app.get("/inputs/{session_id}.wav")
async def input_audio(session_id: str):
    path = session_directory(session_id) / "source.wav"
    if not path.is_file():
        raise HTTPException(404, "未找到输入")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/receipts/{session_id}")
async def receipt_json(session_id: str):
    path = session_directory(session_id) / "receipt.json"
    if not path.is_file():
        raise HTTPException(404, "未找到记录")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.post("/api/convert")
async def convert(
    source_file: UploadFile = File(...),
    target_file: UploadFile | None = File(None),
    self_file: UploadFile | None = File(None),
    alpha: float = Form(1.0),
    mode: Literal["speech", "singing"] = Form("speech"),
    method: Literal["lerp", "slerp"] = Form("lerp"),
    diagnostic_self: bool = Form(False),
    use_default_reference: bool = Form(False),
    reference_id: str = Form(""),
):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise HTTPException(422, detail={"code": "invalid_alpha", "message": "alpha 必须在 0–1 之间。"})
    if CONVERSION_LOCK.locked():
        raise HTTPException(409, detail={"code": "busy", "message": "已有一次转换在执行，请等它完成。"})
    session_id = str(uuid.uuid4())
    directory = SESSION_ROOT / session_id
    directory.mkdir()
    receipt = {"id": session_id, "started_at": now(), "state": "processing", "alpha": alpha,
               "mode": mode, "method": method, "diagnostic_self": diagnostic_self,
               "waveform_dry_wet": False, "source_name": source_file.filename}
    save_receipt(directory, receipt)
    async with CONVERSION_LOCK:
        try:
            state = await asyncio.to_thread(engine_status)
            rates = state.get("sample_rates", MODEL_SAMPLE_RATES)
            sample_rate = int(rates.get(mode, MODEL_SAMPLE_RATES[mode]))
            if not 8000 <= sample_rate <= 192000:
                raise ValueError("invalid model sample rate")
            source_uploaded = await save_upload(source_file, directory, "source")
            source = directory / "source.wav"
            receipt["source"] = await asyncio.to_thread(normalize_audio, source_uploaded, source, sample_rate)
            target, self_ref = None, None
            # Product alpha=0 is explicitly a bypass, even if the engine is unavailable.
            if alpha == 0 and not diagnostic_self:
                output = directory / "output.wav"
                shutil.copyfile(source, output)
                result = {"ai_inference": False, "bypass": True, "message": "0%：原录音经过格式规范化直接输出；没有运行 AI 推理。"}
            else:
                if target_file is not None and not diagnostic_self:
                    target_uploaded = await save_upload(target_file, directory, "target")
                    target = directory / "target.wav"
                    receipt["target"] = await asyncio.to_thread(normalize_audio, target_uploaded, target, sample_rate)
                    receipt["target_name"] = target_file.filename
                    receipt["target_selection"] = "uploaded_target"
                elif reference_id and not diagnostic_self:
                    entry = (await asyncio.to_thread(reference_catalog)).get(reference_id)
                    if entry is None:
                        raise HTTPException(422, detail={"code": "unknown_reference",
                                                        "message": "内置参考尚未准备好或 id 不在白名单中；请重新选择。"})
                    reference_path, reference_info = entry
                    target = directory / "target.wav"
                    receipt["target"] = await asyncio.to_thread(normalize_audio, reference_path, target, sample_rate)
                    receipt["target_selection"] = "local_reference_catalog"
                    receipt["reference_id"] = reference_id
                    receipt["target_reference"] = reference_info
                    receipt["target_original_sha256"] = sha256(reference_path)
                elif use_default_reference and not diagnostic_self:
                    default_path, default_info = default_reference(state)
                    if default_path is None:
                        raise HTTPException(422, detail={"code": "no_default_reference", "message": "默认参考尚未安装；请上传真实目标参考。"})
                    target = directory / "target.wav"
                    receipt["target"] = await asyncio.to_thread(normalize_audio, default_path, target, sample_rate)
                    receipt["target_reference"] = default_info
                    receipt["target_selection"] = "open_default_reference"
                if self_file is not None:
                    self_uploaded = await save_upload(self_file, directory, "self")
                    self_ref = directory / "self.wav"
                    receipt["self"] = await asyncio.to_thread(normalize_audio, self_uploaded, self_ref, sample_rate)
                if target is None and not diagnostic_self:
                    raise HTTPException(422, detail={"code": "target_required", "message": "请上传目标参考，选择已准备的内置候选，或选用开放参考女声。"})
                if state.get("ready") is not True:
                    raise HTTPException(503, detail={"code": "engine_not_ready",
                                                    "message": state.get("message") or "模型尚未准备好；请先安装引擎依赖并准备模型权重。",
                                                    "missing_dependencies": state.get("missing_dependencies", state.get("missing_deps", []))})
                try:
                    engine = await asyncio.to_thread(load_engine)
                    function = getattr(engine, "convert_recording")
                except Exception as exc:
                    raise HTTPException(503, detail={"code": "engine_unavailable", "message": f"推理引擎尚未就绪：{type(exc).__name__}。请按项目安装说明准备模型。"}) from exc
                output = directory / "output.wav"
                result = await asyncio.to_thread(function, source=source, target=target, self_ref=self_ref,
                                               output=output, alpha=alpha, mode=mode, method=method,
                                               diagnostic_self=diagnostic_self)
                if not isinstance(result, dict):
                    raise TypeError("engine returned a non-dictionary result")
                result.setdefault("ai_inference", True)
                result.setdefault("bypass", False)
            if not output.is_file():
                raise RuntimeError("推理引擎没有写入 output.wav")
            receipt.update({"state": "complete", "finished_at": now(), "result": result,
                            "output": {**await asyncio.to_thread(audio_probe, output), "sha256": sha256(output)}})
            save_receipt(directory, receipt)
            return {"id": session_id, "audio_url": f"/outputs/{session_id}.wav",
                    "input_url": f"/inputs/{session_id}.wav", "receipt_url": f"/api/receipts/{session_id}",
                    "result": result, "state": "complete", "notice": "只生成一条声音；没有将原声与换声波形叠加。"}
        except HTTPException as exc:
            receipt.update({"state": "failed", "finished_at": now(), "error": exc.detail})
            save_receipt(directory, receipt)
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail, "id": session_id,
                                                                    "receipt_url": f"/api/receipts/{session_id}"})
        except Exception as exc:
            # Preserve an actionable receipt without pretending inference succeeded.
            message = f"{type(exc).__name__}: {exc}"
            code = "engine_failed"
            status_code = 503 if isinstance(exc, (ImportError, ModuleNotFoundError, FileNotFoundError)) else 500
            receipt.update({"state": "failed", "finished_at": now(), "error": {"code": code, "message": message}})
            save_receipt(directory, receipt)
            return JSONResponse(status_code=status_code, content={"detail": {"code": code, "message": message},
                                                                 "id": session_id, "receipt_url": f"/api/receipts/{session_id}"})
        finally:
            # Also close unconsumed optional uploads on the bypass path and on errors.
            for upload in (source_file, target_file, self_file):
                if upload is not None:
                    await upload.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8872)
