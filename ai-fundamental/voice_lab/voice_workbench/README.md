# 声音工作台：录音 → 模型转换 → 试听

这是可运行的 **OpenVoice V2 预训练模型 CPU 原型**。支持录音/上传、参考音色、模型内部 LERP/SLERP 插值、本人重建诊断和运行收据。没有叠加两条输出波形。

**当前目标参考是 Linda Johnson 的公共领域 LJ Speech，不是 Adele。** Adele 混合还需要实际目标参考。OpenVoice 以讲话为主；用已有清唱试转是未验收的研究实验。这里只做停止录音后的转换，没有实现连续流式变声，也没有新增训练。

## 在这台电脑立即体验

本机环境和模型已准备好。在 PowerShell 中运行：

```powershell
cd 'I:\sideprojects\ai fundamental\voice_lab\voice_workbench'
.\.venv\Scripts\python.exe -X utf8 app.py
```

看到 `Uvicorn running on http://127.0.0.1:8872` 后，打开 **http://127.0.0.1:8872/**。保持窗口打开；Ctrl+C 停止服务。也可执行本目录 `start_workbench.ps1`。

1. 先试听“已生成的小样”，比较原声、模型本人重建及四档插值。一次只播放一个播放器。
2. 点击“开始录音”，允许浏览器使用麦克风，说 5–15 秒；停止后先试听。也可上传 WAV、MP3、M4A、WebM 等。
3. 勾选“使用开放参考 Linda Johnson”，或上传自己的真实目标参考。建议干净单人声，少伴奏、少混响；每个文件最多 30 秒/20 MB。
4. 将 α 设为 100%，先检查完整目标转换。点击“生成并试听”，等真实输出出现。
5. 再试 25/50/75%。α 是音色向量的实验权重，不能当成人耳感知的身份百分比。
6. 勾选“本人重建诊断”，检查模型本身是否引入噪声/双声部。该模式忽略目标与 α，始终运行模型。
7. 点“查看本次运行收据”，能看到版本、输入/输出 SHA256、seed、耗时。录音和参考只保存在本机 `local_sessions/`，不会自动上传 GitHub。

**0% 普通模式是格式规范化后的原声旁路，没有 AI 推理。** 0% 本人重建诊断仍走模型，二者不能混为同一个质量证明。

## 在另一台电脑安装

以下为匹配本机验证版本的 CPU 路线，需 Python 3.10 和 FFmpeg/FFprobe（加入 PATH）。不修改旧 RVC 环境。

```powershell
cd '<你的项目路径>\voice_lab\voice_workbench'
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt -r requirements-ui.txt
.\.venv\Scripts\python.exe prepare_openvoice.py
.\.venv\Scripts\python.exe prepare_reference.py
.\.venv\Scripts\python.exe -X utf8 app.py
```

模型准备脚本只下载固定官方 OpenVoice converter（约 131 MB）与必要源文件，核验 SHA256；不下载整套 TTS/ASR/数据集。Python/Torch 依赖另占空间。若访问官方下载源失败，保留报错，不换用未知模型。

本机已有五档样本；新的 Git checkout 不含私人样例、模型缓存或录音。用麦克风/上传产生你自己的输入。

## 怎样理解这条链

```text
源录音 → 编码原话/语调 → 以源音色做正向 flow
本人参考 → e_self ┐
目标参考 → e_target ┴→ e_mix=(1−α)e_self+αe_target
源内容 latent + e_mix → 逆向 flow/声码器 → 一条生成的声音
```

- **VC — Voice Conversion：声音转换**，保留录音内容，改变音色。
- **Speaker embedding：音色身份向量**，模型提取的数值表示，不是一条可播放音轨。
- **LERP — Linear Interpolation：线性插值**。SLERP 是球面线性插值，不能预先假定更自然。
- **RTF — Real-Time Factor：耗时/音频时长**。RTF<1 也不能证明麦克风到耳机已实时。

模型仍可能产生伪影或不自然的中间音色。先过本人重建和目标转换，再评价插值；不要用低训练 loss 或音高指标代替试听。

## 已有实测与下一步

本机同一 8 秒清唱输入，OpenVoice 本人重建及四档 LERP 都实际生成约 7.999 秒 WAV，波形有限且无数字削波。模型预热后总转换耗时约 5.76–8.07 秒，首次该批转换约 18.37 秒；更早首次库预热的 2 秒说话 smoke 总计约 34.35 秒。耗时不是严格多次重复的速度 benchmark。

同一 8 秒、统一 22050 Hz、原声安静帧代理比较：旧 A/RVC 为 −32.18 dBFS，旧 B 为 −40.48，新模型本人重建为 −60.06。对照还保留“安静能量/整体能量”比，避免只因整体音量较低就称降噪成功。这只说明此代理上的停顿残留更低；**未证明自然度、Adele 相似度或歌声更好。** 详见本机 `benchmarks/BENCHMARK_RECEIPT.json`、`SAME_CLIP_COMPARISON.json` 和 `UI_API_VERIFICATION.json`。

本机服务已经实际验证上传→FFmpeg→真实模型→WAV 下载及收据。麦克风硬件录制仍需要你亲自试一次。新讲话输入和 Adele 参考尚待提供；SoulX-Singer-SVC/Seed-VC 的歌声对照未在本轮运行。

## 文件与接口

| 文件/接口 | 用途 |
|---|---|
| `app.py` | 本地 FastAPI 服务，限时/格式检查，私有会话记录 |
| `static/index.html` | 麦克风、上传、滑杆、独占播放器 |
| `engine.py` | 固定官方 OpenVoice 模型与真实 latent 插值 |
| `prepare_openvoice.py` | 固定版本与模型校验下载 |
| `prepare_reference.py` | 下载一条有来源的公共领域讲话参考 |
| `run_benchmark.py` | 本机固定轻唱小样与五档收据；需准备相应输入 |
| `GET /api/status` | 模型是否就绪、设备、参考、FFmpeg 状态 |
| `POST /api/convert` | multipart 源音频、参考、alpha、mode、method、diagnostic_self |
| `GET /outputs/{id}.wav` | 实际输出，不能把未完成请求当成功 |
| `GET /api/receipts/{id}` | 本次参数/输入/结果/失败证据 |

GitHub Pages 只能播放现有 WAV。要从别处录音并转换，需要正常访问推理后端；本版绑定 localhost，不暴露公网服务。云端路线待确认新运行预算后，用 SSH 转发或另行部署有访问控制的 HTTPS 后端。

## 上游与改动

[OpenVoice 官方](https://github.com/myshell-ai/OpenVoice)代码/权重 MIT；代码固定 `74a1d147b17a8c3092dd5430504bd83ef6c7eb23`，模型固定 `f36e7edfe1684461a8343844af60babc2efbb727`。适配器直接使用官方 encoder/flow/decoder；用干净参考直接 `extract_se`；薄子类采用官方基类初始化，关闭另外的水印模型并明确标注 AI 转换。未改上游源文件。完整原因与哈希留在每次 metadata。

[LJ Speech 官方资料](https://keithito.com/LJ-Speech-Dataset/)；[SoulX-Singer-SVC 歌声候选](https://github.com/Soul-AILab/SoulX-Singer)；[Seed-VC 对照候选](https://github.com/Plachtaa/seed-vc)。源码许可证不自动等同于目标录音的许可。
