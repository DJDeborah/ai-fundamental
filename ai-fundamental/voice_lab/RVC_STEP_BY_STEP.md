# 清唱模型：从文件到试听的深度拆解

本教程对应你这次的真实实验。训练输出最终保存到本机 `voice_lab/runs/new4090d_ab_retry1`；真实完成状态以该目录 `comparison.json` 为准。

## 1. 先明确输入和输出

这次模型接收一段已经唱好的人声，生成目标音色的人声。输入提供歌词、旋律、节奏。只用你的录音训练，因此第一次测试是“未见过的你 → 重建你的音色”。另一位授权歌者 → 你的音色，需要另外准备输入才能验证。

它没有文字聊天界面。界面是音频播放器、评分表和转换命令。权重文件 `.pth` 不能直接双击播放；`.wav` 才是可播放的声音。

## 2. 术语全称与具体作用

| 名称 | 全称 | 用熟悉的话解释 |
|---|---|---|
| RVC | Retrieval-based Voice Conversion | 基于检索的声音转换。本轮检索比例为 0，避免额外索引影响对照 |
| SVC | Singing Voice Conversion | 歌声音色转换，保留内容与旋律，更换音色 |
| HuBERT | Hidden-Unit BERT | 将声音压成内容特征；BERT 为 Bidirectional Encoder Representations from Transformers |
| F0 | Fundamental Frequency | 基频，主要对应唱出的音高 |
| RMVPE | Robust Model for Vocal Pitch Estimation | 提取人声音高的模型；与 HuBERT 一样，两组共用冻结预训练权重 |
| G / D | Generator / Discriminator | 生成器输出声音；判别器学习区分生成片段和真实片段 |
| GAN | Generative Adversarial Network | 生成对抗网络，G 和 D 通过各自目标共同训练 |
| scratch | 随机初始化训练 | 本轮仅转换 G/D 从随机数开始，内容与音高提取器仍来自预训练 |
| fine-tuning | 微调 | 从官方 G/D 已学到的权重开始，用你的录音继续更新 |
| tensor | 张量 | 多维数字数组。例如内容特征是“时间帧 × 768 个数字” |
| parameter / weight | 参数 / 权重 | 模型里可学习的数字，optimizer 每步修改它们 |
| optimizer | 优化器 | 根据梯度修改参数；本轮使用 AdamW |
| AdamW | Adam with decoupled Weight Decay | 自适应梯度优化，并把权重衰减与梯度更新分开处理 |
| batch | 批次 | 一次送入模型的片段集合，这里每批 4 个 |
| step | 更新步 | 一批完成一次 D 更新、一次 G 更新；两组目标各 1000 次 |
| epoch | 数据轮次 | 遍历训练加载器一轮；长度分桶与补齐会造成部分片段重复 |
| checkpoint | 模型快照 | 某一更新时刻保存的权重，例如 step500 |
| seed | 随机种子 | 控制随机数起点；这里 1234，不代表 GPU 逐位完全一致 |
| FP32 | 32-bit Floating Point | 32 位浮点数；本次两组相同精度，避免半精度跳步 |
| GPU | Graphics Processing Unit | 适合大量并行数字计算；本轮是 RTX 4090 D |
| CUDA | Compute Unified Device Architecture | NVIDIA GPU 计算平台，PyTorch 通过它在显卡上计算 |
| VRAM | Video Random Access Memory | 显存，存放模型、梯度、激活和临时数组；不同统计口径数值不同 |
| PCM | Pulse-Code Modulation | 脉冲编码调制；PCM16 WAV 存储 16 位量化的波形样本 |
| sample rate | 采样率 | 每秒记录多少波形数字；40 kHz = 40,000 个/秒 |
| frame / hop | 帧 / 帧移 | 分析声音时的滑动窗口及每次前进的样本数 |
| STFT | Short-Time Fourier Transform | 短时傅里叶变换，把每个时间窗口转为频率能量分布 |
| mel | 梅尔频率尺度 | 频率的一种感知尺度；mel 重建 loss 比较生成与真实的声学特征 |
| KL | Kullback–Leibler divergence | KL 散度，训练中约束潜在分布之间的差异 |
| F0 cent | 音分 | 100 cent 为平均律半音，1200 cent 为八度，用于比较音高误差 |
| RMS | Root Mean Square | 均方根，描述波形幅度；匿名试听做响度近似匹配 |
| SSH | Secure Shell | 安全远程终端，连接你的 AutoDL 云端实例 |
| CLI | Command-Line Interface | 命令行界面，例如终端中运行转换命令 |
| GUI | Graphical User Interface | 图形界面，例如带播放按钮的试听页面 |
| SHA256 | Secure Hash Algorithm 256-bit | 文件指纹，核对上传/下载后字节是否一致 |

## 3. 数据阶段：实际做了什么，得到什么

### 3.1 读取原录音

输入是你本机清唱目录里的 6 个 M4A。转成单声道、40 kHz、PCM16 WAV，原始文件不改动。

结果：`data/self_rvc_v1` 中有 6 个 WAV 和 `manifest.json`。清唱训练无需主动添加伴奏或噪声。含伴奏/噪声输入是否稳健是后续单独测试，不能混进这次干净条件对照后再归因于初始化方式。

### 3.2 按整段录音划分

| 分组 | 文件尾号 | 时长 | 谁可以使用 |
|---|---|---:|---|
| train | 1234、19、57、93 | 311.38 秒 | 参数更新使用 |
| val | 63 | 33.64 秒 | 保留；本次没有用它选择 checkpoint |
| test | 92 | 34.24 秒 | 仅最终评估，固定前 20 秒 |

先划分录音，再切片。这样不会把同一段录音相邻片段同时放到训练和测试里。

### 3.3 共同预处理与特征提取

仅 train 经上游高通、静音切片、归一化，一共生成 62 个切片。每个切片得到内容特征、音高数组、量化音高数组。A/B 复用同一份特征，不分别随机洗一次数据。

结果：云端上游 `logs/..._shared/` 里有 `0_gt_wavs`、`1_16k_wavs`、`3_feature768`、`2a_f0`、`2b-f0nsf`。`features.json` 记录它们的 SHA256。极短尾片段可以是合法提取结果；上游长度分桶决定是否参与训练。

## 4. 训练阶段：一“步”里面发生什么

1. 读取同一 batch 的内容特征、F0 与真实人声。
2. G 根据内容、音高和 speaker id 生成波形片段。
3. D 对真实/生成波形给出预测，计算判别器 loss；反向传播，更新 D。
4. 计算 G 的对抗、特征匹配、mel 重建、KL 等目标；反向传播，更新 G。
5. 确认原始 loss 和梯度有限，记录实际更新计数。
6. 固定步数保存权重；1000 次真实更新完成后停止该组。

A/B 的区别发生在第 0 步：A 的 G/D 随机，B 的 G/D 已加载固定官方预训练。共享的 HuBERT/RMVPE 不更新。B 从公开数据学到的先验不算“你的数据”，但它属于额外预训练成本。

结果：两组各自有 `start.json`（起始配置）、`metrics.jsonl`（真实步数/原始 loss/显存）、`summary.json`（实际完成状态）、`checkpoints/step0.pth`、`step500.pth`、`step1000.pth` 以及最终 `model.pth`。原生完整 G/D 与优化器 checkpoint 另保留在云端上游 `logs`；这里的 `.pth` 推理快照不是完整断点续训包。

## 5. “CUDA 测试通过”应该看什么

`torch.cuda.is_available() == True` 只表示程序能找到 CUDA。证据强度依次为：

| 看到的结果 | 实际能说明什么 |
|---|---|
| `nvidia-smi` 显示 4090 D | 云端能识别显卡 |
| `torch.cuda.is_available()` 为 True | PyTorch 能调用 CUDA |
| `start.json` 的 device、torch、cuda runtime 正确 | 训练初始化到了正确设备 |
| `metrics.jsonl` 中 step=1、10、20 持续增长，loss 有限 | 已发生真实 G/D 参数更新 |
| 两组 summary 的 complete 与相同实际步数 | 本轮更新协议完成 |
| 输出 WAV 能解码、时长正确、实际听过 | 推理与声音体验可以继续评估 |

任何一项都不能单独证明音质好。看 loss 降低也不能替代试听。

## 6. 现在怎么实操试听，界面长什么样

### 第一步：找到本机结果

打开 `voice_lab/runs/new4090d_ab_retry1`。若还有下载进行中，等 `DOWNLOAD_RECEIPT.json` 出现，表示 ZIP 已下载、SHA256 相同并完成解压。

预期结果：有 `LISTEN.html`、`listening`、`blind`、两个模型目录、报告及评分表。`LISTEN.html` 是取回后在本机生成的页面，不需要租 GPU 播放。

### 第二步：打开试听页面

双击 `LISTEN.html`，用浏览器打开。界面有三个音频播放控件：测试原声、匿名 X、匿名 Y。点击三角形播放；可拖动进度条、暂停和下载 WAV。

预期结果：听到固定的 20 秒测试内容及两种重建。你的音量、耳机和播放设置保持一致。

### 第三步：记录评分

用 Excel 打开 `listening_scores.csv`，填写 listener 和 X/Y 的四项 1–5 分。每项备注具体问题，例如“某个字不清楚”“气声变成杂音”“高音尾音不稳”。保存时保留 CSV 格式。

预期结果：得到个人盲评记录。一个人一段音频的评分只描述这次体验；没有统计推广结论。

### 第四步：揭晓

完成评分后展开页面“揭晓身份与原始输出”，打开 `blind_key.json`。再听 A/B 原始 WAV；匿名版做过 RMS 匹配和峰值限制，原始版保持模型直接输出。

预期结果：能把听感与初始化方式对应起来。未评分前不会宣布胜者。

### 第五步：看报告与证据

打开 `MEASURED_REPORT.md` 和 `training_curves.png`。报告列出实际步数、时间、峰值显存、F0 误差、数据划分与失败史。打开 JSON/JSONL 可以逐条核对数字。

预期结果：了解每个结论来自哪个文件，以及哪些问题还没有测。

## 7. 将另一段人声转换成你的音色

本次已经训练过；此操作使用已保存权重，**无需重新训练**。准备获授权、未参与本次训练的 10–20 秒干净人声 WAV。音频提供歌词/旋律；模型负责转换音色。

后续在确认价格与新运行预算后，有卡开机，上传新的 WAV 到 `/root/autodl-tmp/another_voice.wav`。进入你现有项目终端，运行一次推理：

```bash
cd /root/autodl-tmp/voice-self-compare-e64531a3
.rvc-venv/bin/python voicelab/rvc_infer_entry.py --model runs/new4090d_ab_retry1/finetune/model.pth --input /root/autodl-tmp/another_voice.wav --output /root/autodl-tmp/converted_to_me_B.wav --speaker-id 0 --pitch 0 --index-rate 0 --rms-mix-rate 1 --protect 0.33
```

A 的同条件对照把 `finetune` 改为 `scratch`，输出改为 `converted_to_me_A.wav`。下载两个输出 WAV 试听。模型、代码、公共特征权重仍在现有云端项目，无须再次上传原语言模型。若换机器，应复制本项目、公共资产、运行时修复及已训练的两套权重，核对哈希并恢复相同依赖。

移调 `pitch` 的单位是半音；本轮为 0。先保持本次设置再单独探索变调，避免同时改多个变量。本人一个 speaker 的模型没有“两位歌者之间音色比例”滑杆；这需要新的模型与数据实验。

## 8. 学代码的具体顺序

| 顺序 | 文件 / 函数 | 学完应该能回答 |
|---|---|---|
| 1 | `voicelab/rvc_data.py` | 如何解码、重采样、分整录音并验证哈希 |
| 2 | `voicelab/rvc_assets.py` | 哪些模型是公共冻结特征，哪些是 B 的起始 G/D |
| 3 | `preflight_rvc.py` | 为什么导入成功还要用真实 WAV 验证预处理输出 |
| 4 | `voicelab/rvc_compare.py` | 如何保证共同特征、相同 batch/seed、固定测试输入 |
| 5 | `upstream/RVC/train/train.py` | 一批数据如何先更新 D，再更新 G |
| 6 | `voicelab/rvc_bridge.py` | 怎样记录真实更新、未截断 loss，保存 0/500/1000 权重 |
| 7 | `voicelab/rvc_infer_entry.py` | 权重如何加载，输入人声如何变成输出 WAV |
| 8 | `voicelab/inference.py` | 音分误差怎样计算，为什么它不代表自然度 |
| 9 | `launch_rvc.py` | 限时退出、下载窗口、平台关机与真实账单的区别 |
| 10 | `fetch_rvc_results.py`、`render_rvc_report.py` | 如何安全取回实测产物、生成曲线/播放器/报告 |

可以边读边打开本轮 JSON 与 WAV，对照输入输出。不要在已完成的结果目录里重复启动训练；实验目录是证据，新的实验应使用新目录和独立预算。

## 9. 这次小数据试验能说明到哪一步

能检验：两种初始化是否能在你这批数据上完成真实更新、固定片段重建怎样、时间/显存/音高指标怎样。不能检验：任意歌者效果、实时延迟、多卡增益、商业音质、任意用户零样本音色混合。

要加强对照，依次扩展跨歌者授权输入、更多音域/唱法、多个随机种子、多首测试歌曲及多听者盲评。每一步都保留固定输入、公共控制变量与真实失败证据。
