# 歌声音色转换学习实验

这个目录用于学习一个可试听、可记录的歌声音色转换实验。第一轮训练你本人的音色，输入一段未参与训练的清唱，输出 WAV；第二轮在同一模型内加入另一位授权歌手，比较五个音色插值权重。底座采用固定版本的 DDSP-SVC，学习代码负责数据检查、有限训练、试听与实验记录。

先读 [从录音到试听的操作教程](TUTORIAL.md)，其中每一步都写明操作入口、输入、输出和通过标准。首次上传用 `dist/voice_lab_autodl_starter.zip`，不需要上传前面的语言模型。

本学习路线复用预训练 ContentVec、RMVPE 和声码器，训练 DDSP 与 Rectified Flow 转换模块。它的模型规模由配置和运行时参数统计决定，不沿用语言模型的 0.1B 目标。

## 实验入口

| 阶段 | 操作入口 | 输出与用途 |
|---|---|---|
| 录音检查 | `python -m voicelab inventory` | 时长、采样率、音量、削波记录 |
| 切片与分组 | `python -m voicelab prepare --manifest manifest.csv` | `data/self_v1/manifest.json` 和 WAV 切片 |
| 配置 | `python -m voicelab config` | 固定数据来源、模型大小和 batch |
| 预训练部件 | `python -m voicelab assets` | 三个指定部件、下载来源及 SHA256 |
| 特征提取 | `python -m voicelab preprocess` | 内容、F0、音量、Mel 等 `.npy` |
| 学习训练 | `python -m voicelab train --steps 200 --minutes 15` | step 0、step 100/200、latest checkpoint 和日志 |
| 转换 | `python -m voicelab infer ...` | 原始输入的规范 WAV 和转换 WAV |
| 试听 | JupyterLab 打开 `listen.ipynb` | 浏览器播放，和输入做 A/B 比较 |
| 评估 | `python -m voicelab evaluate ...` | 音高误差、清浊音一致性和音量统计 |
| 记录 | `python -m voicelab report --data data/self_v1` | 由真实文件生成的 `runs/REPORT.md` |

## 已验证范围

2026-10-01 本机 CPU 验证了 7 项数据与记录工具测试；没有使用你的录音，也没有完成 GPU 声音模型训练。初版 GPU 训练入口和依赖方案需要在云端按教程验证。首轮训练 200 步只用于发现接口、数值或显存问题，不能保证声音好听。

## 音色比例的含义

DDSP-SVC 在多歌手情况下对已训练歌手 embedding 做加权求和。本工具的 alpha=0 选择歌手 1，alpha=1 选择歌手 2；alpha=0.5 是两个向量等权。它不是经过校准的感知音色百分比，alpha=0 也会经过模型重合成。单歌手模型只允许 alpha=0。新用户无须训练、自动修音 beta、分离与自动混音仍是后续实验。

## 来源和许可

源代码固定在 [DDSP-SVC 的 d6dd52f 版本](https://github.com/yxlllc/DDSP-SVC/tree/d6dd52fcdbce07fbe4c4d120c911b0d63d53e2d2)，上游 MIT 许可保存在 `upstream/DDSP-SVC/LICENSE`。默认声码器权重有非商业限制，见 [许可与路线核对](SOURCES_AND_ROUTE.md)。代码、模型权重和录音数据的许可分别记录在 `sources.json` 与数据 manifest 中。
