"""Build the repository landing README and real-data audio preview graphic."""
import argparse
import json
import shutil
import wave
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repository.resolve()
    project = repo / 'ai-fundamental'
    site = project / 'website'
    pages = 'https://DJDeborah.github.io/ai-fundamental/'
    github = 'https://github.com/DJDeborah/ai-fundamental'
    release = github + '/releases/tag/ai-fundamental-2026-10-03'

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 4.8), constrained_layout=True)
    fig.patch.set_facecolor('#f4f6f5')
    samples = [('Original / held-out vocal', 'listening/input.wav', '#175f69'),
               ('Anonymous X', 'blind/X.wav', '#3973ae'),
               ('Anonymous Y', 'blind/Y.wav', '#bb722a')]
    for ax, (label, relative, color) in zip(axes, samples):
        with wave.open(str(site / relative), 'rb') as wav:
            assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
            rate = wav.getframerate()
            values = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2') / 32768.0
        stride = max(1, len(values) // 2400)
        keep = len(values) // stride * stride
        blocks = values[:keep].reshape(-1, stride)
        times = (np.arange(len(blocks)) + .5) * stride / rate
        ax.fill_between(times, blocks.min(axis=1), blocks.max(axis=1), color=color, alpha=.9)
        ax.set_xlim(0, 20)
        ax.set_ylim(-1, 1)
        ax.set_ylabel('Amplitude')
        ax.set_title(label, loc='left', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=.16)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].set_xlabel('Time (seconds)')
    fig.suptitle('LISTEN TO THE EXPERIMENT  |  Original + blind X/Y', fontsize=17, fontweight='bold')
    fig.savefig(site / 'audio_preview.png', dpi=145, facecolor=fig.get_facecolor())
    plt.close(fig)

    intro = f'''# AI Fundamental

**从代码到真实训练，再到你能亲耳听到的结果。**

Tokenizer、0.1B Transformer、Triton attention、scaling law，以及本人清唱的 RVC 随机初始化 / 微调对照。源码、术语教程、真实日志、图表、失败记录与模型下载放在同一条学习路径里。

### 先体验，再读代码

**[▶ 打开在线试听网页]({pages})** · [声音实测报告]({pages}report.html) · [逐步操作与术语]({pages}tutorial.html) · [完整项目 / 模型下载]({release})

[![原声和匿名 X/Y 的真实波形；点击打开播放器](ai-fundamental/website/audio_preview.png)]({pages})

点击上图进入带播放按钮的网页。手机和电脑都可播放已有音频，无需安装 Python、租 GPU 或登录 AutoDL。

| 试听内容 | 播放 / 下载 |
|---|---|
| 未参与训练的本人原声，20 秒 | [原声 WAV]({pages}listening/input.wav) |
| 匿名 X，19.98 秒 | [X WAV]({pages}blind/X.wav) |
| 匿名 Y，19.98 秒 | [Y WAV]({pages}blind/Y.wav) |

建议先听 X/Y 并评分，再在网页展开身份，查看 A/B 原始音频及模型。

## 项目亮点

- **从零拆解语言模型。** 手写 byte-level BPE、RoPE、RMSNorm、SwiGLU 和因果 Transformer；实际最大模型为 **100,679,424 参数**。
- **用实测检查 scaling。** 三种模型大小 × 三个种子，得到 **27 条观测**；预留最大规模格检验外推，保留误差和不确定性。
- **自己写 GPU 内核，再自己测。** Triton 分块因果注意力先检查数值，再用交替 CUDA Event 对照 PyTorch SDPA；同时公开加速与减速的形状。
- **声音对照可直接试听。** 同一 RVC v2 40 kHz 架构、同一录音划分、batch=4、seed=1234、FP32，A/B **各完成 1000 次真实 G/D 更新**。
- **新手也有可观察的操作结果。** 教程写出术语全称、命令入口、输入 / 输出、应该看到的文件和数字；附可运行的后训练与 Agent 教学接口。
- **保留证据与失败。** 原始 JSONL、环境、固定源码 revision、资产 / 模型 SHA256、0/500/1000 快照和失败修复记录可复核。

```mermaid
flowchart LR
    Text[文本与许可] --> Clean[清洗与固定切分]
    Clean --> BPE[手写 Tokenizer]
    BPE --> LM[0.1B Transformer]
    LM --> Measure[Scaling 与 Triton 实测]
    Vocal[本人清唱] --> Split[整文件 train / val / test]
    Split --> Features[共用冻结 HuBERT / RMVPE]
    Features --> A[A: 随机 G/D]
    Features --> B[B: 预训练 G/D 微调]
    A --> Listen[匿名试听与客观评价]
    B --> Listen
```

## 真实结果一览

| 实验 | 已测结果 | 读结果时的边界 |
|---|---|---|
| 100.68M 模型，300M tokens | 验证 loss **2.16986 ± 0.00877**；三个种子 | 同一语料和分词器的均值 ± 标准差 |
| scaling 留出格 | 预测 **2.20104**，误差 **0.03118** | 局部拟合，外推误差超过种子波动 |
| Triton attention 前向 | T=1024：**1.2936×**；T=2048 / 4096：**0.9663× / 0.8585×** | 指定形状的前向结果，不等同于训练整体加速 |
| RVC A / B | 各 1000 更新；耗时 **320.70 / 313.83 秒** | 包含初始化与保存，共用预训练特征提取器 |
| 固定测试片段音高 | pYIN 误差中位数 A **3050 cent**，B **0 cent** | 离散估计的中位数，不代表真实误差为零 |

**当前声音片段上 B 的估计音高保留更好；还没有盲听评分，因此不宣布自然度或音色质量胜者。** A 仅转换 G/D 随机初始化，不是整个语音系统从零。本人约六分钟数据、单种子、单测试片段的重建，不能验证跨歌者转换能力。

### 语言模型：看 scaling 与外推误差

![三种模型大小、三个种子的实际曲线与留出预测](ai-fundamental/report/figures/scaling_fit.png)

[完整语言模型报告]({pages}lm-report.html) · [PDF](ai-fundamental/output/pdf/0.1B_transformer_final_report.pdf) · [原始指标](ai-fundamental/report/data/cloud/)

### 歌声模型：看原始训练轨迹，再听声音

![A/B 六项真实训练损失曲线](ai-fundamental/voice_lab/runs/new4090d_ab_retry1/training_curves.png)

GAN loss 是训练目标，不能当作听感评分。[打开播放器]({pages})，自行比较清晰度、自然度和音高稳定性。

## 学习路线

| 入口 | 内容 |
|---|---|
| [语言模型新手教程](ai-fundamental/BEGINNER_TUTORIAL.md) | 术语、环境、逐步命令、CUDA 证据和模型体验 |
| [语言模型源码](ai-fundamental/fundamental/) | 数据、BPE、Transformer、训练、Triton、后训练、Agent |
| [声音逐步教程](ai-fundamental/voice_lab/RVC_STEP_BY_STEP.md) | 文件准备、训练 / 微调、试听、换声命令与结果解释 |
| [声音实验实测报告](ai-fundamental/voice_lab/runs/new4090d_ab_retry1/MEASURED_REPORT.md) | 同步数对照、耗时、显存、音高与局限 |
| [声音实验记录](ai-fundamental/voice_lab/RVC_EXPERIMENT_RECORD.md) | 首轮失败、修复、预算与恢复过程 |
| [项目完整操作 README](ai-fundamental/README.md) | 全部模块的命令、文件结构和复现路径 |

## 下载与继续运行

[Release 下载页]({release})提供：

- `rvc_A_scratch_step1000.pth` / `rvc_B_finetune_step1000.pth`：两套最终推理模型。
- `voice_self_rvc_results_20261003.zip`：模型、0/500/1000 快照、WAV、原始日志、报告和教程。
- `ai_fundamental_project_20261003.zip`：便携源码、训练录音、报告及本机教学 checkpoint。
- `ai_fundamental_autodl_starter.zip`：语言模型初学者云端入门包。
- `SHA256SUMS.txt`：下载后校验文件字节。

```bash
git clone https://github.com/DJDeborah/ai-fundamental.git
cd ai-fundamental/ai-fundamental
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m fundamental.model --config tiny --vocab-size 512
```

声音转换另读 `voice_lab/RVC_STEP_BY_STEP.md`，需要按固定 revision 下载共享 HuBERT / RMVPE 资产；播放网页已有 WAV 则没有这些依赖。

## 证据范围与归档状态

SFT、DPO、RLVR 和 Agent 已有教学代码及失败探针；正式后训练、双卡与长程 Agent 对照没有完成。0.1B 云端权重尚未完整下载，本仓库提供其代码、词表、实测指标和报告。NeurIPS 风格是报告组织方式，不表示论文录用或已获得通用能力。

声音最终模型、快照和 WAV 已恢复并核对 SHA256。Release 的声音结果 ZIP 是**完整的本机重新打包交付**；原云端 ZIP 的末段及部分失败历史仍待补取。预算估算不是平台账单。

第三方源码保留原 LICENSE 和固定 revision。录音、模型及数据集的许可与代码许可分别记录，不自动继承第三方代码许可。
'''
    (repo / 'README.md').write_text(intro, encoding='utf-8')
    # Keep the deep project README, with a compact visual/audio entry at the top.
    readme = project / 'README.md'
    text = readme.read_text(encoding='utf-8')
    heading, _, rest = text.partition('\n')
    banner = f'''\n\n**[▶ 在线播放：原声 / 匿名 X / 匿名 Y / A / B]({pages})** · [项目亮点与图表](../README.md) · [模型下载]({release})

[![原声与匿名输出的实际波形；点击进入播放器](website/audio_preview.png)]({pages})

实际完成：100.68M Transformer 三种子实验、Triton 前向基准、局部 scaling 留出检验；本人清唱 RVC A/B 各 1000 次真实更新。后训练与 Agent 的正式能力对照尚未完成，详见报告。
'''
    readme.write_text(heading + banner + '\n' + rest, encoding='utf-8')
    # Branch-based Pages publishes /docs. The same site also stays in the project archive.
    shutil.copytree(site, repo / 'docs', dirs_exist_ok=True)
    (repo / '.gitignore').write_text('''**/.venv/
**/.rvc-venv/
**/.voice-venv/
**/.cloud-access/
**/__pycache__/
**/.pytest_cache/
*.pyc
*.pt
*.pth
*.bin
*.npy
*.partial
''', encoding='utf-8')
    print(json.dumps({'repository': str(repo), 'readme_highlights': True,
                      'visual': 'ai-fundamental/website/audio_preview.png',
                      'pages_source': 'main:/docs', 'pages_url': pages}, ensure_ascii=False))


if __name__ == '__main__':
    main()
