# AutoDL 首次上机：只做 30 分钟内的入门检查

这个压缩包已经包含：项目 Python 源码、测试、教程、示例 raw 文本、匹配玩具模型的 `tokenizer.json`、`runs/chat_toy.pt` 和 `runs/smoke/base.pt`。后两者都是约 111 万参数的**玩具模型**。本包没有训练完成的 0.1B 模型，也不需要先上传一个。

## 租卡页面怎么选

在 AutoDL「我的实例 → 租用新实例」选择：**按量计费、1 张 NVIDIA RTX 4090（24GB）、Linux、带 PyTorch 2.4+ 和 CUDA 的基础镜像**。先用默认磁盘做本包实验；真正下载大语料前，再核算磁盘。结算页确认最终单价和硬盘费用。多卡等单卡流程跑通后再租。

## 上传与解压

实例运行后，打开 JupyterLab。在文件列表里点上传按钮，上传 `ai_fundamental_autodl_starter.zip`。JupyterLab 中打开 Terminal；如果 ZIP 上传到当前用户的主目录，运行：

```bash
mkdir -p ~/ai-fundamental
unzip ~/ai_fundamental_autodl_starter.zip -d ~/ai-fundamental
cd ~/ai-fundamental
ls
```

如果上传到了其他目录，先用 `pwd` 和 `ls` 找到 ZIP，再把 `unzip` 命令里的路径改成实际路径。解压后应该看到 `fundamental/`、`tests/`、`raw/`、`runs/`、`tokenizer.json` 和本说明。

## 验证 GPU 与模型

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install -r requirements.txt
python -m fundamental.chat --checkpoint runs/chat_toy.pt --tokenizer tokenizer.json --once hello
```

预期 `torch.cuda.is_available()` 为 `True`；玩具聊天模型回答 `Hi!`。如果 PyTorch 版本低于 2.4 或显示 `False`，先检查镜像/驱动，别继续付费长跑。命令行 `chat.py` 是本项目的聊天界面。

## 第一次云端训练

```bash
python -m fundamental.data --input raw/demo.jsonl --out clean --licenses CC0
python -m fundamental.tokenizer pack --tokenizer tokenizer.json --input-dir clean --out tokens
python -m fundamental.train --data tokens --tokenizer tokenizer.json --config tiny --batch-size 1 --seq-len 32 --steps 20 --eval-every 10 --out runs/cloud_tiny
tail -2 runs/cloud_tiny/metrics.jsonl
```

这只是在合成文本上验证 GPU 路径。demo 的 train 只有约 45,586 tokens，不能训练出有用的 0.1B 模型。

## Triton 正确性与速度

如果打开了新的 Terminal，先回到项目目录：`cd ~/ai-fundamental`。终端提示符末尾应为 `~/ai-fundamental#`，否则相对路径 `tests/...` 会找不到文件。

```bash
python -m pip install triton
python -m pytest -q tests/test_triton_attention.py
python -m fundamental.bench attention --seq-len 512 --out runs/attention_gpu.jsonl
```

先看到 6 个测试通过，再看 `max_abs_error` 和 `triton_speedup`。测试失败时先保存报错；不要把失败 kernel 的计时写入报告。`triton_speedup < 1` 表示这个形状下自写 kernel 比 PyTorch SDPA 慢，是正常且有价值的观察。

## 结束计费

把需要的 `runs/` 文件通过 JupyterLab 下载回本机，然后在 AutoDL 控制台**关机**。AutoDL 的 [官方快速开始](https://www.autodl.com/docs/quick_start/)说明「运行中」开始计费；[计费说明](https://api.autodl.com/docs/price/)说明按量实例关机后结束 GPU 计费，扩容磁盘可能继续收费。
