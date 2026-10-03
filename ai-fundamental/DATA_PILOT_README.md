# FineWeb-Edu 1000 篇网页文本试点

## 来源与限制

本包的 `raw/fineweb_edu_pilot.jsonl` 来自 [Hugging Face FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 的 `sample-10BT` 配置。使用 [官方 Dataset Viewer `/rows` API](https://huggingface.co/docs/dataset-viewer/rows) 按 row index 读取**前 1000 行**，每页 100 行；全部页面完整、行号连续，没有 `truncated_cells`。这是数据通路试点，不是随机评测样本，也不是原始 Common Crawl WARC/WET dump。

数据卡标注 ODC-By，并说明还受 Common Crawl 使用条款约束。原文只用于当前研究试点；来源 URL/ID 保留在 JSONL 的 `source` / `source_id` 字段。`raw/fineweb_edu_pilot.manifest.json` 记录下载时间、Hub SHA、字节数和 SHA-256。Hub SHA 是下载时查询到的仓库版本；Dataset Viewer 的缓存不保证与该 SHA 精确对应。

## 在 AutoDL 上上传与解压

将 `fineweb_edu_pilot.zip` 上传到 JupyterLab 文件列表。打开 Terminal，运行：

```bash
cd /root/ai-fundamental
unzip -o "$(find /root -maxdepth 4 -name fineweb_edu_pilot.zip -print -quit)" -d /root/ai-fundamental
sha256sum raw/fineweb_edu_pilot.jsonl
```

期望 SHA-256：`ee8301fdc6341f12c91c8b6ae28ee5cd3e60e4990de0d31e03668a43f5d8dc9c`。若 `find` 没有找到 ZIP，先定位上传路径；不要执行空路径 `unzip`。

## 清洗与验收

前一次连接 Hugging Face 失败后生成的空 `clean_fineweb_pilot/manifest.json` 不是有效结果。重新运行：

```bash
time python -m fundamental.data --input raw/fineweb_edu_pilot.jsonl --out clean_fineweb_pilot --licenses odc-by
cat clean_fineweb_pilot/manifest.json
```

本机同版代码的结果：`raw=1000`、`train=969`、`val=14`、`test=17`。此次没有被许可、长度或去重过滤掉的条目。`time` 的墙钟时间依机器而定。这 1000 篇不足以训练有意义的 0.1B 模型，先量化 Tokenizer 的速度和 token 产量，再决定扩量。

`/root/ai-fundamental` 当前位于约 30GB 的系统盘；此试点只有约 5MB，可以放这里。后续大语料和 checkpoint 应先用 `df -h /root/autodl-tmp` 检查独立数据盘，再规划路径和备份。
