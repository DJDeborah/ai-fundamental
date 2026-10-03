# 本人清唱 RVC 对照实验记录

生成于 2026-10-02T16:26:58.705736+00:00。同一 RVC v2 40k 架构、同一训练数据、共享 HuBERT/RMVPE；A 随机初始化生成器与判别器，B 加载官方 f0G40k/f0D40k 预训练权重。

## 数据与控制变量

训练 311.38 秒；验证 33.64 秒；测试 34.24 秒。完整录音先划分，再仅对 train 做上游切片。歌曲分组确认：True。

共同设置：seed=1234，batch=4，FP32，AdamW，40kHz，v2，F0 enabled；检索索引比例=0，移调=0，推理 seed=1234。上游训练预处理还有 48Hz 高通、静音切片及幅度归一化；两组只预处理一次。

| 组别 | 更新步数 | 状态 | 包含初始化/保存的秒数 |
|---|---:|---|---:|
| scratch | 1000 | complete | 320.7 |
| finetune | 1000 | complete | 313.8 |

完成相同步数：True。若为 False，不能宣称已完成受控训练对照。

## 如何体验

打开 rvc_listen.ipynb，选择 RVC Comparison 内核。先听原声和 blind/ 中匿名的 X、Y，填写 listening_scores.csv，再查看 blind_key.json，之后可听两组未匹配响度的原始输出。评分 1–5：自然度、歌词可懂度、音色相似度、音高稳定性；每项应加备注。

原声转回本人用于检查重建保真。因为所有录音都来自本人，此实验还没有评估“另一人的声音转成你”；需要另一位授权歌者的未训练输入才能测这个能力。

## 结论状态

当前没有主观听感胜负结论。训练 loss 不是听感质量；两个初始化的 GAN 总 loss 也不能直接当成质量排名。metrics.jsonl 保存显示截断之前的原始 loss。自动音高评估保存在 listening/*_metrics.json 与 comparison.json，若对应文件缺失表示未完成；它假设输入输出时间对齐，不衡量身份相似度或自然度。

这是低预算学习实验，单个随机种子、约六分钟数据、各一段验证和测试，不能支撑正式论文的统计结论。预训练使用了额外数据，其规模/组成未在本实验复现，因此这里比较的是相同本轮适配成本下的效果，不能声称总训练算力相同。

## 来源

[RVC 官方 CLI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/docs/en/cli.md)；[官方权重仓库](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/e6d0c1a17da07c33557852f9dfa2bd44cc75737d)。上游快照、数据、特征、checkpoint 哈希随结果保存。商业许可需要分别核验代码、权重和数据。
