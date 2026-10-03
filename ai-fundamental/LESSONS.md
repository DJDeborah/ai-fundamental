# 分段学习卡

按 0→7 站走，每站先读代码，再运行最小命令，最后只改一个变量。每次记录命令、随机种子、观察和自己的解释。

## 0. Raw dump → 可追溯语料

读 `data.py` 的 `normalize`、`simhash`、`split_for`。把相同段落重复放进 JSONL，观察 manifest 的 `exact_duplicate`；再把一个来源的 license 改成未授权值，观察过滤计数。思考：按文档哈希切分能防住哪些泄漏，不能防住哪些？真实网页需要额外的解析、质量抽样、隐私及 benchmark 污染检查。

## 1. 手写 byte-level BPE

读 `tokenizer.py` 的 `train_bpe` 和 `encode`。在两句短文本上手算第一次合并，核对训练得到的 pair；试一段中文和 emoji，检查 `decode(encode(x)) == x`。改变 `--vocab-size`，记录 bytes/token 与未见文本的 token 数。注意 EOS 是独立 token，不是 UTF-8 字节。

## 2. Transformer 与 0.1B 参数

读 `model.py` 的 `apply_rope`、`Attention.forward`、`Block.forward`。推导一个 block 的参数主项 (4d^2+3dd_{ff})，再加 tied embedding (Vd)，估算 0.1B。做因果性实验：只改第 9 个 token，前 8 个位置 logits 应不变。看 `tests/test_model.py`。

## 3. 预训练

读 `train.py` 的 `sample_batch`、gradient accumulation 和验证代码。核对 `tokens_seen = step × global_tokens`。用 tiny 跑 20 步后看 `metrics.jsonl`，不要用训练 loss 当泛化结论。尝试更大 `seq_len`，预测注意力计算/内存如何变化。

## 4. Kernel 与多卡

读 `triton_attention.py` 的 online softmax 更新。先用 `bench attention` 检查数值，再对 T=128/512/1024 测 forward 时间。训练端仍走可反传的 SDPA。多卡在同一租赁机上比较 1/2/4 GPU，记录最终 val loss、总 token、world size、吞吐和 GPU-hours；推理脚本测无 KV cache 的 aggregate throughput。

## 5. Scaling law

读 `scaling.py` 的非负拟合与留出点选择。至少准备 3 个 N × 3 个 D 的日志；画 (N,D,L) 散点和残差。比较拟合误差与最大规模留出误差。把目标 N 翻十倍时，明确注明它超出观测范围；不要把拟合曲线当物理定律。

## 6. 同一基座后训练

读 `posttrain.py` 的 `completion_stats`、SFT、DPO、RLVR 三个分支。检查各 checkpoint 的 `root_base_sha256` 一致，且 DPO/RLVR 的 `parent_sha256` 都指向同一份 SFT，再在 `tasks/eval.jsonl` 跑 `evaluate.py`。RLVR 可能持续 `skipped_zero_variance`，这正是稀疏奖励的实验事实，不能伪称提升。换不同 seed，报告均值和不确定性。

## 7. 可验证长程 Agent 与 self-judge

读 `env.py` 的 `step`，手写一个 oracle action 和一个错误 action；随后读 `agent.py` 的 oracle 轨迹、imitation、在线回报和候选排序。用同一组 held-out seeds 比较 horizon=4/8/16 和 judge on/off。记录成功率与每次决策的生成次数；judge 分数只是模型猜测，最终由环境判断。

## 交付报告

打开 `report/neurips_style.md`，逐项填真实数值、设备、数据来源、消融和失败实验。先检查“主要结论是否真的由表格支持”，再写摘要结果句。单机 CPU 冒烟运行只能支持“实现路径可执行”，不能支持模型能力或 GPU 加速结论。
