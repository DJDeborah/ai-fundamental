# 歌声音色转换路线核对

核对日期为 2026-10-01。这里区分公开代码、权重声明和本实验的技术选择。表中的许可标注描述链接内容，不能把一个组件的许可推广成整条处理链的商用结论。

## 这轮选择

| 项目或组件 | 核实的内容 | 本轮安排 |
|---|---|---|
| [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) | 有训练入口、多歌手模型与 speaker-mix；仓库许可 MIT | 训练转换模块，先单歌手，再双歌手 |
| [ContentVec 权重](https://huggingface.co/lengyue233/content-vec-best) | 模型页标注 MIT；固定 revision 在 sources.json | 作为冻结内容编码器 |
| [PC-NSF-HiFiGAN 权重发布](https://github.com/openvpi/vocoders/releases/tag/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02) | 发布页明确标注 CC BY-NC-SA 4.0 | 非商业学习使用；商用须解决此权重许可 |
| [openvpi vocoders 代码许可](https://github.com/openvpi/vocoders/blob/main/LICENSE) | 仓库使用 AGPL；与权重许可分开 | 单独记录，部署前需要逐项检查 |
| [RMVPE 发布](https://github.com/yxlllc/RMVPE/releases/tag/230917) | 上游提供预训练音高提取器；此次未核实独立权重商用授权 | 记录来源，不声称全部组件都允许商用 |
| [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | 有训练网页界面和模型合并功能 | 可作为另一条单音色训练路线，暂不混用实验环境 |
| [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) | 当前仓库定位为推理代码；README 声明代码与权重采用 Apache-2.0 | 后续零样本推理对照；公开推理入口不等于可直接执行的训练教程 |

原始项目列表中的其它项目没有纳入这次工具包，也没有在这轮逐个完成许可核对。商业产品应针对实际采用的版本、模型文件和录音授权做记录。

## 技术假设需要实测

1. 训练单一声音有助于学习它的重合成质量。200 步只验证流程；质量取决于录音、覆盖范围和训练情况。
2. 同一多歌手模型的 embedding 线性插值可能带来连续的音色变化。本轮会用五个 alpha 检查自然度与变化方向，不预设结果单调。
3. SLERP（球面线性插值）只是另一种几何插值；它不保证混合音色好听，也不保证新用户的 embedding 有效。DDSP-SVC 原生实现并非 SLERP。
4. 干净训练、带噪测试适合测泛化。将噪声无标注地混进首轮训练不能直接证明抗噪性；鲁棒训练需要受控的增广与比较。
5. 修音需要一个目标音高，例如音符序列或调式。当前实验保持提取出的 F0，固定半音转调不能替代修音强度 beta。
6. 音高一致性不能证明音色相似；声音模型的自评也不能替代独立评估。人工盲听先作为可理解的质量证据。

## 处理链怎样逐步扩展

第一轮输入干净清唱，测训练与重合成。第二轮增加授权音色，测比例扫描。第三轮固定这些模型，用独立测试录音比较未处理输入、分离后输入、保守降噪后的输入。第四轮再增加可指定目标的修音与伴奏混回。每一轮增加一个变化，保持测试歌曲和评估规则一致。
