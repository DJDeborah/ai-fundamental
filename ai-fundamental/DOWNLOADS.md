# 项目与模型下载

- [GitHub 项目](https://github.com/DJDeborah/sideproject/tree/main/ai-fundamental)：课程源码、固定上游源码、教程、数据划分、实测报告和音频。
- [Release 下载](https://github.com/DJDeborah/sideproject/releases/tag/ai-fundamental-2026-10-03)：两套最终模型、完整本机结果包、可复现项目包。
- `rvc_A_scratch_step1000.pth`：A 的最终转换模型。
- `rvc_B_finetune_step1000.pth`：B 的最终转换模型。
- `voice_self_rvc_results_20261003.zip`：模型与 0/500/1000 快照、WAV、原始训练指标和教程。
- `ai_fundamental_project_20261003.zip`：便携源码、训练录音、报告及本机玩具模型。

试听页面播放已生成的 WAV，不需要 GPU。新音频转换按 `voice_lab/RVC_STEP_BY_STEP.md` 操作。
大规模语言模型的云端权重当前没有完整下载到本机；这里提供其源码、训练指标和报告，不冒充已归档的 0.1B checkpoint。

结果 ZIP 是经过 CRC / 模型与声音 SHA256 验证的**本机重新打包交付**。原云端 ZIP 的末段及部分失败历史仍待无卡模式补取。预算记录是保护估计，不是平台账单。
第三方源码保留各自 LICENSE；录音、模型、数据集不自动继承第三方源码许可。
