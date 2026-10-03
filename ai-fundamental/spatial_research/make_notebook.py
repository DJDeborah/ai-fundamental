"""Build a beginner notebook and code-only upload bundle."""
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
plan = (root / 'RESEARCH_PLAN.txt').read_text(encoding='utf-8')


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(True)}


def code(s):
    return {'cell_type': 'code', 'metadata': {}, 'source': s.splitlines(True),
            'execution_count': None, 'outputs': []}


cells = [
    md('# 合成到真实视觉迁移与 GPU 训练调度\n\n先复刻，再改一个变量。当前真实图像实验未执行。\n\n点击单元格，按 Shift+Enter 运行；不要求先租更多 GPU。'),
    md('## 第 0 步：环境\n\n此格只检查版本与 GPU，不训练。GPU = Graphics Processing Unit，图形处理器；CUDA 是 NVIDIA 的 GPU 计算平台。'),
    code('import sys, pathlib, json, subprocess\nprint("Python:", sys.version)\nprint("目录:", pathlib.Path.cwd())\nimport torch\nprint("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)\nprint("CUDA可用:", torch.cuda.is_available())\nif torch.cuda.is_available():\n    print("GPU:", torch.cuda.get_device_name(0))\n'),
    md('## 第 1 步：调度演示\n\nFCFS = First-Come, First-Served，先到先服务。SPT = Shortest Processing Time，最短任务优先。这里是模拟假设耗时，不是真实 GPU 训练。'),
    code('from schedule import replay\njobs = [\n    {"id":"A", "duration_s":3600, "predicted_s":3500},\n    {"id":"B", "duration_s":600, "predicted_s":650},\n    {"id":"C", "duration_s":300, "predicted_s":320},\n]\nresults = [replay(jobs, policy) for policy in ["fcfs","oracle_spt","predicted_spt"]]\nfor r in results:\n    print(r["policy"], "顺序", [j["id"] for j in r["jobs"]], "平均完成(分钟)", r["mean_flow_s"]/60, "全部完成(分钟)", r["makespan_s"]/60)\npathlib.Path("runs").mkdir(exist_ok=True)\npathlib.Path("runs/queue_demo.json").write_text(json.dumps(results, indent=2))\n'),
    md('预期：FCFS 68.33 分钟，短任务优先 31.67 分钟，全部完成都是 75 分钟。接下来故意把长任务预测成很短，看预测错误是否让调度变差。'),
    code('bad = [dict(j) for j in jobs]\nbad[0]["predicted_s"] = 1\nr = replay(bad, "predicted_spt")\nprint("预测错误后的顺序", [j["id"] for j in r["jobs"]], "平均完成分钟", r["mean_flow_s"]/60)\n'),
    md('## 第 2 步起：完整研究计划\n\n下面包含公开数据、模型、方法、参数对照和验收条件。数据下载和真实训练在环境检查、数据条款确认、分组完成后执行。'),
    md('```text\n' + plan + '\n```'),
    md('## 训练接口预览\n\n只显示帮助，不下载权重或数据。正式开始前先建立三个真实域分组目录。'),
    code('result = subprocess.run([sys.executable, "train.py", "--help"], text=True, capture_output=True)\nprint(result.stdout or result.stderr)\n'),
    md('## 正式命令模板（先读，不自动执行）\n\n```bash\npython train.py --source /root/autodl-tmp/spatial-research/data/train --eval /root/autodl-tmp/spatial-research/data/target_dev --steps 10 --out runs/source_probe_42\npython train.py --source /root/autodl-tmp/spatial-research/data/train --target /root/autodl-tmp/spatial-research/data/target_adapt --eval /root/autodl-tmp/spatial-research/data/target_dev --method coral --steps 100 --coral-weight 1 --out runs/coral_100_42\n```\n\nCORAL 模型每步多算一批真实图像，需要额外计算。每个 --out 用新目录，避免覆盖日志。'),
]
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
                                       'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
for i, cell in enumerate(cells):
    cell['id'] = f'spatial-{i:02d}'
(root / 'research.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
out = root.parent / 'dist' / 'spatial_research_starter.zip'
out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for name in ['schedule.py', 'train.py', 'RESEARCH_PLAN.txt', 'research.ipynb']:
        z.write(root / name, 'spatial_research/' + name)
print(out)
