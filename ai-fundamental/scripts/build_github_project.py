"""Export portable source/evidence and a static audio website for GitHub.

No credentials, virtual environments, interrupted downloads or wheel caches.
Large experiment models are published as Release assets, separately from Git.
"""
import argparse
import hashlib
import html
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'voice_lab/runs/new4090d_ab_retry1'
BLOCKED_PARTS = {'.git', '.venv', '.cloud-access', '__pycache__', '.pytest_cache',
                 '.rvc-venv', '.voice-venv', 'node_modules'}
TEXT_SUFFIXES = {'.py', '.sh', '.md', '.txt', '.json', '.jsonl', '.yaml', '.yml',
                 '.toml', '.html', '.csv', '.ipynb', '.ps1', '.gitignore', '.gitmodules'}
SECRET_RULES = {
    'GitHub token': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
    'OpenAI token': r'\bsk-(?:proj-)?[A-Za-z0-9_-]{35,}\b',
    'private key': r'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----',
    'credential in URL': r'https?://[^\s/:]+:[^\s/@]{6,}@',
    'literal secret assignment': r'''(?im)^\s*(?:password|api_key|access_token|auth_token|jupyter_token)\s*=\s*["'][^"'\s]{8,}["']''',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def eligible(path):
    return (path.is_file() and not BLOCKED_PARTS.intersection(path.parts)
            and path.suffix.lower() not in {'.pyc', '.pt', '.pth', '.bin', '.npy', '.partial', '.zip'}
            and path.stat().st_size < 45 * 1024 * 1024)


def collect_source():
    files = set()
    for path in ROOT.iterdir():
        if path.is_file() and eligible(path):
            files.add(path)
    for directory in ('fundamental', 'scripts', 'tests', 'tasks', 'spatial_research',
                      'report', 'output', 'runs', 'clean', 'tokens'):
        for path in (ROOT / directory).rglob('*'):
            # Terminal screenshots can contain machine/session information.
            if directory == 'report' and 'evidence' in path.relative_to(ROOT).parts:
                continue
            if directory == 'spatial_research' and path.name.startswith('jupyter_'):
                continue
            if eligible(path):
                files.add(path)
    for path in (ROOT / 'voice_lab').iterdir():
        if eligible(path):
            files.add(path)
    for directory in ('voicelab', 'configs', 'tests', 'data'):
        files.update(p for p in (ROOT / 'voice_lab' / directory).rglob('*') if eligible(p))
    # New uploads/references and model caches stay on the user's computer.
    workbench = ROOT / 'voice_lab/voice_workbench'
    files.update(p for p in workbench.glob('*')
                 if eligible(p) and (p.suffix.lower() in {'.py', '.md', '.txt', '.ps1'}
                                     or p.name == '.gitignore'))
    files.update(p for p in (workbench / 'static').rglob('*') if eligible(p))
    files.update(p for p in (workbench / 'static_demo').rglob('*') if eligible(p))
    audit = ROOT / 'voice_lab/runs/quality_audit_20261003'
    files.update(p for p in audit.glob('*')
                 if eligible(p) and p.name in {'QUALITY_AUDIT.md', 'AUDIT.json',
                                              'audit_outputs.py', 'spectrogram_envelope.png',
                                              'OPENVOICE_BENCHMARK.json'})
    for snapshot, vendor in (('RVC_SNAPSHOT.json', 'RVC'), ('UPSTREAM_SNAPSHOT.json', 'DDSP-SVC')):
        manifest = json.loads((ROOT / 'voice_lab' / snapshot).read_text(encoding='utf-8'))
        for relative, expected in manifest['files'].items():
            path = ROOT / 'voice_lab/upstream' / vendor / relative
            if eligible(path):
                if sha(path) != expected:
                    raise RuntimeError('Pinned source changed: ' + str(path.relative_to(ROOT)))
                files.add(path)
    files.update(p for p in RESULT.rglob('*') if eligible(p))
    # Keep the deterministic toy corpus; real text sources have download scripts.
    files.add(ROOT / 'raw/demo.jsonl')
    files.add(ROOT / 'raw/fineweb_edu_pilot.manifest.json')
    return sorted(files)


def scan_secrets(directory):
    hits = []
    for path in directory.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        content = path.read_text(encoding='utf-8', errors='replace')
        for rule, expression in SECRET_RULES.items():
            for match in re.finditer(expression, content):
                hits.append({'file': path.relative_to(directory).as_posix(), 'rule': rule,
                             'line': content.count('\n', 0, match.start()) + 1})
    if hits:
        # Do not print matched values, even when this scan stops publication.
        raise RuntimeError('Review secret scan findings: ' + json.dumps(hits))


CSS = '''
:root{color-scheme:light;--ink:#18212c;--accent:#175f69;--muted:#53636d}
*{box-sizing:border-box}body{margin:0;background:#f4f6f5;color:var(--ink);font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1040px;margin:auto;padding:28px 22px 60px}a{color:var(--accent);text-underline-offset:3px}
nav{display:flex;gap:18px;flex-wrap:wrap;border-bottom:1px solid #ccd7d7;padding-bottom:14px;margin-bottom:30px}
h1{font-size:clamp(27px,5vw,40px);line-height:1.25}h2{font-size:24px;margin-top:34px}h3{font-size:19px}
.eyebrow{font-size:13px;letter-spacing:.08em;color:var(--accent);font-weight:700}.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px;margin:24px 0}
.card,.panel{background:white;border:1px solid #d5dfdf;border-radius:14px;padding:22px}.card h2,.card h3{margin-top:0}
audio{display:block;width:100%;margin-top:16px}.tag{font-size:12px;display:inline-block;border-radius:5px;background:#e7f0ef;padding:3px 8px}
.button{display:inline-block;padding:9px 15px;border:1px solid #175f69;border-radius:8px;background:#175f69;color:white;text-decoration:none;margin:6px 8px 6px 0}
.secondary{background:white;color:var(--accent)}summary{cursor:pointer;font-weight:600}details{margin-top:25px}
table{display:block;overflow:auto;border-collapse:collapse;width:100%;font-size:14px;margin:22px 0}th,td{text-align:left;border:1px solid #cbd5d5;padding:9px 12px}
th{background:#e7efee}pre{overflow:auto;padding:16px;background:#e8eeed;border-radius:9px;font-size:13px}code{font-family:Consolas,monospace;font-size:.9em}
img{max-width:100%;height:auto}blockquote{border-left:3px solid #92b6b4;padding-left:16px;color:var(--muted)}footer{margin-top:44px;border-top:1px solid #ccd7d7;padding-top:16px;font-size:13px;color:var(--muted)}
'''


def frame(title, content, source, release):
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><main><nav><a href="index.html">试听首页</a><a href="report.html">声音实验报告</a><a href="tutorial.html">操作教程</a><a href="lm-report.html">语言模型报告</a><a href="{html.escape(source)}">GitHub 项目</a></nav>{content}
<footer>2026-10-03 · 实测记录 · <a href="{html.escape(release)}">模型与完整结果下载</a><br>网页播放现有 WAV；新音频转换需要运行模型。</footer></main><script>
document.addEventListener('play', function(event) {{
  if (!(event.target instanceof HTMLAudioElement)) return;
  document.querySelectorAll('audio').forEach(function(player) {{
    if (player !== event.target) player.pause();
  }});
}}, true);
</script></body></html>'''


def make_site(site, source, release, asset_base):
    site.mkdir(parents=True, exist_ok=True)
    for sub in ('listening', 'blind'):
        for path in (RESULT / sub).glob('*.wav'):
            dest = site / sub / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    for name in ('blind_key.json', 'listening_scores.csv', 'training_curves.png', 'comparison.json'):
        shutil.copy2(RESULT / name, site / name)
    audio_card = lambda title, path, caption: f'<section class="card"><span class="tag">{caption}</span><h2>{title}</h2><audio controls preload="metadata" src="{path}">请下载 WAV 播放。</audio><a href="{path}" download>下载 WAV</a></section>'
    content = f'''<p class="eyebrow">AI FUNDAMENTAL / VOICE LAB</p><h1>听见训练的结果</h1><p class="muted">同一套本人清唱录音，两种初始化方式，各完成 1000 次真实更新。先听匿名输出，再看模型身份。</p>
<div class="grid">{audio_card('固定测试原声','listening/input.wav','20 秒 · 未参与训练')}{audio_card('匿名 X','blind/X.wav','19.98 秒 · 近似响度匹配')}{audio_card('匿名 Y','blind/Y.wav','19.98 秒 · 近似响度匹配')}</div>
<section class="panel"><h2>怎样体验</h2><ol><li>点三角形播放原声、X、Y。用耳机，保持同一音量。</li><li>比较自然度、歌词清晰度、像不像本人、音高稳定性，各记 1–5 分。</li><li>保存评分后，展开下面的身份与原始输出。</li></ol><a class="button" href="listening_scores.csv" download>下载盲评表</a><a class="button secondary" href="report.html">阅读实测报告</a></section>
<details class="panel"><summary>揭晓身份与 A/B 原始输出</summary><p><a href="blind_key.json">查看 X/Y 对应关系</a>。匿名输出做了近似 RMS 匹配；以下保留原始输出响度。</p>
<div class="grid">{audio_card('A：随机初始化 G/D','listening/scratch.wav','转换生成器与判别器从随机权重开始')}{audio_card('B：预训练 G/D 微调','listening/finetune.wav','相同架构，加载固定官方权重')}</div>
<a class="button secondary" href="{asset_base}/rvc_A_scratch_step1000.pth">下载 A 模型</a><a class="button secondary" href="{asset_base}/rvc_B_finetune_step1000.pth">下载 B 模型</a><p class="muted">.pth 是模型，不能直接播放。完整结果包包含快照、声音、原始训练日志和教程。</p></details>
<h2>这次实际测到了什么</h2><table><thead><tr><th>指标</th><th>A：随机 G/D</th><th>B：微调 G/D</th></tr></thead><tbody><tr><td>真实更新数</td><td>1000</td><td>1000</td></tr><tr><td>训练耗时（含初始化与保存）</td><td>320.70 秒</td><td>313.83 秒</td></tr><tr><td>此片段估计音高误差中位数</td><td>3050 cent</td><td>0 cent</td></tr><tr><td>有声 / 无声不一致率</td><td>50.38%</td><td>8.48%</td></tr></tbody></table>
<p>本片段 B 的估计音高保留更好；尚未收集盲听评分，没有自然度或音色质量胜者。0 cent 是离散音高估计的中位数，不代表真实误差为零。</p><p class="muted">两组共享冻结的预训练 HuBERT / RMVPE，因此 A 只称为转换 G/D 随机初始化。约六分钟、一位歌者、一个种子、一个固定测试片段的重建实验，不能验证跨歌者能力。</p>
<img src="training_curves.png" alt="两组实际记录的六项训练损失随更新步数变化"><p class="muted">GAN loss 是训练目标，不是听感评分。</p>
<h2>在另一台电脑继续学习</h2><a class="button" href="tutorial.html">术语与每步操作</a><a class="button secondary" href="{release}">下载项目与模型</a><a class="button secondary" href="{source}">查看全部源码</a>'''
    audit = ROOT / 'voice_lab/runs/quality_audit_20261003/spectrogram_envelope.png'
    if audit.is_file():
        shutil.copy2(audit, site / 'quality_audit.png')
        notice = '<section class="panel"><h2>2026-10-03 音质反馈</h2><p>用户试听反馈：X 有滋滋噪声，Y 有多声部感；原结果尚未通过音质验收。审计未发现包装器叠轨，匿名 X 增益约 +9.32 dB。停顿残留、F0 清浊音条件和快照仍需分别排查。</p><a class="button" href="next-voice/">新的本人重建与四档插值小样</a><p>新参考为 Linda Johnson，不是 Adele；轻唱与插值音质待试听验收。</p><details><summary>频谱与停顿诊断</summary><img src="quality_audit.png" alt="原声与两组模型输出的同尺度频谱和响度包络"></details></section>'
        content = notice + content
    demo_template = ROOT / 'voice_lab/voice_workbench/static_demo/index.html'
    if demo_template.is_file():
        demo = site / 'next-voice'
        (demo / 'audio').mkdir(parents=True, exist_ok=True)
        shutil.copy2(demo_template, demo / 'index.html')
        for name in ('self_singing_input_8s', 'model_self_reconstruction',
                     'experimental_alpha_025', 'experimental_alpha_050',
                     'experimental_alpha_075', 'experimental_alpha_100'):
            candidates = (ROOT / f'voice_lab/voice_workbench/benchmarks/{name}.wav',
                          ROOT / f'website/next-voice/audio/{name}.wav')
            original = next((p for p in candidates if p.is_file()), None)
            if original is None:
                raise FileNotFoundError('Verified demo audio missing: ' + name)
            shutil.copy2(original, demo / 'audio' / (name + '.wav'))
    adele_demo = ROOT / 'voice_lab/voice_workbench/static_adele_demo'
    if (adele_demo / 'index.html').is_file():
        # Only the portable page, sanitized evidence and explicitly published
        # user/generated audio may enter the site; no reference interviews.
        names = ('source', 'vogue_singing_alpha050', 'vogue_singing_alpha100',
                 'vogue_speech_alpha100', 'npr_speech_alpha050', 'npr_speech_alpha100')
        for relative in ('index.html', 'summary.json', 'mobile-recorder.html') + tuple('listening/' + name + '.wav' for name in names):
            original = adele_demo / relative
            destination = site / 'adele-exploration' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
    (site / 'index.html').write_text(frame('AI Fundamental · 本人清唱试听', content, source, release), encoding='utf-8')
    import markdown
    for source_md, target, title in ((RESULT / 'MEASURED_REPORT.md', 'report.html', '清唱实测报告'),
                                     (ROOT / 'voice_lab/RVC_STEP_BY_STEP.md', 'tutorial.html', '清唱模型逐步教程'),
                                     (ROOT / 'report/FINAL_REPORT.md', 'lm-report.html', '0.1B 语言模型实测报告'),
                                     (ROOT / 'BEGINNER_TUTORIAL.md', 'lm-tutorial.html', '语言模型从零学习教程')):
        text = source_md.read_text(encoding='utf-8')
        # Resolve document references in the portable website.
        text = text.replace('(RVC_STEP_BY_STEP.md)', '(tutorial.html)').replace('(MEASURED_REPORT.md)', '(report.html)')
        text = text.replace('(RVC_EXPERIMENT_RECORD.md)', '(' + source + '/voice_lab/RVC_EXPERIMENT_RECORD.md)')
        text = text.replace('(../BEGINNER_TUTORIAL.md)', '(lm-tutorial.html)')
        text = text.replace('(neurips_style.md)', '(' + source + '/report/neurips_style.md)')
        text = text.replace('(README.md#环境)', '(' + source + '/README.md#环境)')
        text = text.replace('(report/figures/', '(figures/').replace('(report/data/', '(data/')
        text = text.replace('(report/neurips_style.md)', '(' + source + '/report/neurips_style.md)')
        text = re.sub(r'\[([^\]]+)\]\(evidence/[^)]+\)', r'\1（本机归档，未公开截图）', text)
        content = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc'])
        (site / target).write_text(frame(title, '<article>' + content + '</article>', source, release), encoding='utf-8')
    for path in (ROOT / 'report/figures').glob('*'):
        dest = site / 'figures' / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    for path in (ROOT / 'report/data').rglob('*'):
        if eligible(path):
            dest = site / 'data' / path.relative_to(ROOT / 'report/data')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    (site / '.nojekyll').touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repo-url', default='https://github.com/DJDeborah/ai-fundamental')
    parser.add_argument('--project-path', default='ai-fundamental')
    parser.add_argument('--branch', default='main')
    parser.add_argument('--release-tag', default='ai-fundamental-2026-10-03')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_url = args.repo_url.rstrip('/') + '/tree/' + args.branch + '/' + args.project_path.strip('/')
    release_url = args.repo_url.rstrip('/') + '/releases/tag/' + args.release_tag
    asset_base = args.repo_url.rstrip('/') + '/releases/download/' + args.release_tag
    copied = []
    for path in collect_source():
        dest = out / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        copied.append(path.relative_to(ROOT).as_posix())
    make_site(out / 'website', source_url, release_url, asset_base)
    (out / 'DOWNLOADS.md').write_text(f'''# 项目与模型下载

- [GitHub 项目]({source_url})：课程源码、固定上游源码、教程、数据划分、实测报告和音频。
- [Release 下载]({release_url})：两套最终模型、完整本机结果包、可复现项目包。
- `rvc_A_scratch_step1000.pth`：A 的最终转换模型。
- `rvc_B_finetune_step1000.pth`：B 的最终转换模型。
- `voice_self_rvc_results_20261003.zip`：模型与 0/500/1000 快照、WAV、原始训练指标和教程。
- `ai_fundamental_project_20261003.zip`：便携源码、训练录音、报告及本机玩具模型。

试听页面播放已生成的 WAV，不需要 GPU。新音频转换按 `voice_lab/RVC_STEP_BY_STEP.md` 操作。
大规模语言模型的云端权重当前没有完整下载到本机；这里提供其源码、训练指标和报告，不冒充已归档的 0.1B checkpoint。

结果 ZIP 是经过 CRC / 模型与声音 SHA256 验证的**本机重新打包交付**。原云端 ZIP 的末段及部分失败历史仍待无卡模式补取。预算记录是保护估计，不是平台账单。
第三方源码保留各自 LICENSE；录音、模型、数据集不自动继承第三方源码许可。
''', encoding='utf-8')
    readme = out / 'README.md'
    text = readme.read_text(encoding='utf-8')
    first, _, rest = text.partition('\n')
    parts = args.repo_url.rstrip('/').split('/')
    pages_url = f'https://{parts[-2]}.github.io/{parts[-1]}/' if len(parts) >= 5 else 'website/index.html'
    text = first + '\n\n**新增：本人清唱 A/B 实验已完成。** [在线试听](' + pages_url + ') · [声音实测报告](voice_lab/runs/new4090d_ab_retry1/MEASURED_REPORT.md) · [术语与操作教程](voice_lab/RVC_STEP_BY_STEP.md) · [模型和完整项目下载](DOWNLOADS.md)\n\n' + rest
    readme.write_text(text, encoding='utf-8')
    # Runtime/generated files stay excluded in a developer clone. Existing tracked evidence stays in Git.
    (out / '.gitignore').write_text('''.venv/
.rvc-venv/
.voice-venv/
.cloud-access/
__pycache__/
.pytest_cache/
*.pyc
*.pt
*.pth
*.bin
*.npy
*.partial
dist/
tmp/
**/upstream/*/.git/
**/upstream/*/logs/
''', encoding='utf-8')
    scan_secrets(out)
    hashes = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*'))
              if p.is_file() and p.name != 'PUBLICATION_MANIFEST.json'}
    write_json(out / 'PUBLICATION_MANIFEST.json', {
        'files': hashes, 'files_count': len(hashes), 'source_url': source_url,
        'release_url': release_url, 'credential_files_included': False,
        'large_models_in_git': False, 'audio_publication_authorized': True,
        'upstream_fixed_source_hash_verified': True,
        'original_cloud_zip_fully_downloaded': False,
        'omitted': ['virtual environments', 'credential files', 'wheel caches', 'incomplete downloads',
                    'browser/session screenshots', 'large downloaded third-party weights',
                    'raw FineWeb text (source download scripts and provenance included)']})
    print(json.dumps({'out': str(out), 'files': len(hashes),
                      'bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
                      'secret_scan': 'passed'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
