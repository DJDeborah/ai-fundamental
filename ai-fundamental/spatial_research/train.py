"""Teaching reproduction: ImageFolder source-only or Deep CORAL, fixed-step budget.

Not the historical paper's exact architecture/protocol. Target labels are never
used for adaptation loss. Target folders supplied here must exclude final test.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


def coral(source, target):
    source, target = source.float(), target.float()
    source = source - source.mean(0)
    target = target - target.mean(0)
    cs = source.T @ source / (len(source) - 1)
    ct = target.T @ target / (len(target) - 1)
    return (cs - ct).square().sum() / (4 * source.shape[1] ** 2)


def sample_dataset(root, transform, per_class, seed):
    ds = datasets.ImageFolder(root, transform=transform)
    rng = random.Random(seed)
    indices = []
    for label in range(len(ds.classes)):
        group = [i for i, (_, y) in enumerate(ds.samples) if y == label]
        rng.shuffle(group)
        indices.extend(group[:per_class] if per_class else group)
    return ds, Subset(ds, indices), indices


def cycle(loader):
    while True:
        yield from loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--target', type=Path)
    p.add_argument('--eval', required=True, type=Path)
    p.add_argument('--method', choices=['source', 'coral'], default='source')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--coral-weight', type=float, default=1.0)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--per-class', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-seed', type=int, default=123)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--augment', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory to preserve results')
    if args.batch_size < 2 or args.steps < 1:
        raise ValueError('batch-size >= 2 and steps >= 1 required')
    if args.method == 'coral' and not args.target:
        raise ValueError('CORAL requires a separate unlabeled target-adapt directory')
    if args.target and args.target.resolve() == args.eval.resolve():
        raise ValueError('Target adaptation and evaluation must be separate directories')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    norm = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    train_ops = [transforms.Resize((args.size, args.size))]
    if args.augment:
        train_ops += [transforms.ColorJitter(.2, .2, .2, .05), transforms.RandomGrayscale(.1)]
    train_tf = transforms.Compose(train_ops + [transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([transforms.Resize((args.size, args.size)), transforms.ToTensor(), norm])
    sd, source, ids = sample_dataset(args.source, train_tf, args.per_class, args.data_seed)
    ed = datasets.ImageFolder(args.eval, transform=eval_tf)
    if sd.class_to_idx != ed.class_to_idx:
        raise ValueError('Source and evaluation class names must match exactly')
    loader_opts = dict(batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=device == 'cuda')
    sl = DataLoader(source, shuffle=True, drop_last=True, **loader_opts)
    if not len(sl):
        raise ValueError('Not enough source images for one batch')
    target_it, target_paths = None, []
    if args.method == 'coral':
        td = datasets.ImageFolder(args.target, transform=train_tf)
        if td.class_to_idx != sd.class_to_idx:
            raise ValueError('Target class folder mapping must match source')
        if {str(Path(x).resolve()) for x, _ in td.samples} & {str(Path(x).resolve()) for x, _ in ed.samples}:
            raise ValueError('Target/eval path overlap')
        tl = DataLoader(td, shuffle=True, drop_last=True, **loader_opts)
        if not len(tl):
            raise ValueError('Not enough target images')
        target_it = cycle(tl)
        target_paths = [str(x) for x, _ in td.samples]
    args.out.mkdir(parents=True)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    dim = model.fc.in_features
    model.fc = nn.Identity()
    head = nn.Linear(dim, len(sd.classes))
    model, head = model.to(device), head.to(device)
    optimizer = torch.optim.SGD(list(model.parameters()) + list(head.parameters()), lr=args.lr, momentum=.9)
    amp = args.amp and device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    manifest = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'torch': torch.__version__, 'device': device,
                'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                'classes': sd.classes, 'source_files': [str(sd.samples[i][0]) for i in ids],
                'eval_files': [str(x) for x, _ in ed.samples], 'target_files': target_paths,
                'weights': 'ResNet18_Weights.IMAGENET1K_V1',
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    source_it = cycle(sl)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    with (args.out / 'metrics.jsonl').open('w') as log:
        for step in range(1, args.steps + 1):
            x, y = next(source_it)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                fs = model(x)
                ce = nn.functional.cross_entropy(head(fs), y)
                align = torch.zeros((), device=device)
                if target_it is not None:
                    xt, _ = next(target_it)  # Ignore target labels.
                    align = coral(fs, model(xt.to(device)))
                loss = ce + args.coral_weight * align
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step == 1 or step % 10 == 0 or step == args.steps:
                row = {'step': step, 'ce': ce.item(), 'coral': align.item(), 'loss': loss.item()}
                log.write(json.dumps(row) + '\n')
                log.flush()
                print(json.dumps(row), flush=True)
    if device == 'cuda':
        torch.cuda.synchronize()
    train_s = time.perf_counter() - start
    model.eval()
    head.eval()
    correct, totals = torch.zeros(len(sd.classes)), torch.zeros(len(sd.classes))
    eval_start = time.perf_counter()
    with torch.inference_mode():
        for x, y in DataLoader(ed, shuffle=False, **loader_opts):
            pred = head(model(x.to(device))).argmax(-1).cpu()
            for cls in range(len(sd.classes)):
                mask = y == cls
                totals[cls] += mask.sum()
                correct[cls] += ((pred == y) & mask).sum()
    if (totals == 0).any():
        raise ValueError('Evaluation must cover every class')
    summary = {'train_s': train_s, 'eval_s': time.perf_counter() - eval_start,
               'source_images_per_s': args.steps * args.batch_size / train_s,
               'accuracy': (correct.sum() / totals.sum()).item(),
               'mean_class_accuracy': (correct / totals).mean().item(),
               'per_class_accuracy': dict(zip(sd.classes, (correct / totals).tolist())),
               'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    torch.save({'backbone': model.state_dict(), 'head': head.state_dict(), 'manifest': manifest}, args.out / 'model.pt')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
