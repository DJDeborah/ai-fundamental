"""SFT, DPO, and small on-policy RLVR branches from a common base checkpoint."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import load_checkpoint, save_checkpoint
from .tokenizer import ByteBPE, EOS


def make_data(out: Path, variant: str = "pilot"):
    if variant not in {"pilot", "research"}:
        raise ValueError("variant must be pilot or research")
    out.mkdir(parents=True, exist_ok=True)
    names = ("sft", "dpo", "rlvr", "eval", "eval_ood") if variant == "research" else ("sft", "dpo", "rlvr", "eval")
    files = {name: (out / f"{name}.jsonl").open("w", encoding="utf-8") for name in names}
    counts = {name: 0 for name in names}
    try:
        pairs = [(x, y) for x in range(60 if variant == "research" else 30)
                 for y in range(20 if variant == "research" else 10)]
        if variant == "research":
            random.Random(42).shuffle(pairs)
        for x, y in pairs:
            prompt = f"Compute {x} + {y}. Answer: "
            answer = str(x + y)
            if variant == "research":
                bucket = int(hashlib.sha256(f"{x}:{y}".encode()).hexdigest()[:8], 16) % 10
                split = "eval_ood" if x >= 50 else "eval" if bucket == 0 else None
            else:
                split = "eval" if x >= 25 else None
            if split:
                files[split].write(json.dumps({"prompt": prompt, "answer": answer}) + "\n")
                counts[split] += 1
                continue
            files["sft"].write(json.dumps({"prompt": prompt, "completion": answer}) + "\n")
            files["dpo"].write(json.dumps({"prompt": prompt, "chosen": answer, "rejected": str(x+y+1)}) + "\n")
            files["rlvr"].write(json.dumps({"prompt": prompt, "answer": answer}) + "\n")
            for name in ("sft", "dpo", "rlvr"):
                counts[name] += 1
    finally:
        for f in files.values():
            f.close()
    manifest = {"variant": variant, "counts": counts,
                "train_range": "x=0..49,y=0..19" if variant == "research" else "x=0..24,y=0..9",
                "id_eval_split": "sha256('x:y') mod 10 == 0" if variant == "research" else "x=25..29",
                "ood_eval_range": "x=50..59,y=0..19" if variant == "research" else None,
                "train_order_seed": 42 if variant == "research" else None}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


def read_rows(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def token_sequence(tok, prompt, completion, max_len, device):
    p = tok.encode(prompt)
    c = tok.encode(completion) + [EOS]
    if not p or len(p) + len(c) > max_len + 1:
        raise ValueError("empty prompt or sequence too long for checkpoint context")
    ids = torch.tensor([p + c], dtype=torch.long, device=device)
    return ids, len(p)


def completion_stats(model, tok, prompt, completion, device, reference=None):
    ids, n_prompt = token_sequence(tok, prompt, completion, model.cfg.max_seq_len, device)
    logits = model(ids[:, :-1]).float()[:, n_prompt-1:, :]
    target = ids[:, n_prompt:]
    logp = F.log_softmax(logits, dim=-1)
    selected = logp.gather(-1, target[..., None]).squeeze(-1)
    if reference is None:
        return selected.sum(), selected.numel(), None
    with torch.no_grad():
        ref_logits = reference(ids[:, :-1]).float()[:, n_prompt-1:, :]
        ref_logp = F.log_softmax(ref_logits, dim=-1)
    probs = logp.exp()
    kl = (probs * (logp - ref_logp)).sum(-1).mean()
    return selected.sum(), selected.numel(), kl


def sampled_token_stats(model, prompt_ids, sampled_ids, device, reference=None):
    """Log probability of the exact sampled policy actions, including EOS only if drawn.

    Decoding and re-encoding generated text can change its tokenization. An EOS
    token must not be inserted when generation merely hit the length limit.
    """
    if not prompt_ids or not sampled_ids:
        raise ValueError("sampled trajectory needs prompt and generated token IDs")
    if len(prompt_ids) + len(sampled_ids) - 1 > model.cfg.max_seq_len:
        raise ValueError("sampled trajectory exceeds checkpoint context")
    inputs = torch.tensor([prompt_ids + sampled_ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([sampled_ids], dtype=torch.long, device=device)
    logits = model(inputs).float()[:, len(prompt_ids)-1:, :]
    logp = F.log_softmax(logits, dim=-1)
    selected = logp.gather(-1, targets[..., None]).squeeze(-1)
    if reference is None:
        return selected.sum(), selected.numel(), None
    with torch.no_grad():
        ref_logits = reference(inputs).float()[:, len(prompt_ids)-1:, :]
        ref_logp = F.log_softmax(ref_logits, dim=-1)
    probs = logp.exp()
    kl = (probs * (logp - ref_logp)).sum(-1).mean()
    return selected.sum(), selected.numel(), kl


def verified_answer(text: str, answer: str) -> float:
    # Full-string check: no reward for extra prose or a substring hit.
    return float(text.strip() == answer)


@torch.inference_mode()
def sample_completion(model, tok, prompt, device, max_new=8, temperature=1.0,
                      return_tokens=False):
    ids = tok.encode(prompt)
    generated = []
    for _ in range(max_new):
        if len(ids) >= model.cfg.max_seq_len:
            break
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float() / temperature
        next_id = int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())
        generated.append(next_id)
        if next_id == EOS:
            break
        ids.append(next_id)
    text = tok.decode(generated[:-1] if generated and generated[-1] == EOS else generated)
    return (text, generated) if return_tokens else text


def train_branch(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, parent_metadata = load_checkpoint(args.base, device)
    tok = ByteBPE.load(args.tokenizer)
    if tok.size != model.cfg.vocab_size:
        raise ValueError("tokenizer/model vocab mismatch")
    rows = read_rows(args.data)
    order = list(range(len(rows)))
    data_rng = random.Random(args.seed)
    data_rng.shuffle(order)
    reference = None
    if args.command in ("dpo", "rlvr"):
        reference, _ = load_checkpoint(args.base, device)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    sampled_tokens = 0
    skipped = 0
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            if step > 1 and (step - 1) % len(rows) == 0:
                data_rng.shuffle(order)
            row = rows[order[(step - 1) % len(rows)]]
            model.train()
            opt.zero_grad(set_to_none=True)
            extra = {}
            if args.command == "sft":
                lp, n, _ = completion_stats(model, tok, row["prompt"], row["completion"], device)
                loss = -lp / n
            elif args.command == "dpo":
                win, _, _ = completion_stats(model, tok, row["prompt"], row["chosen"], device)
                lose, _, _ = completion_stats(model, tok, row["prompt"], row["rejected"], device)
                with torch.no_grad():
                    ref_win, _, _ = completion_stats(reference, tok, row["prompt"], row["chosen"], device)
                    ref_lose, _, _ = completion_stats(reference, tok, row["prompt"], row["rejected"], device)
                margin = (win - lose) - (ref_win - ref_lose)
                loss = F.softplus(-args.beta * margin)
                extra = {"margin": margin.item()}
            else:
                model.eval()
                samples = [sample_completion(model, tok, row["prompt"], device, args.max_new_tokens,
                                             return_tokens=True) for _ in range(args.group_size)]
                sampled_tokens += sum(len(ids) for _, ids in samples)
                rewards = torch.tensor([verified_answer(text, row["answer"]) for text, _ in samples], device=device)
                model.train()
                extra = {"reward_mean": rewards.mean().item(), "reward_variance": rewards.var(unbiased=False).item()}
                if rewards.max() == rewards.min():
                    skipped += 1
                    rec = {"step": step, "loss": None, "skipped_zero_variance": True,
                           "sampled_tokens_cumulative": sampled_tokens, "elapsed_s": time.perf_counter() - started, **extra}
                    log.write(json.dumps(rec) + "\n")
                    continue
                advantage = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
                terms = []
                prompt_ids = tok.encode(row["prompt"])
                for (_, sampled_ids), adv in zip(samples, advantage):
                    lp, n, kl = sampled_token_stats(model, prompt_ids, sampled_ids, device, reference)
                    terms.append(-adv * lp / n + args.kl_beta * kl)
                loss = torch.stack(terms).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            rec = {"step": step, "loss": loss.item(), "sampled_tokens_cumulative": sampled_tokens,
                   "elapsed_s": time.perf_counter() - started, **extra}
            log.write(json.dumps(rec) + "\n")
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(json.dumps(rec), flush=True)
    digest = hashlib.sha256(args.base.read_bytes()).hexdigest()
    root_digest = parent_metadata.get("root_base_sha256", parent_metadata.get("base_sha256", digest))
    save_checkpoint(args.out, model, stage=args.command, parent_sha256=digest,
                    root_base_sha256=root_digest, steps=args.steps, skipped_zero_variance=skipped,
                    sampled_tokens=sampled_tokens, elapsed_s=time.perf_counter() - started, seed=args.seed)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("make-data")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--variant", choices=["pilot", "research"], default="pilot")
    for name in ("sft", "dpo", "rlvr"):
        q = sub.add_parser(name)
        q.add_argument("--base", type=Path, required=True)
        q.add_argument("--data", type=Path, required=True)
        q.add_argument("--tokenizer", type=Path, required=True)
        q.add_argument("--out", type=Path, required=True)
        q.add_argument("--steps", type=int, default=100)
        q.add_argument("--lr", type=float, default=1e-5)
        q.add_argument("--beta", type=float, default=0.1)
        q.add_argument("--kl-beta", type=float, default=0.01)
        q.add_argument("--group-size", type=int, default=4)
        q.add_argument("--max-new-tokens", type=int, default=8)
        q.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.command == "make-data":
        make_data(args.out, args.variant)
    else:
        train_branch(args)


if __name__ == "__main__":
    main()
