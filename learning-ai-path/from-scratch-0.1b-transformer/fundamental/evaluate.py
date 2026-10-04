"""Held-out exact-match evaluation for the post-training branches."""
import argparse
import json
from pathlib import Path

import torch

from .model import load_checkpoint
from .posttrain import read_rows, verified_answer
from .tokenizer import ByteBPE, EOS


@torch.inference_mode()
def greedy_answer(model, tok, prompt, device, max_new):
    ids = tok.encode(prompt)
    output = []
    for _ in range(max_new):
        if len(ids) >= model.cfg.max_seq_len:
            break
        x = torch.tensor([ids], dtype=torch.long, device=device)
        token = int(model(x)[0, -1].argmax().item())
        if token == EOS:
            break
        output.append(token)
        ids.append(token)
    return tok.decode(output)


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    radius = z * ((p*(1-p)/n + z*z/(4*n*n))**0.5) / denom
    return [center-radius, center+radius]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = ByteBPE.load(a.tokenizer)
    rows = read_rows(a.data)
    results = []
    for checkpoint in a.checkpoints:
        model, metadata = load_checkpoint(checkpoint, device)
        model.eval()
        if model.cfg.vocab_size != tok.size:
            raise ValueError(f"vocab mismatch: {checkpoint}")
        cases = []
        for row in rows:
            output = greedy_answer(model, tok, row["prompt"], device, a.max_new_tokens)
            cases.append({"prompt": row["prompt"], "expected": row["answer"], "output": output,
                          "correct": bool(verified_answer(output, row["answer"]))})
        wins = sum(c["correct"] for c in cases)
        results.append({"checkpoint": str(checkpoint), "stage": metadata.get("stage", "base"),
                        "n": len(cases), "correct": wins, "accuracy": wins/len(cases),
                        "wilson95": wilson(wins, len(cases)), "cases": cases})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps([{k:v for k,v in x.items() if k != "cases"} for x in results], ensure_ascii=False))


if __name__ == "__main__":
    main()
