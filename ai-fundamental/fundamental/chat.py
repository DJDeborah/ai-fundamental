"""Tiny terminal interface for inspecting a checkpoint's generated text."""
import argparse
from pathlib import Path

import torch

from .model import load_checkpoint
from .tokenizer import ByteBPE, EOS


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens=80, temperature=0.0, top_k=20):
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt cannot be empty")
    output = []
    for _ in range(max_new_tokens):
        if len(ids) >= model.cfg.max_seq_len:
            break
        x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
        logits = model(x)[0, -1].float()
        if temperature <= 0:
            next_id = int(logits.argmax().item())
        else:
            values, indices = torch.topk(logits / temperature, min(top_k, logits.numel()))
            next_id = int(indices[torch.multinomial(torch.softmax(values, -1), 1)].item())
        if next_id == EOS:
            break
        ids.append(next_id)
        output.append(next_id)
    return tokenizer.decode(output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--once", help="Ask once; omit for interactive terminal loop")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = deterministic greedy decoding")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--history", action="store_true", help="Keep prior turns until the short context fills")
    p.add_argument("--raw", action="store_true", help="Use input verbatim as the prompt; useful for base LM completions and task-specific SFT")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(a.checkpoint, device)
    model.eval()
    tok = ByteBPE.load(a.tokenizer)
    if tok.size != model.cfg.vocab_size:
        raise ValueError("checkpoint and tokenizer vocabulary sizes differ")
    history = ""

    def answer(message):
        nonlocal history
        turn = message if a.raw else f"User: {message}\nAssistant: "
        prompt = (history if a.history else "") + turn
        # Keep complete current turn; shorten earlier context if necessary.
        if len(tok.encode(prompt)) >= model.cfg.max_seq_len:
            history = ""
            prompt = turn
        if len(tok.encode(prompt)) >= model.cfg.max_seq_len:
            raise ValueError("question is too long for this model's context window")
        response = generate(model, tok, prompt, a.max_new_tokens, a.temperature, a.top_k)
        if a.history:
            history = prompt + response + "\n"
        return response

    if a.once is not None:
        print(answer(a.once))
        return
    print("Model terminal. Type /exit to quit. Use --raw for verbatim prompts. Research checkpoints may not follow chat instructions.")
    while True:
        try:
            message = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message == "/exit":
            break
        if not message:
            continue
        print("Model>", answer(message))


if __name__ == "__main__":
    main()
