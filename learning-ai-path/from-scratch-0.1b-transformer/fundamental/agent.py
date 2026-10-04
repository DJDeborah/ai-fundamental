"""Oracle imitation, online verified policy gradient, and long-horizon harness."""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from .env import ArithmeticChainEnv
from .model import load_checkpoint, save_checkpoint
from .posttrain import completion_stats, sample_completion, sampled_token_stats
from .tokenizer import ByteBPE


def collect(out: Path, episodes: int, horizon: int):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for seed in range(episodes):
            env = ArithmeticChainEnv(seed, horizon)
            while not env.done:
                prompt, action = env.observation(), env.oracle_action()
                f.write(json.dumps({"prompt": prompt, "completion": action, "seed": seed, "step": env.index}) + "\n")
                assert env.step(action)["valid"]


def load_model(path, tokenizer, device):
    model, _ = load_checkpoint(path, device)
    tok = ByteBPE.load(tokenizer)
    if tok.size != model.cfg.vocab_size:
        raise ValueError("tokenizer/model vocab mismatch")
    return model, tok


def imitate(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.base, args.tokenizer, device)
    rows = [json.loads(s) for s in args.data.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("empty trajectory file")
    order = list(range(len(rows)))
    data_rng = random.Random(args.seed)
    data_rng.shuffle(order)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        if step > 0 and step % len(rows) == 0:
            data_rng.shuffle(order)
        row = rows[order[step % len(rows)]]
        opt.zero_grad(set_to_none=True)
        lp, n, _ = completion_stats(model, tok, row["prompt"], row["completion"], device)
        loss = -lp / n
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == args.steps:
            print(json.dumps({"step": step+1, "imitation_loss": loss.item()}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.out, model, stage="agent_imitation", steps=args.steps, seed=args.seed)


@torch.no_grad()
def self_judge_score(model, tok, observation: str, action: str, device):
    # The model's own judgement is a ranking signal, never environment truth.
    # Keep the judge query short enough for the tiny teaching model's context.
    state = observation.split(" Recent actions:")[0]
    judge_prompt = f"{state} Candidate: {action}. Valid? "
    yes, _, _ = completion_stats(model, tok, judge_prompt, "YES", device)
    no, _, _ = completion_stats(model, tok, judge_prompt, "NO", device)
    return (yes - no).item()


def choose_action(model, tok, observation, device, self_judge, candidates, max_new):
    if candidates < 1:
        raise ValueError("candidates must be positive")
    started = time.perf_counter()
    with torch.no_grad():
        proposed = [sample_completion(model, tok, observation, device, max_new,
                                      return_tokens=True) for _ in range(candidates)]
        details = {"candidate_count": len(proposed),
                   "sampled_tokens": sum(len(ids) for _, ids in proposed)}
        if not self_judge:
            details["sampled_token_ids"] = proposed[0][1]
            details["decision_wall_s"] = time.perf_counter() - started
            return proposed[0][0], details
        scores = [self_judge_score(model, tok, observation, action, device) for action, _ in proposed]
        best = max(range(len(scores)), key=lambda i: scores[i])
        details.update({"candidates": [action for action, _ in proposed],
                        "sampled_token_ids": proposed[best][1], "judge_scores": scores,
                        "decision_wall_s": time.perf_counter() - started})
        return proposed[best][0], details


def rollout(model, tok, seed, horizon, device, self_judge=False, candidates=3, max_new=12):
    env = ArithmeticChainEnv(seed, horizon)
    trajectory = []
    while not env.done:
        observation = env.observation()
        action, judge = choose_action(model, tok, observation, device, self_judge, candidates, max_new)
        result = env.step(action)
        trajectory.append({"observation": observation, "action": action, "oracle_action": f"ADD {env.deltas[len(trajectory)]}",
                           "judge": judge, **result})
    return trajectory, env.success


def online_rl(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.base, args.tokenizer, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    baseline = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".jsonl").open("w", encoding="utf-8") as log:
        for episode in range(args.episodes):
            model.eval()
            trajectory, success = rollout(model, tok, args.seed + episode, args.horizon, device,
                                          candidates=1, max_new=args.max_new_tokens)
            reward = sum(t["reward"] for t in trajectory)
            model.train()
            opt.zero_grad(set_to_none=True)
            terms = []
            for step in trajectory:
                lp, n, _ = sampled_token_stats(model, tok.encode(step["observation"]),
                                               step["judge"]["sampled_token_ids"], device)
                terms.append(lp / n)
            advantage = reward - baseline
            loss = -advantage * torch.stack(terms).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            baseline = 0.9 * baseline + 0.1 * reward
            rec = {"episode": episode, "reward": reward, "success": success, "steps": len(trajectory), "loss": loss.item()}
            log.write(json.dumps(rec) + "\n")
            if episode == 0 or (episode + 1) % 10 == 0 or episode + 1 == args.episodes:
                print(json.dumps(rec))
    save_checkpoint(args.out, model, stage="agent_online_verified_rl", episodes=args.episodes, seed=args.seed)


def evaluate(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.base, args.tokenizer, device)
    model.eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wins = 0
    total_sampled_tokens = 0
    total_decision_wall_s = 0.0
    total_candidates = 0
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.episodes):
            seed = args.seed + i
            torch.manual_seed(seed)  # Pair the first candidate across judge on/off conditions.
            trajectory, success = rollout(model, tok, seed, args.horizon, device, args.self_judge, args.candidates, args.max_new_tokens)
            wins += int(success)
            total_sampled_tokens += sum(step["judge"]["sampled_tokens"] for step in trajectory)
            total_candidates += sum(step["judge"]["candidate_count"] for step in trajectory)
            total_decision_wall_s += sum(step["judge"]["decision_wall_s"] for step in trajectory)
            f.write(json.dumps({"seed": seed, "success": success, "steps": len(trajectory), "trajectory": trajectory}) + "\n")
    print(json.dumps({"episodes": args.episodes, "horizon": args.horizon, "successes": wins,
                      "success_rate": wins / args.episodes, "self_judge": args.self_judge,
                      "sampled_tokens": total_sampled_tokens, "candidate_count": total_candidates,
                      "decision_wall_s": total_decision_wall_s, "out": str(args.out)}))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--episodes", type=int, default=1000)
    c.add_argument("--horizon", type=int, default=8)
    for name in ("train", "rl", "eval"):
        q = sub.add_parser(name)
        q.add_argument("--base", type=Path, required=True)
        q.add_argument("--tokenizer", type=Path, required=True)
        q.add_argument("--out", type=Path, required=True)
        q.add_argument("--seed", type=int, default=100000 if name == "eval" else 42)
        if name == "train":
            q.add_argument("--data", type=Path, required=True)
            q.add_argument("--steps", type=int, default=1000)
            q.add_argument("--lr", type=float, default=1e-5)
        else:
            q.add_argument("--episodes", type=int, default=100)
            q.add_argument("--horizon", type=int, default=8)
            q.add_argument("--max-new-tokens", type=int, default=12)
            if name == "rl":
                q.add_argument("--lr", type=float, default=1e-6)
            else:
                q.add_argument("--self-judge", action="store_true")
                q.add_argument("--candidates", type=int, default=3)
    args = p.parse_args()
    if args.command == "collect":
        collect(args.out, args.episodes, args.horizon)
    elif args.command == "train":
        imitate(args)
    elif args.command == "rl":
        online_rl(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
