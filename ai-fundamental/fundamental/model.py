"""Decoder-only Transformer: tied embeddings, RoPE, RMSNorm, SwiGLU, SDPA."""
import argparse
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    dim: int
    layers: int
    heads: int
    hidden: int
    max_seq_len: int


def config_for(name: str, vocab_size: int) -> GPTConfig:
    if name == "tiny":
        return GPTConfig(vocab_size, 128, 4, 4, 512, 128)
    if name == "small":
        return GPTConfig(vocab_size, 256, 6, 8, 1024, 1024)
    if name == "medium":
        return GPTConfig(vocab_size, 512, 8, 8, 2048, 1024)
    if name == "base100m":
        return GPTConfig(vocab_size, 768, 10, 12, 3072, 1024)
    raise ValueError(name)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def apply_rope(x):
    # x: (batch, heads, time, head_dim); rotate adjacent pairs.
    _, _, time, width = x.shape
    freq = 1.0 / (10000 ** (torch.arange(0, width, 2, device=x.device, dtype=torch.float32) / width))
    phase = torch.outer(torch.arange(time, device=x.device, dtype=torch.float32), freq)
    cos, sin = phase.cos()[None, None], phase.sin()[None, None]
    a, b = x.float()[..., 0::2], x.float()[..., 1::2]
    return torch.stack((a*cos-b*sin, a*sin+b*cos), dim=-1).flatten(-2).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.heads = cfg.heads
        self.dim = cfg.dim
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x):
        batch, time, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=-1)
        reshape = lambda z: z.view(batch, time, self.heads, width // self.heads).transpose(1, 2)
        q, k, v = apply_rope(reshape(q)), apply_rope(reshape(k)), reshape(v)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(batch, time, width))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.gate = nn.Linear(cfg.dim, cfg.hidden, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.hidden, bias=False)
        self.down = nn.Linear(cfg.hidden, cfg.dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        y = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.dim % cfg.heads == 0 and (cfg.dim // cfg.heads) % 2 == 0
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.final_norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids, targets=None):
        if ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def save_checkpoint(path, model: GPT, **extra):
    torch.save({"config": asdict(model.cfg), "model": model.state_dict(), **extra}, path)


def load_checkpoint(path, device="cpu"):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**obj["config"]))
    model.load_state_dict(obj["model"])
    return model.to(device), obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=["tiny", "small", "medium", "base100m"], default="tiny")
    p.add_argument("--vocab-size", type=int, default=512)
    args = p.parse_args()
    cfg = config_for(args.config, args.vocab_size)
    model = GPT(cfg)
    print(f"{model.count_parameters():,} parameters; config={asdict(cfg)}")
    if args.config == "tiny":
        ids = torch.randint(cfg.vocab_size, (2, 16))
        _, loss = model(ids[:, :-1], ids[:, 1:])
        loss.backward()
        print(f"forward/backward OK; loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
