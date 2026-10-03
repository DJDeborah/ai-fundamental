"""Instrument pinned upstream without changing its GAN losses or architecture.

Single GPU only. Step means one batch with both G/D optimizer updates in FP32.
Checkpoints are written at step 0 and fixed steps; SIGINT stops after a batch.
"""
import argparse
import math
import os
import random
import signal
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voicelab.core import append_json, sha256, utc_now, write_json
from voicelab.rvc_assets import RVC


class TrainingFinished(Exception):
    pass


def instrument(source):
    # Fail closed if the upstream's interfaces change.
    changes = {
        "training_dtype = get_training_dtype()": "training_dtype = torch.float32  # fixed for both arms; avoids FP16 skipped optimizer steps",
        "    cache = []\n": "    _lab_init(locals())\n    cache = []\n",
        "        scaler.update()\n": "        scaler.update()\n        _lab_losses(locals())\n",
        "        global_step += 1\n": "        global_step += 1\n        _lab_step(locals(), global_step)\n",
    }
    for old, new in changes.items():
        if source.count(old) != 1:
            raise ValueError(f"Pinned training source changed at {old!r}")
        source = source.replace(old, new)
    compile(source, str(RVC / "train/train.py"), "exec")
    return source


class Recorder:
    def __init__(self, output, steps, seconds, checkpoint_every):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.steps, self.seconds, self.every = steps, seconds, checkpoint_every
        self.started = time.monotonic()
        self.interrupted, self.last_losses = False, {}
        self.scope = {}
        self.last_step = 0

    def save(self, ctx, step, full=False):
        hps, net_g = ctx["hps"], ctx["net_g"]
        name = f"{hps.name}_step{step}"
        model = RVC / f"assets/weights/{name}.pth"
        self.scope["savee"](net_g.state_dict(), hps.sample_rate, hps.if_f0,
                            name, ctx.get("epoch", 0), hps.version, hps)
        if not model.is_file():
            raise RuntimeError(f"Upstream failed to export {model}")
        preserved = self.output / "checkpoints" / f"step{step}.pth"
        preserved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model, preserved)
        if full:
            for label in ("g", "d"):
                self.scope["utils"].save_checkpoint(ctx[f"net_{label}"], ctx[f"optim_{label}"],
                    ctx[f"optim_{label}"].param_groups[0]["lr"], ctx.get("epoch", 0),
                    str(Path(hps.model_dir) / f"{label.upper()}_{step}.pth"))
        append_json(self.output / "checkpoints.jsonl", {"step": step, "path": str(model),
                    "preserved": str(preserved), "sha256": sha256(model)})
        return model

    def init(self, ctx):
        if not len(ctx["train_loader"]):
            raise ValueError("No batches available after upstream bucketing")
        import torch
        write_json(self.output / "start.json", {"utc": utc_now(), "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "seed": ctx["hps"].train.seed, "dtype": "torch.float32", "steps_requested": self.steps,
            "batch_size": ctx["hps"].train.batch_size, "batches_per_epoch": len(ctx["train_loader"]),
            "generator_parameters": sum(p.numel() for p in ctx["net_g"].parameters()),
            "discriminator_parameters": sum(p.numel() for p in ctx["net_d"].parameters()),
            "pretrained_g": ctx["hps"].pretrainG, "pretrained_d": ctx["hps"].pretrainD,
            "step_definition": "One FP32 batch, one G and one D optimizer update; no GradScaler skips."})
        self.save(ctx, 0)

    def losses(self, ctx):
        self.last_losses = {key: float(ctx[key].detach().cpu()) for key in
                            ("loss_gen_all", "loss_disc", "loss_gen", "loss_fm", "loss_mel", "loss_kl")}
        gradients = {key: float(ctx[key]) for key in ("grad_norm_g", "grad_norm_d")}
        if not all(math.isfinite(x) for x in [*self.last_losses.values(), *gradients.values()]):
            raise FloatingPointError("Nonfinite raw loss/gradient; abort comparison")

    def step(self, ctx, step):
        import torch
        self.last_step = step
        elapsed = time.monotonic() - self.started
        stop = "steps" if step >= self.steps else "interrupt" if self.interrupted else "time" if elapsed >= self.seconds else None
        if step == 1 or step % 10 == 0 or stop:
            torch.cuda.synchronize()
            row = {"step": step, "epoch": ctx["epoch"], **self.last_losses,
                   "wall_s": time.monotonic() - self.started,
                   "learning_rate": ctx["optim_g"].param_groups[0]["lr"],
                   "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "raw_losses_before_upstream_display_caps": True}
            append_json(self.output / "metrics.jsonl", row)
            print(row, flush=True)
        if step % self.every == 0 or stop:
            model = self.save(ctx, step, full=bool(stop))
        if stop:
            for writer in ctx["writers"]:
                writer.flush()
                writer.close()
            write_json(self.output / "summary.json", {"status": "complete" if stop == "steps" else "partial",
                "stop_reason": stop, "steps": step, "wall_s": time.monotonic() - self.started,
                "model": str(model), "model_sha256": sha256(model), "final_train_losses": self.last_losses,
                "quality_claim": "Training losses are not perceptual quality scores; compare held-out WAVs."})
            raise TrainingFinished()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    args, upstream = parser.parse_known_args()
    if upstream and upstream[0] == "--":
        upstream = upstream[1:]
    if min(args.steps, args.seconds, args.checkpoint_every) <= 0:
        raise ValueError("Training limits must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["RVC_CUDA_GRAPH"] = "0"
    os.environ["RVC_AUDIO_FORCE_CPU"] = "1"
    os.environ.setdefault("rmvpe_root", str(RVC / "assets/rmvpe"))
    os.chdir(RVC)
    sys.path.insert(0, str(RVC))
    sys.argv = [str(RVC / "train/train.py"), *upstream]
    import numpy as np
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This comparison requires one visible CUDA GPU")
    random.seed(1234)
    np.random.seed(1234)
    recorder = Recorder(args.output, args.steps, args.seconds, args.checkpoint_every)
    signal.signal(signal.SIGINT, lambda *_: setattr(recorder, "interrupted", True))
    (RVC / "assets/weights").mkdir(parents=True, exist_ok=True)
    source = (RVC / "train/train.py").read_text(encoding="utf-8")
    patched = instrument(source)
    (recorder.output / "instrumented_train.py").write_text(patched, encoding="utf-8")
    write_json(recorder.output / "instrumentation.json", {"original_sha256": sha256(RVC / "train/train.py"),
              "instrumented_sha256": sha256(recorder.output / "instrumented_train.py"),
              "changes": ["same FP32 precision for both arms", "step/time stop and export", "raw-loss recording before display caps"],
              "upstream_file_changed": False, "bitwise_reproducibility_claimed": False})
    scope = {"__name__": "__main__", "__file__": str(RVC / "train/train.py"),
             "_lab_init": recorder.init, "_lab_losses": recorder.losses, "_lab_step": recorder.step}
    recorder.scope = scope
    try:
        exec(compile(patched, str(RVC / "train/train.py"), "exec"), scope)
    except TrainingFinished:
        return
    except BaseException as error:
        write_json(recorder.output / "failure.json", {"utc": utc_now(), "step": recorder.last_step,
                    "error": repr(error), "status": "failed"})
        raise
    raise RuntimeError("Upstream ended without hitting the requested step limit")


if __name__ == "__main__":
    main()
