"""Finite training of the upstream conversion module with explicit checkpoints."""
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from .core import ROOT, UPSTREAM, UPSTREAM_SHA, append_json, environment, sha256, write_json


def make_config(dataset, output, run_name, batch_size=2, profile="learning"):
    dataset = Path(dataset).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a single directory name")
    config = yaml.safe_load((UPSTREAM / "configs/reflow.yaml").read_text())
    config["data"].update(train_path=str(dataset / "train"), valid_path=str(dataset / "val"),
                           encoder_ckpt=str(UPSTREAM / "pretrain/contentvec/pytorch_model.bin"),
                           f0_min=50, f0_max=1100)
    config["model"]["n_spk"] = manifest["n_speakers"]
    if profile == "learning":
        config["model"].update(n_aux_layers=4, n_aux_chans=256, n_layers=4, n_chans=512)
    config["vocoder"]["ckpt"] = str(UPSTREAM / "pretrain/nsf_hifigan/model")
    config["env"].update(expdir=str(ROOT / "runs" / run_name), gpu_id=0)
    config["train"].update(batch_size=batch_size, num_workers=0, cache_all_data=False,
                            cache_device="cpu", amp_dtype="fp32", interval_val=100,
                            interval_log=10, save_opt=True)
    config["infer"]["infer_step"] = 10
    # fp32 is a stability baseline; compare bf16 only as an explicit later ablation.
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(output.with_suffix(".metadata.json"), {"profile": profile, "dataset_manifest": str(dataset / "manifest.json"),
               "dataset_sha256": sha256(dataset / "manifest.json"), "upstream_sha": UPSTREAM_SHA})
    return {"config": str(output), "speakers": manifest["n_speakers"], "profile": profile,
            "note": "First-run config uses fp32, batch 2, and a smaller learning model."}


def train(config_path, steps, minutes, resume=None, seed=42, val_clips=3):
    import torch
    from .checkpoint_policy import allow_named_checkpoints
    allow_named_checkpoints(ROOT, UPSTREAM)
    if not torch.cuda.is_available():
        raise RuntimeError("This training stage requires a CUDA GPU; use doctor --require-gpu")
    if steps < 1 or minutes <= 0 or val_clips < 1:
        raise ValueError("steps, minutes and val_clips must be positive")
    config_path = Path(config_path).resolve()
    resume = Path(resume).resolve() if resume else None
    config_sha = sha256(config_path)
    metadata = json.loads(config_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if sha256(metadata["dataset_manifest"]) != metadata["dataset_sha256"]:
        raise ValueError("Dataset manifest changed; create a new config/run")
    dataset = Path(metadata["dataset_manifest"]).parent
    feature_manifest = dataset / "features.json"
    feature_records = json.loads(feature_manifest.read_text(encoding="utf-8"))
    if feature_records["dataset_manifest_sha256"] != metadata["dataset_sha256"]:
        raise ValueError("Features belong to a different prepared dataset")
    if sha256(ROOT / "runs/assets.json") != feature_records["asset_manifest_sha256"]:
        raise ValueError("Asset provenance changed; create a new preprocessing experiment")
    assets = json.loads((ROOT / "runs/assets.json").read_text(encoding="utf-8"))
    for rel, asset in assets["installed"].items():
        if sha256(UPSTREAM / "pretrain" / rel) != asset["sha256"]:
            raise ValueError(f"Downloaded asset changed: {rel}")
    for row in feature_records["files"]:
        if sha256(dataset / row["path"]) != row["sha256"]:
            raise ValueError(f"Preprocessed feature changed: {row['path']}")
    features_sha = sha256(feature_manifest)
    sys.path.insert(0, str(UPSTREAM))
    os.chdir(UPSTREAM)
    from logger.utils import load_config
    from logger.saver import Saver
    from reflow.data_loaders import get_data_loaders
    from reflow.vocoder import Vocoder, Unit2Wav
    from reflow.solver import test
    from optimizer.muon import Muon_AdamW

    args = load_config(config_path)
    out = Path(args.env.expdir)
    if out.exists() and (out / "latest.pt").exists() and resume is None:
        raise ValueError("Run already has latest.pt; use --resume or create a new run")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(args.env.gpu_id)
    start = time.perf_counter()
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)
    model = Unit2Wav(args.data.sampling_rate, args.data.block_size, args.model.win_length,
                     args.data.encoder_out_channels, args.model.n_spk, args.model.use_norm,
                     args.model.use_attention, args.model.use_pitch_aug, vocoder.dimension,
                     args.model.n_aux_layers, args.model.n_aux_chans, args.model.n_layers,
                     args.model.n_chans).to(args.device)
    optimizer = Muon_AdamW(model, muon_args={"weight_decay": args.train.weight_decay},
                           adamw_args={"weight_decay": 0})
    for group in optimizer.param_groups:
        group["lr"] = args.train.lr
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step,
                                                 gamma=args.train.gamma)
    initial_step = 0
    if resume:
        # Load only your own checkpoint produced by this harness.
        cp = torch.load(Path(resume).resolve(), map_location=args.device, weights_only=False)
        if cp.get("config_sha256") != config_sha or cp.get("dataset_sha256") != metadata["dataset_sha256"] or cp.get("features_sha256") != features_sha:
            raise ValueError("Resume config/dataset mismatch")
        model.load_state_dict(cp["model"], strict=True)
        optimizer.load_state_dict(cp["optimizer"])
        scheduler.load_state_dict(cp["scheduler"])
        initial_step = int(cp["global_step"])
    if steps <= initial_step:
        raise ValueError("--steps is the total target, and must exceed the resume step")
    loader, validation = get_data_loaders(args, whole_audio=False)
    if len(loader) == 0 or len(validation) == 0:
        raise ValueError("Empty train/validation loader; check preprocessing")
    # Hold a deterministic, balanced subset fixed across evaluations.
    indices = []
    for speaker in range(1, args.model.n_spk + 1):
        matches = [i for i, name in enumerate(validation.dataset.paths)
                   if int(validation.dataset.data_buffer[name]["spk_id"].item()) == speaker]
        matches.sort(key=lambda i: validation.dataset.paths[i])
        if not matches:
            raise ValueError(f"Missing validation features for speaker {speaker}")
        indices.extend(matches[:val_clips])
    validation = torch.utils.data.DataLoader(torch.utils.data.Subset(validation.dataset, indices),
                                             batch_size=1, shuffle=False)

    class RecordingSaver(Saver):
        def log_value(self, values):
            super().log_value(values)
            append_json(out / "scalars.jsonl", {"step": self.global_step,
                                               **{k: float(v) for k, v in values.items()}})

    saver = RecordingSaver(args, initial_global_step=initial_step)
    amp_name = args.train.amp_dtype
    if amp_name not in {"fp32", "bf16", "fp16"}:
        raise ValueError(f"Unsupported amp_dtype: {amp_name}")
    amp_dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_name == "fp16")
    if resume and "scaler" in cp:
        scaler.load_state_dict(cp["scaler"])
    torch.cuda.reset_peak_memory_stats()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GPU={torch.cuda.get_device_name()} | trainable_params={params:,} | start_step={initial_step}", flush=True)
    iterator = iter(loader)
    last_step = initial_step
    status = "target_reached"
    stop_requested = False
    previous_handler = signal.getsignal(signal.SIGINT)

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; finishing current update before saving.", flush=True)

    signal.signal(signal.SIGINT, request_stop)

    def save(step, permanent=False):
        state = {"global_step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "config_sha256": config_sha, "dataset_sha256": metadata["dataset_sha256"],
                 "features_sha256": features_sha,
                 "upstream_sha": UPSTREAM_SHA, "seed": seed,
                 "resume_note": "Model and optimizer resume; shuffled sample order is not bitwise reproducible."}
        temp = out / "latest.tmp.pt"
        torch.save(state, temp)
        temp.replace(out / "latest.pt")
        if permanent:
            torch.save(state, out / f"model_{step}.pt")

    try:
        # A step-0 checkpoint enables a true before/after comparison.
        if not resume:
            save(0, permanent=True)
        for step in range(initial_step + 1, steps + 1):
            if stop_requested:
                status = "interrupted"
                break
            if time.perf_counter() - start >= minutes * 60:
                status = "time_limit"
                break
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            model.train()
            batch = {k: v.to(args.device) if not k.startswith("name") else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_name != "fp32"):
                ddsp_loss, flow_loss = model(batch["units"].float(), batch["f0"], batch["volume"], batch["spk_id"],
                        aug_shift=batch["aug_shift"], vocoder=vocoder, gt_spec=batch["mel"].float(),
                        infer=False, t_start=args.model.t_start)
                loss = args.train.lambda_ddsp * ddsp_loss + flow_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()
            last_step = step
            saver.global_step = step
            if step % args.train.interval_log == 0 or step == steps:
                record = {"step": step, "train_loss": float(loss), "ddsp_loss": float(ddsp_loss),
                          "flow_loss": float(flow_loss), "batch_examples": batch["units"].shape[0],
                          "step_compute_s": time.perf_counter() - t0,
                          "elapsed_s": time.perf_counter() - start,
                          "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}
                append_json(out / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
                saver.log_value({"train/loss": record["train_loss"], "train/ddsp_loss": record["ddsp_loss"],
                                 "train/reflow_loss": record["flow_loss"]})
            if step % args.train.interval_val == 0 or step == steps:
                save(step, permanent=True)
                ddsp_val, flow_val = test(args, model, vocoder, validation, saver)
                val = {"step": step, "validation_ddsp_loss": ddsp_val, "validation_flow_loss": flow_val,
                       "validation_clips": len(validation)}
                append_json(out / "validation.jsonl", val)
                print(json.dumps(val), flush=True)
                saver.log_value({"validation/ddsp_loss": ddsp_val, "validation/reflow_loss": flow_val})
    except KeyboardInterrupt:
        status = "interrupted"
    except BaseException:
        status = "failed"
        raise
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        if status != "failed":
            save(last_step)
        saver.writer.flush()
        saver.writer.close()
        summary = {"status": status, "initial_step": initial_step, "final_step": last_step,
                   "target_steps": steps, "time_limit_minutes": minutes, "wall_s": time.perf_counter() - start,
                   "trainable_parameters": params, "gpu": torch.cuda.get_device_name(),
                   "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                   "config_sha256": config_sha, "dataset_sha256": metadata["dataset_sha256"],
                   "features_sha256": features_sha,
                   "environment": environment(), "resume_sample_order": "not bitwise reproducible"}
        write_json(out / f"summary_{initial_step}_{last_step}.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary
