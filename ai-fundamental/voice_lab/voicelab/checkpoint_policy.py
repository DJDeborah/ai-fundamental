from pathlib import Path


def allow_named_checkpoints(root, upstream):
    """Allow full loading only for official assets and this harness's own runs."""
    import torch
    if getattr(torch.load, "_voice_lab_policy", False):
        return
    original = torch.load
    assets = {(upstream / "pretrain" / p).resolve() for p in
              ("contentvec/pytorch_model.bin", "nsf_hifigan/model", "rmvpe/model.pt")}

    def load(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            path = Path(file).resolve()
            if path in assets or path.is_relative_to((root / "runs").resolve()):
                kwargs.setdefault("weights_only", False)
        return original(file, *args, **kwargs)

    load._voice_lab_policy = True
    torch.load = load
