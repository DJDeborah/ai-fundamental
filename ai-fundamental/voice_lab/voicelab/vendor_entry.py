"""Execute upstream scripts with an explicit seed and trusted checkpoint policy."""
import argparse
import runpy
import random
import sys
from pathlib import Path

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=42)
p.add_argument("--script", choices=["main_reflow.py", "preprocess.py"], required=True)
a, rest = p.parse_known_args()
root = Path(__file__).resolve().parents[1]
upstream = root / "upstream/DDSP-SVC"
np.random.seed(a.seed)
random.seed(a.seed)
torch.manual_seed(a.seed)
sys.path.insert(0, str(upstream))

# PyTorch >=2.6 changed its default checkpoint loader. Permit full loading only
# for the named official assets and checkpoints created in this harness's runs/.
from checkpoint_policy import allow_named_checkpoints
allow_named_checkpoints(root, upstream)
sys.argv = [a.script, *rest]
runpy.run_path(str(upstream / a.script), run_name="__main__")
