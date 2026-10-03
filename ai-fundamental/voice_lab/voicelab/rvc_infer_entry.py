import os
import random
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voicelab.rvc_assets import RVC

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["RVC_CUDA_GRAPH"] = "0"
os.environ["RVC_AUDIO_FORCE_CPU"] = "1"
os.chdir(RVC)
sys.path.insert(0, str(RVC))
import numpy as np
import torch

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
sys.argv[0] = str(RVC / "infer/cli.py")
runpy.run_path(sys.argv[0], run_name="__main__")
