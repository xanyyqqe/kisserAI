import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from config import SimpleRegressor
from wrappers.inference import TorchInference


def test_torch_inferenceNOLOG():

    torch_model = SimpleRegressor()
    