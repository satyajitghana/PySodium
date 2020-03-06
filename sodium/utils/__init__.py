import os
import random
import numpy as np
import torch

from .logger import setup_logger
from .config import load_config, setup_device, setup_param_groups, get_instance


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
