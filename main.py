import argparse
import os
import random
from typing import Any, List, Tuple, Dict
from types import ModuleType

import numpy as np
import torch
import torch.nn as nn
import torch.optim as module_optimizer
import torch.optim.lr_scheduler as module_scheduler
from torchvision import datasets, transforms

from sodium.utils import setup_logger, load_config, seed_everything
from sodium.trainer import Trainer

import sodium.runner as runner
import sodium.plot as plot

logger = setup_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Sodium Model')

    parser.add_argument('-c', '--config', default=None,
                        type=str, help='config file path (default: None)')

    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    parser.add_argument('--tsai-mode', action='store_true',
                        help='Enable TSAI Mode')

    # parse the arguments
    args = parser.parse_args()

    config = load_config(args.config)

    train_metric, test_metric = runner.train(config, tsai_mode=args.tsai_mode)

    plot.plot_metrics(train_metric, test_metric)
