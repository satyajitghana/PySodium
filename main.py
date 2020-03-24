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

logger = setup_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Sodium Model')

    parser.add_argument('-c', '--config', default=None,
                        type=str, help='config file path (default: None)')

    parser.add_argument('--tsai-mode', action='store_true',
                        help='Enable TSAI Mode')

    # parse the arguments
    args = parser.parse_args()

    # load the config
    config = load_config(args.config)

    # create a runner
    runner = runner.Runner(config)

    # setup train parameters
    runner.setup_train(tsai_mode=args.tsai_mode)

    # find lr
    # runner.find_lr()

    # train the network
    runner.train()

    # plot metrics
    runner.plot_metrics()

    # plot gradcam
    target_layers = ["layer1", "layer2", "layer3", "layer4"]
    runner.plot_gradcam(target_layers=target_layers)
