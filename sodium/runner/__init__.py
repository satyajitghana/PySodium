import torch
import torch.nn as nn
import torch.optim as module_optimizer
import torch.optim.lr_scheduler as module_scheduler
from torchvision import datasets, transforms

from typing import Any, List, Tuple, Dict
from types import ModuleType

from sodium.utils import get_instance, setup_device, setup_param_groups, setup_logger, seed_everything

import sodium.model.model as module_arch
import sodium.model.loss as module_loss
import sodium.data_loader.augmentation as module_aug
import sodium.data_loader.data_loaders as module_data

from sodium.trainer import Trainer

logger = setup_logger(__name__)


def train(cfg: Dict) -> None:
    logger.info(f'Training: {cfg}')
    seed_everything(cfg['seed'])

    model = get_instance(module_arch, 'arch', cfg)

    model, device = setup_device(model, cfg['target_device'])

    param_groups = setup_param_groups(model, cfg['optimizer'])
    optimizer = get_instance(module_optimizer, 'optimizer', cfg, param_groups)
    # lr_scheduler = get_instance(
    #     module_scheduler, 'lr_scheduler', cfg, optimizer)

    transforms = get_instance(module_aug, 'augmentation', cfg)
    train_loader = get_instance(module_data, 'data_loader', cfg, transforms)
    test_loader = train_loader.test_split()

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.05, steps_per_epoch=len(train_loader), epochs=cfg['training']['epochs'])

    logger.info('Getting loss function handle')
    loss = getattr(module_loss, cfg['loss'])

    logger.info('Initializing trainer')
    trainer = Trainer(model, loss, optimizer, cfg, device,
                      train_loader, test_loader, lr_scheduler=lr_scheduler)

    trainer.train()

    logger.info('Finished!')
