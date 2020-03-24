import torch
import torch.nn as nn
import torch.optim as module_optimizer
import torch.optim.lr_scheduler as module_scheduler
from torchvision import datasets, transforms

from typing import Any, List, Tuple, Dict
from types import ModuleType

from sodium.utils import get_instance, setup_device, setup_param_groups, setup_logger, seed_everything

import sodium.model.loss as module_loss
import sodium.data_loader.augmentation as module_aug
import sodium.data_loader.data_loaders as module_data

from sodium.trainer import Trainer
import sodium.plot as plot
from sodium.gradcam import get_gradcam, plot_gradcam

from pprint import pformat
import pprint

import copy

from torch_lr_finder import LRFinder

logger = setup_logger(__name__)


class Runner:
    def __init__(self, config):
        self.config = config

    def find_lr(self):
        logger.info('finding the best learning rate')

        # create a model instance
        model = get_instance(module_arch, 'arch', cfg)

        # setup the model with the device
        model, device = setup_device(model, cfg['target_device'])

        param_groups = setup_param_groups(model, cfg['optimizer'])
        optimizer = get_instance(
            module_optimizer, 'optimizer', cfg, param_groups)

        criterion = getattr(module_loss, cfg['criterion'])()

        self.lr_finder = LRFinder(model, optimizer, criterion, device="cuda")

        self.lr_finder.range_test(self.train_loader, val_loader=self.test_loader, start_lr=1e-3,
                                  end_lr=2, num_iter=len(self.train_loader)//self.train_loader.batch_size, step_mode='exp')

        self.lr_finder.find_lr()
        self.lr_finder.plot()
        self.lr_finder.reset()

        del model, optimizer, criterion

    def train(self):
        self.trainer.train()
        logger.info('Finished!')

    def setup_train(self, tsai_mode=False):
        cfg = self.config

        if tsai_mode:
            import sodium.tsai_model as module_arch
        else:
            import sodium.model.model as module_arch

        logger.info('Training Config')

        # display the config
        for line in pprint.pformat(cfg).split('\n'):
            logger.info(line)

        # to get consistent results, seed everything
        seed_everything(cfg['seed'])

        # create a model instance
        model = get_instance(module_arch, 'arch', cfg)

        # setup the model with the device
        model, device = setup_device(model, cfg['target_device'])

        param_groups = setup_param_groups(model, cfg['optimizer'])
        optimizer = get_instance(
            module_optimizer, 'optimizer', cfg, param_groups)

        self.transforms = get_instance(module_aug, 'augmentation', cfg)

        # get the train and test loaders
        self.data_loader = get_instance(
            module_data, 'data_loader', cfg, self.transforms)
        train_loader, test_loader = self.data_loader.get_loaders()

        logger.info('Getting loss function handle')
        criterion = getattr(module_loss, cfg['criterion'])()

        if cfg['lr_scheduler']['type'] == 'OneCycleLR':
            logger.info('Building: torch.optim.lr_scheduler.OneCycleLR')
            sch_cfg = cfg['lr_scheduler']['args']
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=sch_cfg['max_lr'], steps_per_epoch=len(train_loader), epochs=cfg['training']['epochs'])
        else:
            lr_scheduler = get_instance(
                module_scheduler, 'lr_scheduler', cfg, optimizer)

        logger.info('Initializing trainer')
        self.trainer = Trainer(model, criterion, optimizer, cfg, device,
                               train_loader, test_loader, lr_scheduler=lr_scheduler, batch_scheduler=cfg['lr_scheduler']['args']['batch_scheduler'])

    def plot_metrics(self):
        logger.info('Plotting Metrics...')
        plot.plot_metrics(self.trainer.train_metric, self.trainer.test_metric)
        plot.plot_lr_metric(self.trainer.lr_metric)

    def plot_gradcam(self, target_layers):
        logger.info('Plotting Grad-CAM...')

        # use the test images
        data, target = next(iter(self.trainer.test_loader))
        data, target = data.to(self.trainer.device), target.to(
            self.trainer.device)

        logger.info('Taking {5} samples')
        # get 5 images
        data = data[:5]
        target = target[:5]

        # get the generated grad cam
        gcam_layers, predicted_probs, predicted_classes = get_gradcam(
            data, target, self.trainer.model, self.trainer.device, target_layers)

        # get the denomarlization function
        unorm = module_aug.UnNormalize(
            mean=self.transforms.mean, std=self.transforms.std)

        plot_gradcam(gcam_layers, data, target, predicted_classes,
                     self.data_loader.class_names, unorm)
