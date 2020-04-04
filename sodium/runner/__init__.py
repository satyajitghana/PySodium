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

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

logger = setup_logger(__name__)


class Runner:
    def __init__(self, config):
        self.config = config

    def find_lr(self):
        from torch_lr_finder import LRFinder

        logger.info('finding the best learning rate')

        cfg = self.config

        if self.tsai_mode:
            import sodium.tsai_model as module_arch
        else:
            import sodium.model.model as module_arch

        # create a model instance
        model = get_instance(module_arch, 'arch', cfg)

        # setup the model with the device
        model, device = setup_device(model, cfg['target_device'])

        param_groups = setup_param_groups(model, cfg['optimizer'])
        optimizer = get_instance(
            module_optimizer, 'optimizer', cfg, param_groups)

        criterion = getattr(module_loss, cfg['criterion'])()

        self.lr_finder = LRFinder(model, optimizer, criterion, device="cuda")

        # my method
        # self.lr_finder.range_test(self.trainer.train_loader, start_lr=1e-3,
        #                           end_lr=5, num_iter=len(self.trainer.train_loader), step_mode='exp')

        # leslie smith method
        # self.lr_finder.range_test(self.trainer.train_loader, val_loader = self.trainer.test_loader,
        # end_lr=1, num_iter=len(self.trainer.train_loader), step_mode='linear')

        # fast ai method
        self.lr_finder.range_test(
            self.trainer.train_loader, end_lr=100, num_iter=len(self.trainer.train_loader))

        self.best_lr = self.lr_finder.history['lr'][self.lr_finder.history['loss'].index(
            self.lr_finder.best_loss)]

        sorted_lrs = [x for _, x in sorted(
            zip(self.lr_finder.history['loss'], self.lr_finder.history['lr']))]

        logger.info(f'sorted lrs : {sorted_lrs[:10]}')

        logger.info(f'found the best lr : {self.best_lr}')

        logger.info('plotting lr_finder')

        plt.style.use("dark_background")
        self.lr_finder.plot()

        # reset the model and the optimizer
        self.lr_finder.reset()
        plt.show()

        del model, optimizer, criterion

    def train(self, use_bestlr=False, lr_value=None):

        # if the best lr was found use that value instead
        if use_bestlr and self.best_lr is not None:
            for param_group in self.trainer.optimizer.param_groups:
                param_group['lr'] = self.best_lr

        if not use_bestlr and (lr_value is not None):
            for param_group in self.trainer.optimizer.param_groups:
                param_group['lr'] = lr_value

        self.trainer.train()
        logger.info('Finished!')

    def setup_train(self, tsai_mode=False):
        cfg = self.config

        self.tsai_mode = tsai_mode

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

        batch_scheduler = False
        if cfg['lr_scheduler']['type'] == 'OneCycleLR':
            logger.info('Building: torch.optim.lr_scheduler.OneCycleLR')
            sch_cfg = cfg['lr_scheduler']['args']
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=sch_cfg['max_lr'], steps_per_epoch=len(train_loader), epochs=cfg['training']['epochs'])
            batch_scheduler = True
        else:
            lr_scheduler = get_instance(
                module_scheduler, 'lr_scheduler', cfg, optimizer)

        logger.info('Initializing trainer')
        self.trainer = Trainer(model, criterion, optimizer, cfg, device,
                               train_loader, test_loader, lr_scheduler=lr_scheduler, batch_scheduler=batch_scheduler)

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

    def plot_misclassifications(self, target_layers):
        assert(self.trainer.model is not None)
        # get the data, target of only missclassified and do what you do for gradcam

        logger.info('getting misclassifications')

        misclassified = []
        misclassified_target = []
        misclassified_pred = []

        model, device = self.trainer.model, self.trainer.device

        # set the model to evaluation mode
        model.eval()

        # turn off gradients
        with torch.no_grad():
            for data, target in self.trainer.test_loader:
                # move them to respective device
                data, target = data.to(device), target.to(device)

                # do inferencing
                output = model(data)

                # get the predicted output
                pred = output.argmax(dim=1, keepdim=True)

                # get the current misclassified in this batch
                list_misclassified = (target.eq(pred.view_as(target)) == False)
                batch_misclassified = data[list_misclassified]
                batch_mis_pred = pred[list_misclassified]
                batch_mis_target = target[list_misclassified]

                # batch_misclassified =

                misclassified.append(batch_misclassified)
                misclassified_pred.append(batch_mis_pred)
                misclassified_target.append(batch_mis_target)

        # group all the batched together
        misclassified = torch.cat(misclassified)
        misclassified_pred = torch.cat(misclassified_pred)
        misclassified_target = torch.cat(misclassified_target)

        logger.info('Taking {25} samples')
        # get 5 images
        data = misclassified[:25]
        target = misclassified_target[:25]

        # get the generated grad cam
        gcam_layers, predicted_probs, predicted_classes = get_gradcam(
            data, target, self.trainer.model, self.trainer.device, target_layers)

        # get the denomarlization function
        unorm = module_aug.UnNormalize(
            mean=self.transforms.mean, std=self.transforms.std)

        plot_gradcam(gcam_layers, data, target, predicted_classes,
                     self.data_loader.class_names, unorm)
