from pathlib import Path
from typing import List, Tuple

import yaml
import torch

from sodium.utils import setup_logger
from sodium.metrics import Metrics

logger = setup_logger(__name__)


class BaseTrainer:
    '''Base Trainer for all models'''

    def __init__(self, model, criterion, optimizer, config, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.lr_scheduler = None
        self.epochs = config['training']['epochs']
        self.metrics = Metrics()

    def train(self) -> Tuple[List, List]:
        logger.info('Starting training ...')
        logger.info(f'Training the model for {self.epochs} epochs')

        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accracy = []
        lr_metric = []

        for epoch in range(1, self.epochs+1):
            logger.info(f'Training Epoch: {epoch}')

            if self.lr_scheduler:
                lr_value = [group['lr']
                            for group in self.optimizer.param_groups][0]
                logger.info(f'LR was set to : {lr_value}')
                lr_metric.append(lr_value)

            trn_metric = self._train_epoch(epoch)  # train this epoch

            logger.info(f'Testing Epoch: {epoch}')

            tst_metric = self._test_epoch(epoch)  # test this epoch

            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = tst_metric[2]
                self.lr_scheduler.step(val_loss)

            train_loss.extend(trn_metric[0])
            train_accuracy.extend(trn_metric[1])
            test_loss.extend(tst_metric[0])
            test_accracy.extend(tst_metric[1])

        self.train_metric = (train_loss, train_accuracy)
        self.test_metric = (test_loss, test_accracy)
        self.lr_metric = lr_metric

        return (self.train_metric, self.test_metric)

    def _train_epoch(self, epoch: int) -> dict:
        raise NotImplementedError

    def _test_epoch(self, epoch: int):
        raise NotImplementedError

    def _setup_monitoring(self, config: dict) -> None:
        self.epochs = config['epochs']
