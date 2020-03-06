from pathlib import Path
from typing import List, Tuple

import yaml
import torch

from sodium.utils import setup_logger

logger = setup_logger(__name__)


class BaseTrainer:
    """Base Trainer for all models
    """

    def __init__(self, model, criterion, optimizer, config, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device

        self.epochs = config['training']['epochs']

    def train(self) -> Tuple[List, List]:
        logger.info('Starting training ...')
        logger.info(f'Training the model for {self.epochs} epochs')

        self.train_metric = []
        self.test_metric = []

        for epoch in range(1, self.epochs+1):
            print(f'\nTraining Epoch: {epoch}')

            trn_metric = self._train_epoch(epoch)  # train this epoch

            print(f'Testing Epoch: {epoch}')

            tst_metric = self._test_epoch(epoch)  # test this epoch

            self.train_metric.extend(trn_metric)
            self.test_metric.extend(tst_metric)

        return (self.train_metric, self.test_metric)

    def _train_epoch(self, epoch: int) -> dict:
        raise NotImplementedError

    def _test_epoch(self, epoch: int):
        raise NotImplementedError

    def _setup_monitoring(self, config: dict) -> None:
        self.epochs = config['epochs']
