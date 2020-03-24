from typing import Tuple, List

from sodium.utils import setup_logger
from sodium.base import BaseTrainer

from tqdm.auto import tqdm, trange
import torch

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.style.use("dark_background")

logger = setup_logger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, model, loss, optimizer, config, device, train_loader, test_loader, lr_scheduler=None, batch_scheduler=False):
        super().__init__(model, loss, optimizer, config, device)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.batch_scheduler = batch_scheduler

    def _train_epoch(self, epoch: int) -> List[Tuple]:

        loss_history = []
        accuracy_history = []

        self.model.train()  # set the model in training mode

        train_loss = 0
        correct = 0
        total = 0
        processed = 0

        pbar = tqdm(self.train_loader, dynamic_ncols=True)

        for batch_idx, (data, target) in enumerate(pbar):

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)

            loss = self.criterion(output, target)

            loss.backward()

            self.optimizer.step()

            train_loss += loss.item()

            _, predicted = output.max(1)

            total += target.size(0)

            correct += predicted.eq(target).sum().item()

            processed += len(data)

            pbar.set_description(
                desc=f'epoch={epoch-1+batch_idx/len(pbar):.2f} | loss={train_loss/(batch_idx+1):.10f} | accuracy={100.*correct/total:.2f} {correct}/{total} | batch_id={batch_idx}')

            accuracy_history.append(100.*correct/processed)
            loss_history.append(loss.data.cpu().numpy().item())

            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.lr_scheduler.step()

        torch.cuda.empty_cache()

        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.StepLR):
            self.lr_scheduler.step()

        return (loss_history, accuracy_history)

    def _test_epoch(self, epoch: int) -> List[Tuple]:

        loss_history = []
        accuracy_history = []

        self.model.eval()  # set the model in evaluation mode

        test_loss = 0
        correct = 0
        total = 0

        # pbar = tqdm(self.test_loader, dynamic_ncols=True)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                loss = self.criterion(output, target)

                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # pbar.set_description(
                #     desc=f'epoch={epoch+batch_idx/len(pbar):.2f} | loss={test_loss/(batch_idx+1):.10f} | accuracy={100.*correct/total:.2f} {correct}/{total} | batch_id={batch_idx}')

        logger.info(
            f'Test Set: Average Loss: {test_loss/len(self.test_loader):.8f}, Accuracy: {100. * correct / total:.2f} ({correct}/{total})')

        loss_history.append(test_loss/len(self.test_loader))
        accuracy_history.append((100. * correct) / total)

        return (loss_history, accuracy_history)
