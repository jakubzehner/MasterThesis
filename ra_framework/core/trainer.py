import logging

import torch
from tqdm import tqdm

from core.early_stopper import EarlyStopper
from core.models.binary_classifier_base import BinaryClassifierBase
from core.retrieval_augmented.db import RetrievalAugmentedDB

logger = logging.getLogger(__name__)


# TODO experiment with retrieval faster than in every bach in every epoch
class Trainer:
    def __init__(
        self,
        model: BinaryClassifierBase,
        optimizer: torch.optim.Optimizer,
        early_stopper: EarlyStopper,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor],
        max_epochs: int,
        batch_size: int,
        device: str,
        ra_db: RetrievalAugmentedDB | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.early_stopper = early_stopper
        self.train_data = train_data
        self.val_data = val_data
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.ra_db = ra_db

    def train(self) -> tuple[float, float]:
        """
        Train the model using the provided training data and validation data.
        Returns:
            A tuple containing the training loss and validation loss.
        """
        x_train, y_train = self.train_data
        x_val, y_val = self.val_data

        train_losses = []
        val_losses = []

        for epoch in tqdm(range(self.max_epochs), desc="Training epochs", unit="epoch"):
            train_loss = (0.0, 0)

            self.model.train()
            for i in range(0, len(x_train), self.batch_size):
                x_batch = x_train[i : i + self.batch_size].to(self.device)
                y_batch = y_train[i : i + self.batch_size].to(self.device)

                if self.ra_db is not None:
                    x_batch = [x_batch, self.ra_db.retrieve(x_batch)]
                else:
                    x_batch = [x_batch]

                self.optimizer.zero_grad()
                _, loss = self.model(x_batch, y_batch)
                loss.backward()
                self.optimizer.step()

                train_loss = (
                    train_loss[0] + loss.item() * y_batch.size(0),
                    train_loss[1] + y_batch.size(0),
                )
            train_losses.append(train_loss[0] / train_loss[1])

            self.model.eval()
            with torch.no_grad():
                x_val_cp = x_val.to(self.device)
                if self.ra_db is not None:
                    x_val_cp = [x_val_cp, self.ra_db.retrieve(x_val_cp)]
                else:
                    x_val_cp = [x_val_cp]

                _, loss = self.model(x_val_cp, y_val.to(self.device))
                self.early_stopper(loss.item())
                val_losses.append(loss.item())

            if self.early_stopper.should_stop():
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        return train_losses, val_losses
