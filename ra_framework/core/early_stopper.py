import numpy as np


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Early stopping to prevent overfitting.

        :param patience: Number of epochs with no improvement after which training will be stopped.
        :param delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_loss):
        """
        Check if the current loss is an improvement over the best loss.
        Args:
            current_loss: The loss value for the current epoch.
        """
        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience

    def should_stop(self) -> bool:
        """
        Check if the training should be stopped.
        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        return self.early_stop
