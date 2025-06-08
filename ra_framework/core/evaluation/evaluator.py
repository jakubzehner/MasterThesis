import logging

import torch
from sklearn.metrics import classification_report

from core.evaluation.metrics.metric_base import MetricBase
from core.models.binary_classifier_base import BinaryClassifierBase
from core.retrieval_augmented.db import RetrievalAugmentedDB

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        model: BinaryClassifierBase,
        test_data: tuple[torch.Tensor, torch.Tensor],
        metrics: list[MetricBase],
        device: str,
        ra_db: RetrievalAugmentedDB | None = None,
    ):
        self.model = model
        self.test_data = test_data
        self.metrics = metrics
        self.device = device
        self.ra_db = ra_db

    def evaluate(self) -> dict[str, float]:
        """
        Evaluate the model using the provided test data and metrics.
        Returns:
            A dictionary containing the metric names and their corresponding values.
        """
        x_test, y_test = self.test_data

        results = []

        self.model.eval()
        with torch.no_grad():
            if self.ra_db is not None:
                x_test = [x_test, self.ra_db.retrieve(x_test.to(self.device))]
            else:
                x_test = [x_test]

            outputs, _ = self.model(x_test, y_test.to(self.device))

            true_labels = y_test.cpu().numpy()
            labels_probs = outputs.reshape(-1).cpu().numpy()

            logger.info("Gathering metrics...")
            results = [
                metric.compute(true_labels, labels_probs) for metric in self.metrics
            ]

            logger.info(
                "Classification report:\n%s",
                classification_report(true_labels, (labels_probs > 0.5).astype(int)),
            )
        return results
