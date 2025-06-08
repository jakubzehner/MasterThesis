import numpy
from sklearn.metrics import recall_score

from core.evaluation.metrics.metric_base import MetricBase


class RecallMetric(MetricBase):
    def __init__(self):
        super().__init__("Rec.")

    def _compute_metric(
        self,
        true_labels: numpy.ndarray,
        predicted_labels: numpy.ndarray,
        labels_probs: numpy.ndarray,
    ) -> float:
        return recall_score(true_labels, predicted_labels)
