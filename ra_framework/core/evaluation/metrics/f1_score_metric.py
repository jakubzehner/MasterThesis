import numpy
from sklearn.metrics import f1_score

from core.evaluation.metrics.metric_base import MetricBase


class F1ScoreMetric(MetricBase):
    def __init__(self):
        super().__init__("F1")

    def _compute_metric(
        self,
        true_labels: numpy.ndarray,
        predicted_labels: numpy.ndarray,
        labels_probs: numpy.ndarray,
    ) -> float:
        return f1_score(true_labels, predicted_labels)
