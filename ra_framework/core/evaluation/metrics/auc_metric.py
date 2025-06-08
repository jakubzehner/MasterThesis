import numpy
from sklearn.metrics import roc_auc_score

from core.evaluation.metrics.metric_base import MetricBase


class AUCMetric(MetricBase):
    def __init__(self):
        super().__init__("AUC")

    def _compute_metric(
        self,
        true_labels: numpy.ndarray,
        predicted_labels: numpy.ndarray,
        labels_probs: numpy.ndarray,
    ) -> float:
        return roc_auc_score(true_labels, labels_probs)
