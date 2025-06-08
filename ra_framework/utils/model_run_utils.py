import logging
from typing import Literal

import torch

from core.models.binary_classifier_base import BinaryClassifierBase
from core.models.ra_attention_like_bc import RaAttentionLikeClassifier
from core.models.ra_average_bc import RaAverageClassifier
from core.models.ra_weighted_bc import RaWeightedClassifier
from core.models.simple_bc import SimpleBinaryClassifier
from core.retrieval_augmented.retrieval_methods.cos_sim_retriever import (
    CosineSimilarityRetriever,
)
from core.retrieval_augmented.retrieval_methods.euclidean_retriever import (
    EuclideanRetriever,
)
from core.retrieval_augmented.retrieval_methods.manhattan_retriever import (
    ManhattanRetriever,
)
from core.retrieval_augmented.retrieval_methods.retriever_base import RetrieverBase

logger = logging.getLogger(__name__)


def get_model(
    model_name: Literal["simple", "ra_average", "ra_weighted", "ra_attention"],
    input_dim: int,
) -> BinaryClassifierBase:
    match model_name:
        case "simple":
            return SimpleBinaryClassifier(input_dim)
        case "ra_average":
            return RaAverageClassifier(input_dim)
        case "ra_weighted":
            return RaWeightedClassifier(input_dim)
        case "ra_attention":
            return RaAttentionLikeClassifier(input_dim)
        case _:
            error_msg = f"Model {model_name} not implemented."
            logger.error(error_msg)
            raise NotImplementedError(error_msg)


def get_retriever(
    retriever_name: Literal["cosine", "euclidean", "manhattan"],
) -> RetrieverBase:
    match retriever_name:
        case "cosine":
            return CosineSimilarityRetriever()
        case "euclidean":
            return EuclideanRetriever()
        case "manhattan":
            return ManhattanRetriever()
        case _:
            error_msg = f"Retriever {retriever_name} not implemented."
            logger.error(error_msg)
            raise NotImplementedError(error_msg)


def pretty_print_metrics(metrics: list[tuple[str, float]]):
    max_name_length = max(len(name) for name, _ in metrics)

    for name, value in metrics:
        print(f"{name:<{max_name_length}}: {value:.5f}")


def calculate_average_metrics(
    metric_list: list[list[tuple[str, float]]],
) -> list[tuple[str, float]]:
    if not metric_list:
        return []

    num_metrics = len(metric_list[0])
    average_metrics = []

    for i in range(num_metrics):
        metric_name = metric_list[0][i][0]
        metric_values = [metric[i][1] for metric in metric_list]
        average_value = sum(metric_values) / len(metric_values)
        average_metrics.append(("Avg. " + metric_name, average_value))

    return average_metrics


def save_metrics_to_file(path: str, metrics: list[tuple[str, float]]):
    with open(path, "w") as file:
        for name, value in metrics:
            file.write(f"{name}: {value:.5f}\n")


def get_x_and_y_tensors(embeddings: list[dict], device: str) -> torch.Tensor:
    input_keys = [key for key in embeddings[0].keys() if key not in ["label", "code"]]
    x_input_lists = [[item[key] for item in embeddings] for key in input_keys]

    x_tensor = _concat_embeddings(_list_of_lists_to_tensor(x_input_lists, device))
    y_tensor = _labels_to_tensor([item["label"] for item in embeddings], device)
    return x_tensor, y_tensor


def _labels_to_tensor(labels: list[int], device: str) -> torch.Tensor:
    return torch.tensor(labels, dtype=torch.float32).to(device).view(-1)


def _list_of_lists_to_tensor(list_of_lists: list[list], device: str) -> torch.Tensor:
    transposed = list(zip(*list_of_lists))
    stacked = [torch.tensor(group, dtype=torch.float32) for group in transposed]
    return torch.stack(stacked).to(device)


def _concat_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.cat([embeddings[:, i, :] for i in range(embeddings.shape[1])], dim=1)
