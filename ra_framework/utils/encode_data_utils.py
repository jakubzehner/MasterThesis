from typing import Any

import torch
from tqdm import tqdm

from readers.code_change_model import CodeChangeModel
from readers.generic_reader import GenericReader


def read_all_datasets(
    datasets_dict: dict[str, list[tuple[str, str]]],
    readers: dict[str, GenericReader],
) -> list[tuple[str, str, list[CodeChangeModel]]]:
    return [
        (dataset_name, file_name, readers[dataset_name].read(file_path))
        for dataset_name, file_paths in datasets_dict.items()
        for file_name, file_path in file_paths
    ]


def tokenize(
    data: list[CodeChangeModel],
    tokenizer: Any,
    device: str,
    model_max_length: int,
) -> list[dict]:
    """
    Tokenizes the input data using the provided tokenizer.

    Args:
        data: List of CodeChangeModel instances to tokenize.
        tokenizer: The tokenizer to use.
        device: Device to move the tensors to (e.g., 'cuda').
        model_max_length: Maximum length for the tokenized sequences.

    Returns:
        A list of dictionaries containing both raw and tokenized fields.
    """
    result = []
    for model in tqdm(data, leave=False):
        entry = {**model.not_tokenized_data()}

        for key, value in model.tokenized_data().items():
            entry[key] = _tokenize_text(value, tokenizer, device, model_max_length)

        result.append(entry)
    return result


def compute_embeddings(model: Any, tokenized_data: list[dict]) -> list[dict]:
    """
    Computes embeddings for tokenized data using the provided model.

    Args:
        model: The encoder model used to compute embeddings.
        tokenized_data: A list of dictionaries containing tokenized fields.

    Returns:
        A list of dictionaries with the same structure, but with embeddings instead of tokenized inputs.
    """
    embeddings = []
    for entry in tqdm(tokenized_data, desc="Computing embeddings", leave=False):
        entry_result = {}
        for key, value in entry.items():
            if key in ["msg", "label"]:
                entry_result[key] = value
            else:
                embedding = _compute_embedding(
                    model, value["input_ids"], value["attention_mask"]
                )
                entry_result[key] = embedding
        embeddings.append(entry_result)
    return embeddings


def _tokenize_text(
    text: str, tokenizer: Any, device: str, model_max_length: int
) -> dict:
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=model_max_length,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    }


def _compute_embedding(
    model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> list[float]:
    with torch.no_grad():
        outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.cpu().tolist()[0]
