import argparse
import logging

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from readers.apca_reader import APCAReader
from readers.jitdp_reader import JITDPReader
from utils.encode_data_utils import compute_embeddings, read_all_datasets, tokenize
from utils.pickle_saver import save_pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ],
)

logger = logging.getLogger(__name__)

DEVICE = "cuda"
PT_MODEL = "Salesforce/codet5p-770m"
MODEL_INPUT_MAX_LENGTH = 512

DATASETS = {
    "apca": [
        ("small/cv1_test", "data/apca/Small/cv/cv_1/test_patches.pkl"),
        ("small/cv1_train", "data/apca/Small/cv/cv_1/train_patches.pkl"),
        ("small/cv2_test", "data/apca/Small/cv/cv_2/test_patches.pkl"),
        ("small/cv2_train", "data/apca/Small/cv/cv_2/train_patches.pkl"),
        ("small/cv3_test", "data/apca/Small/cv/cv_3/test_patches.pkl"),
        ("small/cv3_train", "data/apca/Small/cv/cv_3/train_patches.pkl"),
        ("small/cv4_test", "data/apca/Small/cv/cv_4/test_patches.pkl"),
        ("small/cv4_train", "data/apca/Small/cv/cv_4/train_patches.pkl"),
        ("small/cv5_test", "data/apca/Small/cv/cv_5/test_patches.pkl"),
        ("small/cv5_train", "data/apca/Small/cv/cv_5/train_patches.pkl"),
        ("small/all", "data/apca/Small/all/all_patches.pkl"),
    ],
    "jitdp": [
        ("gerrit/test", "data/jitdp/gerrit/test.pkl"),
        ("gerrit/train", "data/jitdp/gerrit/train.pkl"),
        ("go/test", "data/jitdp/go/test.pkl"),
        ("go/train", "data/jitdp/go/train.pkl"),
        ("jdt/test", "data/jitdp/jdt/test.pkl"),
        ("jdt/train", "data/jitdp/jdt/train.pkl"),
    ],
}

READERS = {"apca": APCAReader(), "jitdp": JITDPReader()}


def main(pt_model: str, input_max_length: int, device: str):
    logger.info("Starting data encoding...")

    logger.info("Loading datasets...")
    datasets = read_all_datasets(DATASETS, READERS)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(pt_model, trust_remote_code=True)

    logger.info("Tokenizing datasets...")
    tokenized_datasets = {}
    for dataset_name, file_name, code_change_models in tqdm(datasets):
        tokenized_data = tokenize(
            code_change_models, tokenizer, device, input_max_length
        )
        tokenized_datasets[f"{dataset_name}/{file_name}"] = tokenized_data

    logger.info("Tokenization complete.")
    del tokenizer, datasets

    logger.info("Loading model...")
    model = AutoModel.from_pretrained(pt_model, trust_remote_code=True).to(device)

    for dataset_path, tokenized_data in tqdm(tokenized_datasets.items()):
        embeddings = compute_embeddings(model, tokenized_data)
        save_pickle(embeddings, f"data/embeddings/{pt_model}/{dataset_path}.pkl")

    logger.info("Data encoding complete.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Encode data with a pretrained model."
    )
    arg_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=PT_MODEL,
        help="Pretrained model to use for encoding.",
    )
    arg_parser.add_argument(
        "-l",
        "--input_max_length",
        type=int,
        default=MODEL_INPUT_MAX_LENGTH,
        help="Maximum input length for the model.",
    )
    arg_parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=DEVICE,
        help="Device to run the model on (e.g., 'cuda' or 'cpu').",
    )

    args = arg_parser.parse_args()
    main(args.model, args.input_max_length, args.device)
