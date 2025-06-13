# RAFramework

This project explores how artificial intelligence can be used to learn meaningful representations of source code changes. By leveraging a large pretrained encoder and experimenting with retrieval augmentation. The project is a part of a master's thesis in software engineering.

## Requirements

- [CUDA 12.8.1](https://developer.nvidia.com/cuda-12-8-1-download-archive) – Required for GPU acceleration.
- [uv](https://github.com/astral-sh/uv) – Python package manager for managing virtual environments and dependencies.

## Setup

To install the necessary dependencies, run:

```bash
uv sync
```

## Dataset

Download the dataset from the following link:

[CCRep-data.zip](https://drive.google.com/file/d/1s4k2KT3p7XrnxbDXvTvzhQexxLCk4dQd/view?usp=share_link)

Extract the contents into the `data/` directory located in the project root.

## Usage

### Step 1: Encode the Dataset

To preprocess and encode the dataset:

```bash
uv run encode_data.py
```

To see all available options and arguments:

```bash
uv run encode_data.py --help
```

### Step 2: Train the Model

Once the data is encoded, run the training script:

```bash
uv run model_run.py
```

To view available training parameters:

```bash
uv run model_run.py --help
```
