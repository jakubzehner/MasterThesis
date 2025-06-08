# CCREP Experiments: Query-Back Mechanism Exploration

This repository contains experiments related to the **Query Back mechanism** proposed in the paper [_CCREP: Learning Code Change Representations via the Query Back Mechanism_](https://dl.acm.org/doi/abs/10.1109/ICSE48619.2023.00014), published at ICSE 2023.
The project builds upon the official implementation provided by the authors and serves as part of a master's thesis on **Learning Code Change Representations via Artificial Intelligence Code Model**.

## Usage

You can run the experiments in one of two ways:

---

### Option 1: Using the _Apptainer_ Container

This is the recommended way to reproduce the original experiments in an isolated and GPU-enabled environment.

#### Requirements

- [Apptainer](https://apptainer.org/docs/user/latest/)
- NVIDIA GPU and drivers for CUDA support

#### Step 1: Build the container

```bash
sudo apptainer build --bind $(pwd):/mnt ccrep_env.sif apptainer.def
```

- This command builds the Apptainer image `ccrep_env.sif` using the `apptainer.def` definition file.
- The `--bind` option maps the current working directory into the container as `/mnt`.

#### Step 2: Run the container

```bash
apptainer exec --nv --bind /tmp:/tmp ccrep_env.sif python {script}
```

Replace `{script}` with the desired Python script.

- `--nv` enables GPU support
- `/tmp` binding is required for some operations

---

### Option 2: Manual Environment Setup

If you prefer not to use containers, you can manually replicate the environment based on the steps in `apptainer.def`.

---

## Task Execution

Below are examples of how to run each supported task. Hovewer, you need to download the datasets from [CCRep-data.zip](https://drive.google.com/file/d/1s4k2KT3p7XrnxbDXvTvzhQexxLCk4dQd/view?usp=share_link) first and extract them into the `data` directory in project root.

### APCA Task

```bash
apptainer exec --nv --bind /tmp:/tmp ccrep_env.sif python tasks/apca/apca_cv_train_helper.py -model {apca_model} -dataset {Small|Large} -cuda 0
```

Where possible combinations of model and dataset are:

| Model     | Dataset       |
| --------- | ------------- |
| token     | Small / Large |
| line      | Small / Large |
| hybrid    | Small / Large |
| graph     | Small / Large |
| unixcoder | Small         |
| codet5    | Small         |
| codet5p   | Small         |
| codet5pl  | Small         |

---

### CMG Task

```bash
apptainer exec --nv --bind /tmp:/tmp ccrep_env.sif python tasks/cmg/cmg_train_from_config.py -model {cmg_model} -dataset {corec|fira} -cuda 0
```

Where possible combinations of model and dataset are:

| Model     | Dataset      |
| --------- | ------------ |
| token     | corec / fira |
| line      | corec / fira |
| hybrid    | corec / fira |
| graph     | fira         |
| unixcoder | fira         |
| codet5    | fira         |
| codet5p   | fira         |
| codet5pl  | fira         |

---

### JIT-DP Task

```bash
python tasks/jitdp/jitdp_train_from_config.py -model {jitdp_model} -project {project} -cuda 0
```

Where possible combinations of model and project are:

| Model     | Project                                  |
| --------- | ---------------------------------------- |
| token     | gerrit / go / jdt / openstack / platform |
| line      | gerrit / go / jdt / openstack / platform |
| hybrid    | gerrit / go / jdt / openstack / platform |
| graph     | gerrit / go / jdt                        |
| unixcoder | gerrit / go / jdt                        |
| codet5    | gerrit / go / jdt                        |
| codet5p   | gerrit / go / jdt                        |
| codet5pl  | gerrit / go / jdt                        |

---

## 📘 Reference

- **Paper**: [CCREP: Learning Code Change Representations via the Query Back Mechanism](https://dl.acm.org/doi/abs/10.1109/ICSE48619.2023.00014)
- **Original Repository**: [GitHub - CCREP](https://github.com/ZJU-CTAG/CCRep)
