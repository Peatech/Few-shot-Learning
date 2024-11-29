# Matching Networks for Few-Shot Learning

This project implements Matching Networks, reproducing results from Vinyals et al.'s paper. It supports the **Omniglot** and **MiniImageNet** datasets for N-way K-shot tasks.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
    - [Command-Line Arguments](#command-line-arguments)
4. [Project Structure](#project-structure)
5. [Key Components](#key-components)
6. [How It Works](#how-it-works)
7. [Experimentation](#experimentation)
8. [References](#references)

---

## Introduction

Matching Networks are neural architectures designed for few-shot learning, where models are trained to classify unseen examples by leveraging small support sets. This implementation supports:
- Fully Conditional Embeddings (FCE).
- Customizable distance metrics (e.g., cosine, L2).
- Few-shot learning on Omniglot and MiniImageNet datasets.

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/matching-networks.git
    cd matching-networks
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Prepare the datasets:
    - **Omniglot**: Download from [Omniglot GitHub](https://github.com/brendenlake/omniglot/tree/master/python) and process it:
      ```bash
      python scripts/prepare_omniglot.py
      ```
    - **MiniImageNet**: Follow instructions in `few_shot/datasets.py` to load and preprocess.

---

## Usage

Run the experiment using:
```bash
python -m experiments.matching_nets --dataset omniglot --fce False --k-test 5 --n-test 1 --distance l2
```

## Command-Line Arguments


| Argument           | Default | Description                                                                 |
|--------------------|---------|-----------------------------------------------------------------------------|
| `--dataset`        | None    | Dataset to use (`omniglot` or `miniImageNet`).                              |
| `--fce`            | False   | Whether to enable Fully Conditional Embedding (FCE).                       |
| `--distance`       | cosine  | Distance metric for embeddings (`cosine` or `l2`).                         |
| `--n-train`        | 1       | Number of support examples per class during training.                      |
| `--k-train`        | 5       | Number of classes per task during training.                                |
| `--q-train`        | 15      | Number of query examples per class during training.                        |
| `--n-test`         | 1       | Number of support examples per class during testing.                       |
| `--k-test`         | 5       | Number of classes per task during testing.                                 |
| `--q-test`         | 1       | Number of query examples per class during testing.                         |
| `--lstm-layers`    | 1       | Number of LSTM layers for Fully Conditional Embedding (if enabled).        |
| `--unrolling-steps`| 2       | Number of unrolling steps for the LSTM.                                    |

---

## Project Structure

---

## Key Components

1. **Matching Network (`few_shot/models.py`)**:
   - Computes embeddings for support and query examples.
   - Matches query embeddings to support embeddings using a distance metric.

2. **Task Samplers (`few_shot/core.py`)**:
   - Samples tasks for N-way K-shot learning.
   - Splits data into support and query sets.

3. **Training Loop (`few_shot/train.py`)**:
   - Handles training using the `fit` function.
   - Integrates callbacks for evaluation, logging, and learning rate adjustment.

4. **Callbacks (`few_shot/callbacks.py`)**:
   - **EvaluateFewShot**: Evaluates the model after each epoch.
   - **ModelCheckpoint**: Saves the best-performing model.
   - **CSVLogger**: Logs metrics to a CSV file.

---

## How It Works

### **Flow of Execution**

1. **Initialization**:
   - Parse command-line arguments.
   - Set up directories and verify GPU availability.

2. **Dataset Preparation**:
   - Load the background and evaluation datasets.
   - Use `NShotTaskSampler` to sample tasks for training and validation.

3. **Model Setup**:
   - Define the Matching Network architecture with options like FCE and custom distance metrics.

4. **Training**:
   - Run the `fit` function to train the model.
   - Evaluate the model using callbacks.

5. **Evaluation**:
   - Test the model on unseen tasks sampled from the evaluation dataset.

---

## Experimentation

### Tuning Hyperparameters

- **Task Setup**:
  - Change `--n-train`, `--k-train`, and `--q-train` to modify task complexity.
  - Example: Train with 1-shot 5-way tasks and evaluate with 5-shot 5-way tasks:
    ```bash
    python -m experiments.matching_nets --dataset omniglot --n-train 1 --k-train 5 --n-test 5 --k-test 5
    ```

- **Distance Metric**:
  - Compare performance with `--distance cosine` vs. `--distance l2`.

- **Fully Conditional Embedding**:
  - Enable FCE using `--fce True` to test its effect on performance.

---




## References

- **Original Paper**: [Matching Networks for One-Shot Learning](https://arxiv.org/abs/1606.04080) by Vinyals et al.
- **Omniglot Dataset**: [Omniglot GitHub](https://github.com/brendenlake/omniglot/tree/master/python)


