![Overview](https://anonymous.4open.science/r/causality_grammar-DB41/image.png)




## Table of Contents

* [Installation](#installation)

  * [Prerequisites](#prerequisites)
* [Project Structure](#project-structure)
* [Usage](#usage)

  * [Quick Start](#quick-start)
  * [Command Line Arguments](#command-line-arguments)
* [Configuration](#configuration)
* [Examples](#examples)

---

## Installation

### Prerequisites

Make sure you have Python installed along with the required packages. You can install the dependencies using:

```bash
pip install -r requirements.txt
```

Or manually install the main dependencies:

```bash
pip install torch numpy pandas
```

*(Add any other packages your project requires)*

---

## Project Structure

The repository is organized as follows:

```
causality_grammar/
├── DAG_processing.ipynb
├── data_generation.py
├── evaluation.py
├── format_prompt.py
├── data/
│   └── vocab.txt
├── src/
│   ├── DAG_processing (1).ipynb
│   ├── data_annotator.ipynb
│   ├── decoder_inference.py
│   └── decoder_train.py
└── README.md
```

* `DAG_processing.ipynb`: Notebook for processing Directed Acyclic Graphs (DAGs).
* `data_generation.py`: Script for generating synthetic datasets.
* `evaluation.py`: Script to evaluate model performance.
* `format_prompt.py`: Script for formatting prompts.
* `data/`: Directory containing dataset files like vocabulary.
* `src/`: Source code files including training and inference scripts.

---

## Usage

### Quick Start

Run a quick test with synthetic data:

```bash
python src/decoder_train.py
```


---

### Command Line Arguments

| Argument       | Type   | Default | Description               |
| -------------- | ------ | ------- | ------------------------- |
| `--epochs`     | int    | 10      | Number of training epochs |
| `--batch_size` | int    | 32      | Batch size for training   |
| `--lr`         | float  | 0.001   | Learning rate             |
| `--data_path`  | string | ./data  | Path to the dataset       |

*(Add or update arguments according to your scripts)*

---

## Configuration

Configuration parameters can be set directly in the training script or via a config file (if applicable). Key parameters include:

* Learning rate
* Batch size
* Number of epochs
* Dataset paths
* Model save/load paths

---

## Examples

You can try running the notebooks for interactive exploration:

* `DAG_processing.ipynb` for understanding data preprocessing.
* `data_annotator.ipynb` for data annotation steps.

To run the main training pipeline:

```bash
python src/decoder_train.py --epochs 20 --batch_size 64 --lr 0.0005
```

For inference or evaluation:

```bash
python src/decoder_inference.py --model_path models/best_model.pth --data_path ./data/test_data.json
```

---
