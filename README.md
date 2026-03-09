## Dataset

**LAG (Large-scale Attention-based Glaucoma) dataset** — sourced from Kaggle:
[https://www.kaggle.com/datasets/sreeharims/glaucoma-dataset](https://www.kaggle.com/datasets/sreeharims/glaucoma-dataset)

### Structure

The dataset is pre-split into three folders. Labels are encoded directly in the filename prefix:
- `g.XXXX.jpg` → glaucoma (positive, label = `1`)
- `ng.XXXX.jpg` → non-glaucoma (negative, label = `0`)

### Class distribution

| Split | Glaucoma (`g`) | Non-glaucoma (`ng`) | Total | Ratio |
|---|---|---|---|---|
| `train` | 1 081 | 1 982 | 3 063 | 1 : 1.83 |
| `validation` | 396 | 405 | 801 | ~balanced |
| `test` | 234 | 756 | 990 | 1 : 3.23 |
| **Total** | **1 711** | **3 143** | **4 854** | |

### Class imbalance strategy

The training set is moderately imbalanced. To address this:
- The loss function uses **class weights** inversely proportional to class frequency, penalizing misclassification of the minority class (glaucoma) more heavily.
- Evaluation relies on **AUC-ROC** as the primary metric rather than accuracy.

### Downloading the dataset

The dataset is downloaded automatically via `kagglehub`. Run the exploration script (see [Usage](#usage)) — it will download and copy the data to `datasets/LAG/` on first run.

> The `datasets/` folder is excluded from version control via `.gitignore`. Do not commit raw data.

---

## Project Structure

```
P-le-Projet-Zenkolab/
│
├── src/                        ← all source code
│   ├── __init__.py
│   ├── data/                   ← data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py          ← LAGDataset: PyTorch Dataset class
│   │   └── transforms.py       ← train/val image augmentation pipelines
│   ├── models/                 ← model architecture
│   │   ├── __init__.py
│   │   ├── backbone.py         ← DINOv2 ViT-Large loader and freezing logic
│   │   └── head.py             ← custom classification head
│   ├── training/               ← training logic
│   │   ├── __init__.py
│   │   ├── trainer.py          ← train_one_epoch() and validate() loops
│   │   └── metrics.py          ← AUC, F1, sensitivity, specificity
│   ├── utils/                  ← reusable utilities
│   │   ├── __init__.py
│   │   └── mlflow_utils.py     ← MLflow setup and logging helpers
│   └── train.py                ← main entry point (CLI via argparse)
│
├── scripts/
│   └── train_dino.sh           ← SLURM job submission script
│
├── configs/
│   └── default.yaml            ← hyperparameters and paths
│
├── datasets/                   ← raw data (gitignored)
│   └── LAG/
│       └── LAG/
│           ├── train/
│           ├── validation/
│           └── test/
│
├── checkpoints/                ← saved model weights (gitignored)
├── logs/                       ← SLURM output logs (gitignored)
├── results/                    ← plots, metrics, confusion matrices
│
├── explore_dataset.py          ← one-off dataset exploration script
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Design principles

The project follows **separation of concerns**: each file has one clearly defined responsibility.

| File | Responsibility |
|---|---|
| `dataset.py` | Knows how to read images from disk and extract labels from filenames |
| `transforms.py` | Defines augmentation pipelines — separate from loading logic |
| `backbone.py` | Loads the pre-trained DINOv2 model, handles layer freezing |
| `head.py` | The custom classifier built on top of backbone embeddings |
| `trainer.py` | The training and validation loops — nothing else |
| `metrics.py` | Metric computation — independently testable |
| `mlflow_utils.py` | MLflow boilerplate — write once, reuse everywhere |
| `train.py` | Orchestrator: reads CLI args, instantiates all components, runs training |

---

## Installation

### Prerequisites

- Python >= 3.10
- CUDA >= 11.8 (for GPU training)
- A Kaggle account with API credentials configured for `kagglehub`

### Setup

```bash
# Clone the repository
git clone https://github.com/nicodalmonte/P-le-Projet-Zenkolab
cd P-le-Projet-Zenkolab

# Create and activate a virtual environment
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# Install dependencies
uv sync
```

### Dependencies

```
torch>=2.0
torchvision>=0.15
timm>=0.9
mlflow>=2.0
scikit-learn>=1.3
pandas>=2.0
matplotlib>=3.7
pyyaml>=6.0
pillow>=10.0
kagglehub
```

---

## Usage

### 1. Explore and download the dataset

Run this once to download the dataset and verify its structure:

```bash
python explore_dataset.py
```

This will:
- Download the LAG dataset via `kagglehub` (~295 MB)
- Copy it to `datasets/LAG/`
- Print the folder structure, image counts per split, and label distribution

### 2. Train the model

```bash
# Using default config
python src/train.py --config configs/default.yaml --mlflow

# Overriding specific parameters from CLI
python src/train.py \
    --epochs        30     \
    --batch_size    32     \
    --lr            1e-4   \
    --weight_decay  0.05   \
    --freeze_epochs 3      \
    --mlflow               \
    --mlflow_exp    dino-glaucoma
```

### 3. Quick debug run

To verify the full pipeline works before committing to a long training run:

```bash
python src/train.py --config configs/debug.yaml --debug
```

The debug config uses 2 epochs and a small batch size to catch errors fast.

---

## Experiment Tracking

This project uses **MLflow** to track all experiments. Hyperparameters, metrics, and artifacts are automatically logged for every run.

### What gets logged

| Type | Examples |
|---|---|
| **Params** | `lr`, `batch_size`, `backbone`, `freeze_epochs` |
| **Metrics** | `train_loss`, `val_loss`, `val_auc`, `val_f1` — logged per epoch |
| **Artifacts** | Best model checkpoint, confusion matrix, AUC curve |

### Viewing results

```bash
# Start the MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
```

Then open `http://localhost:5000` in your browser. You can compare runs side by side, inspect metric curves per epoch, and download saved artifacts.

### Storage

MLflow stores all data locally:

```
mlflow.db     ← experiment metadata and metrics (SQLite)
mlruns/       ← artifacts (checkpoints, plots)
```



---





## License

This project was developed as part of an academic machine learning course.
The LAG dataset is subject to its own license — see the [Kaggle dataset page](https://www.kaggle.com/datasets/sreeharims/glaucoma-dataset) for details.
