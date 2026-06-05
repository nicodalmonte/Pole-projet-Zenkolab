# P-le-Projet-Zenkolab

## Glaucoma Detection from Fundus Images

This project implements a deep learning model for detecting glaucoma from retinal fundus images using PyTorch and transfer learning.

## Dataset

The project uses the glaucoma detection dataset with the following structure:
- **Training set**: 520 images (134 positive, 386 negative)
- **Validation set**: 130 images
- Images are organized in `Glaucoma_Positive` and `Glaucoma_Negative` folders

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Training

### Basic Usage

Train a model with default parameters (ResNet50):
```bash
python train.py
```

### Advanced Usage

Train with custom parameters:
```bash
python train.py \
  --model_name resnet50 \
  --epochs 30 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --image_size 224
```

### Available Models

- `resnet18` - ResNet-18 (faster, less accurate)
- `resnet50` - ResNet-50 (default, good balance)
- `efficientnet_b0` - EfficientNet-B0 (efficient)
- `vgg16` - VGG-16 (slower, high capacity)

### Training Parameters

- `--data_dir`: Path to dataset directory (default: `datasets/sshikamaru__glaucoma-detection`)
- `--output_dir`: Directory to save model and logs (default: `outputs`)
- `--model_name`: Model architecture (default: `resnet50`)
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.0001)
- `--image_size`: Input image size (default: 224)
- `--num_workers`: Number of data loading workers (default: 4)

## Inference

Predict glaucoma from a single image:
```bash
python predict.py \
  --image path/to/image.jpg \
  --model_path outputs/best_model_resnet50.pth \
  --model_name resnet50
```

## Model Features

- **Transfer Learning**: Uses pretrained ImageNet weights
- **Data Augmentation**: Random flips, rotations, and color jittering
- **Class Imbalance Handling**: Weighted loss function
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, AUC-ROC
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Model Checkpointing**: Saves best model based on validation loss

## Output Files

After training, the following files are saved in the output directory:
- `best_model_{model_name}.pth` - Best model checkpoint
- `training_history_{model_name}.json` - Training and validation metrics
- `args.json` - Training configuration

## Example Training Output

```
Using device: cuda
Loaded 520 images from datasets/.../Train
  - Positive: 134
  - Negative: 386
Loaded 130 images from datasets/.../Validation
  - Positive: 35
  - Negative: 95

Creating model: resnet50

Starting training for 30 epochs...
================================================================================

Epoch 1/30
--------------------------------------------------------------------------------
Train Loss: 0.4532 | Train Acc: 0.8192
Val Loss:   0.3821 | Val Acc:   0.8615
Val Precision: 0.7500 | Val Recall: 0.7143
Val F1: 0.7317 | Val AUC-ROC: 0.8945
✓ Saved best model to outputs/best_model_resnet50.pth
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- PIL
- scikit-learn
- numpy