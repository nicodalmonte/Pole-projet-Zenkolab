import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import json
from datetime import datetime
import argparse


class GlaucomaDataset(Dataset):
    """Custom Dataset for loading glaucoma images"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load Glaucoma Positive images (label 1)
        positive_dir = self.data_dir / "Glaucoma_Positive"
        if positive_dir.exists():
            for img_path in positive_dir.glob("*.jpg"):
                self.images.append(str(img_path))
                self.labels.append(1)
        
        # Load Glaucoma Negative images (label 0)
        negative_dir = self.data_dir / "Glaucoma_Negative"
        if negative_dir.exists():
            for img_path in negative_dir.glob("*.jpg"):
                self.images.append(str(img_path))
                self.labels.append(0)
        
        print(f"Loaded {len(self.images)} images from {data_dir}")
        print(f"  - Positive: {sum(self.labels)}")
        print(f"  - Negative: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size=224, is_training=True):
    """Get image transformations for training and validation"""
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_model(model_name='resnet50', num_classes=2, pretrained=True):
    """Create a model for glaucoma detection"""
    
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def calculate_metrics(all_labels, all_preds, all_probs):
    """Calculate various evaluation metrics"""
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate specificity
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC-ROC
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except:
        auc_roc = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'auc_roc': auc_roc,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    return epoch_loss, metrics


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    return epoch_loss, metrics


def train_model(args):
    """Main training function"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths
    train_dir = Path(args.data_dir) / "Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Train"
    val_dir = Path(args.data_dir) / "Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Validation"
    
    # Create datasets
    train_dataset = GlaucomaDataset(
        train_dir, 
        transform=get_transforms(args.image_size, is_training=True)
    )
    val_dataset = GlaucomaDataset(
        val_dir, 
        transform=get_transforms(args.image_size, is_training=False)
    )
    
    # Calculate class weights for imbalanced dataset
    train_labels = train_dataset.labels
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"\nClass weights: {class_weights}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print(f"\nCreating model: {args.model_name}")
    model = create_model(args.model_name, num_classes=2, pretrained=args.pretrained)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1_score']:.4f} | Val AUC-ROC: {val_metrics['auc_roc']:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc_roc'])
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_metrics['accuracy']
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'args': vars(args)
            }
            
            save_path = args.output_dir / f"best_model_{args.model_name}.pth"
            torch.save(checkpoint, save_path)
            print(f"✓ Saved best model to {save_path}")
    
    print("\n" + "=" * 80)
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Save training history
    history_path = args.output_dir / f"training_history_{args.model_name}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history to {history_path}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train glaucoma detection model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, 
                       default='datasets/sshikamaru__glaucoma-detection',
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory to save outputs')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='resnet50',
                       choices=['resnet18', 'resnet50', 'efficientnet_b0', 'vgg16'],
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save arguments
    with open(args.output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Train model
    model, history = train_model(args)


if __name__ == '__main__':
    main()
