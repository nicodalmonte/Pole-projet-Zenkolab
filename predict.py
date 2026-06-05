import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
from pathlib import Path
import json


def create_model(model_name='resnet50', num_classes=2):
    """Create a model for glaucoma detection"""
    
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def get_transforms(image_size=224):
    """Get image transformations for inference"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def predict_image(model, image_path, transform, device):
    """Predict glaucoma from a single image"""
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    return pred_class, confidence, probs[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Predict glaucoma from fundus images')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_name', type=str, default='resnet50',
                       help='Model architecture')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = create_model(args.model_name, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Get transforms
    transform = get_transforms(args.image_size)
    
    # Make prediction
    pred_class, confidence, probs = predict_image(model, args.image, transform, device)
    
    # Print results
    print("\n" + "=" * 50)
    print(f"Image: {args.image}")
    print(f"Prediction: {'Glaucoma Positive' if pred_class == 1 else 'Glaucoma Negative'}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Probabilities:")
    print(f"  - Negative: {probs[0]:.2%}")
    print(f"  - Positive: {probs[1]:.2%}")
    print("=" * 50)


if __name__ == '__main__':
    main()
