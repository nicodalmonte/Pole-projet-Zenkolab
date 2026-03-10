import torch 
from src.training.metrics import compute_metrics
import numpy as np

def train_one_epoch(dataloader, model, criterion, optimizer, device):
    model.train()
    train_total_loss=0.0
    for batch, (image, label) in enumerate(dataloader):
        image , label = image.to(device) , label.to(device)
        optimizer.zero_grad()
        pred=model(image)
        loss=criterion(pred, label)
        train_total_loss+=loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss=train_total_loss/len(dataloader)
    return avg_train_loss


def validate(dataloader, model, criterion, device):
    model.eval()
    test_total_loss=0.0
    all_labels = []
    all_probs  = []
    with torch.no_grad():
        for (image, label) in dataloader:
            image , label = image.to(device), label.to(device)
            pred=model(image)
            test_loss=criterion(pred, label)
            test_total_loss+=test_loss.item()
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(torch.softmax(pred, dim=1)[:, 1].cpu().numpy())
    all_preds = (np.array(all_probs) >= 0.5).astype(int)
    metrics = compute_metrics(all_labels, all_probs, all_preds)
    avg_test_loss= test_total_loss/len(dataloader)
    return avg_test_loss, metrics


