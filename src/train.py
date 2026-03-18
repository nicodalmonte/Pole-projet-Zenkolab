from src.data.dataset import LAGDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader
import torch
from src.models.model import DINOv3large
from src.training.trainer import train_one_epoch, validate
import mlflow
from src.utils import mlflow_utils
import random
import numpy as np
import torch
from src.utils import early_stopping
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)
EPOCHS=50
BATCH_SIZE=64
LR= 1e-4
THRESHOLD = 0.4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

training_set= LAGDataset(root= "datasets/LAG/LAG/train/", transform=get_train_transforms())
validation_set= LAGDataset(root= "datasets/LAG/LAG/validation/", transform=get_val_transforms())
test_set= LAGDataset(root= "datasets/LAG/LAG/test/", transform=get_val_transforms())

train_dataloader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader= DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)    

model= DINOv3large().to(device)
loss= torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params= model.parameters() , lr=LR)

# Basic setup for early stopping criteria
patience = 5  # epochs to wait after no improvement
delta = 0.01  # minimum change in the monitored metric
best_val_loss = float("inf")  # best validation loss to compare against
no_improvement_count = 0  # count of epochs with no improvement
early_stopping = early_stopping.EarlyStopping(patience=patience, delta=delta, verbose=True)

mlflow_utils.setup_mlflow("DINOv3-LAG")

with mlflow.start_run() as run:
    run_id=run.info.run_id
    # log degli iperparametri (una volta sola, fuori dal loop)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning rate", LR)
    mlflow.log_param("threshold", THRESHOLD)


    for epoch in range(EPOCHS):
        train_loss=train_one_epoch(train_dataloader, model, loss, optimizer, device)
        val_loss, val_metrics = validate(validation_dataloader, model, loss, device, threshold=THRESHOLD)# Check early stopping condition
        early_stopping.check_early_stop(val_loss, model, run_id)
        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | AUC: {val_metrics['auc']:.4f}")
        mlflow_utils.log_epoch(epoch, train_loss, val_loss, val_metrics)
        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")
            break
    test_loss, metric_loss=validate(test_dataloader,model,loss,device, threshold=THRESHOLD)
    mlflow.log_metric("test_loss", test_loss)
    for key, value in metric_loss.items():
        mlflow.log_metric(f"test_{key}", value)