from src.data.dataset import GlaucomaDataset
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
#yaml 
import yaml

with open("configs/train.yaml", "r") as f:
    cfg = yaml.safe_load(f)

#hyperparameters
EPOCHS=cfg["epochs"]
BATCH_SIZE=cfg["batch_size"]
LR= cfg["lr"]
THRESHOLD = cfg["threshold"]
TRAIN_DATASET = cfg["train_dataset"]
VAL_DATASET = cfg["val_dataset"]
TEST_DATASET = cfg["test_dataset"]
PATIENCE= cfg["patience"]
DELTA= cfg["delta"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Instanziamo i dataset (nota come qui puoi usare qualsiasi nome per la cartella)
training_set = GlaucomaDataset(root=f"datasets/{TRAIN_DATASET}/train/", transform=get_train_transforms())
validation_set = GlaucomaDataset(root=f"datasets/{VAL_DATASET}/validation/", transform=get_val_transforms())
test_set = GlaucomaDataset(root=f"datasets/{TEST_DATASET}/train/images/", transform=get_val_transforms())

train_dataloader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader= DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)    

model= DINOv3large().to(device)
loss= torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params= model.parameters() , lr=LR)


mlflow_utils.setup_mlflow("DINOv3-LAG")

with mlflow.start_run() as run:
    run_id=run.info.run_id
    # log degli iperparametri (una volta sola, fuori dal loop)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning rate", LR)
    mlflow.log_param("patience", PATIENCE)
    mlflow.log_param("threshold", THRESHOLD)
    mlflow.log_param("train dataset", TRAIN_DATASET)
    mlflow.log_param("val dataset", VAL_DATASET)
    mlflow.log_param("test dataset", TEST_DATASET)
    mlflow.log_param("delta", DELTA)
    
    # Basic setup for early stopping criteria
    patience = PATIENCE  # epochs to wait after no improvement
    delta = DELTA  # minimum change in the monitored metric
    best_val_loss = float("inf")  # best validation loss to compare against
    no_improvement_count = 0  # count of epochs with no improvement
    early_stopping = early_stopping.EarlyStopping(patience=patience, delta=delta, verbose=True)

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