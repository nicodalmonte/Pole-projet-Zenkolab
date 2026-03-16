import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def setup_mlflow(experiment_name: str):
    mlflow.set_experiment(experiment_name)

def log_epoch(epoch: int, train_loss: float, val_loss: float, metrics: dict):
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=epoch)