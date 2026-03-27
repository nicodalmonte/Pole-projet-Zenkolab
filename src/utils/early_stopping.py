
import os
import torch
import mlflow

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss, model, run_id):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
            checkpoint_dir = f"checkpoints/{run_id}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = f"{checkpoint_dir}/best_model.pth"
            torch.save(model.state_dict(), checkpoint_path)
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
            if self.verbose:
                print(f"Checkpoint saved: {checkpoint_path}")
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")

