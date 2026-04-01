import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, accuracy_score
import mlflow
import mlflow.pytorch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dataset import build_train_dataset, build_val_dataset
from model import build_retfound, build_vit

from model import StudentViT, FitNet, ImportanceAdapter   # distillation/model.py
from loss import loss_KD, loss_angle, loss_HT, total_loss  # distillation/loss.py


CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

# ---------------------------------------------------------------------------
# Teacher wrapper — extrait (logits, features) d'un modèle teacher figé
# ---------------------------------------------------------------------------
class TeacherWrapper(nn.Module):
    """Encapsule un teacher (backbone + head) et renvoie ses features et logits."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """Retourne (logits, features) où features ∈ (B, teacher_embed_dim)."""
        features = self.model.backbone(x)  # (B, 1024) pour ViT-Large
        logits = self.model.head(features)
        return logits, features


def load_teachers(device: torch.device) -> list:
    """Charge les 2 teachers pré-entraînés depuis les checkpoints."""
    teachers = []
    specs = [
        (build_retfound, CHECKPOINT_DIR / "best_retfound.pth", "RETFound"),
        (build_vit,      CHECKPOINT_DIR / "best_vit.pth",      "ViT-Large"),
    ]
    for builder, ckpt_path, name in specs:
        model = builder(num_classes=2, freeze_backbone=False, load_pretrained=False)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        wrapper = TeacherWrapper(model).to(device)
        print(f"[teacher] {name} chargé depuis {ckpt_path.name}")
        teachers.append(wrapper)
    return teachers


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------
def run_epoch(
    student, adapter, fitnets, teachers,
    loader, optimizer, scaler, device,
    alpha=1.0, beta=2.0, temperature=5.0, lambda_kl=0.7,
    train: bool = True,
):
    student.train(train)
    adapter.train(train)
    for fn in fitnets:
        fn.train(train)

    total_loss_sum = 0.0
    all_labels, all_probs = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=scaler is not None):
            # ── Student forward
            feat_S = student.forward_features(images)   # (B, 768)
            logits_S = student.head(feat_S)              # (B, 2)

            # ── Poids d'importance via l'adapter  (B, num_teachers)
            weights = adapter(feat_S)                    # (B, T)

            # ── Teachers forward (no grad)
            teacher_logits, teacher_feats = [], []
            for t in teachers:
                with torch.no_grad():
                    lg, ft = t(images)
                teacher_logits.append(lg)   # (B, 2)
                teacher_feats.append(ft)    # (B, 1024)

            # ── Intégration soft targets  (Eq. 7)
            # soft_T_tilde[i] = sum_t w_{t,i} * y_T_{t,i}
            stacked = torch.stack(teacher_logits, dim=1)  # (B, T, 2)
            soft_T = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # (B, 2)

            # ── L_KD  (Eq. 8)
            L_kd = loss_KD(logits_S, labels, soft_T, temperature, lambda_kl)

            # ── L_Angle  (Eq. 10) — sur les soft targets intégrés
            soft_S = logits_S / temperature
            soft_T_norm = soft_T / temperature
            L_ang = loss_angle(soft_T_norm, soft_S)

            # ── L_HT  (Eq. 11) — un FitNet par teacher
            L_ht = sum(
                loss_HT(teacher_feats[t_idx], fitnets[t_idx](feat_S))
                for t_idx in range(len(teachers))
            )

            loss = total_loss(L_kd, L_ang, L_ht, alpha=alpha, beta=beta)

        if train:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss_sum += loss.item()
        probs = torch.softmax(logits_S.detach(), dim=-1)[:, 1].cpu().tolist()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().tolist())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    acc = accuracy_score(all_labels, [int(p > 0.5) for p in all_probs])
    return total_loss_sum / len(loader), auc, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Hyper-paramètres
    nb_epochs   = 30
    size_batch  = 32
    lr          = 1e-4
    weight_decay = 0.05
    patience    = 7
    alpha, beta = 1.0, 2.0        # poids L_Angle et L_HT dans la loss totale
    temperature = 5.0
    lambda_kl   = 0.7
    num_workers = 4
    data_root   = "datasets/"
    use_mlflow  = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Données
    train_ds = build_train_dataset(data_root)
    val_ds   = build_val_dataset(data_root)
    train_loader = DataLoader(train_ds, batch_size=size_batch, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=size_batch, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # ── Teachers (figés)
    teachers = load_teachers(device)
    num_teachers = len(teachers)
    teacher_embed_dim = 1024   # ViT-Large embed dim
    student_embed_dim = 768    # ViT-Base embed dim

    # ── Student, FitNets, Adapter
    student = StudentViT(num_classes=2).to(device)
    fitnets = nn.ModuleList([
        FitNet(student_embed_dim, teacher_embed_dim).to(device)
        for _ in range(num_teachers)
    ])
    adapter = ImportanceAdapter(num_teachers, student_embed_dim).to(device)

    # ── Optimiseur (student + fitnets + adapter)
    params = [
        *student.parameters(),
        *fitnets.parameters(),
        *adapter.parameters(),
    ]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb_epochs)
    scaler    = GradScaler()

    best_val_auc   = 0.0
    no_improve     = 0
    best_ckpt_path = CHECKPOINT_DIR / "best_distilled.pth"

    if use_mlflow:
        mlflow.set_experiment("glaucoma-distillation")
        mlflow.start_run()
        mlflow.log_params(dict(
            epochs=nb_epochs, batch_size=size_batch, lr=lr,
            alpha=alpha, beta=beta, temperature=temperature,
            lambda_kl=lambda_kl, num_teachers=num_teachers,
        ))

    for epoch in range(1, nb_epochs + 1):
        train_loss, train_auc, train_acc = run_epoch(
            student, adapter, fitnets, teachers,
            train_loader, optimizer, scaler, device,
            alpha=alpha, beta=beta,
            temperature=temperature, lambda_kl=lambda_kl,
            train=True,
        )
        val_loss, val_auc, val_acc = run_epoch(
            student, adapter, fitnets, teachers,
            val_loader, None, None, device,
            alpha=alpha, beta=beta,
            temperature=temperature, lambda_kl=lambda_kl,
            train=False,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{nb_epochs} "
            f"│ train loss {train_loss:.4f}  AUC {train_auc:.4f}  acc {train_acc:.4f} "
            f"│ val   loss {val_loss:.4f}  AUC {val_auc:.4f}  acc {val_acc:.4f}"
        )

        if use_mlflow:
            mlflow.log_metrics(dict(
                train_loss=train_loss, train_auc=train_auc, train_acc=train_acc,
                val_loss=val_loss,   val_auc=val_auc,   val_acc=val_acc,
            ), step=epoch)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve   = 0
            torch.save(student.state_dict(), best_ckpt_path)
            print(f"  ↳ Meilleur modèle sauvegardé (AUC={best_val_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[early stopping] pas d'amélioration depuis {patience} epochs.")
                break

    if use_mlflow:
        mlflow.log_artifact(str(best_ckpt_path))
        mlflow.end_run()

    print(f"\nEntraînement terminé. Meilleur AUC validation : {best_val_auc:.4f}")


if __name__ == "__main__":
    main()

