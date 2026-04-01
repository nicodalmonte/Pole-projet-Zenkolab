import torch
import torch.nn.functional as F


def loss_KD(
    logits_S,
    labels,        
    soft_T,       
    temperature=5.0,
    lambda_kl=0.7,
):
    ce = F.cross_entropy(logits_S, labels)
    kl = F.kl_div(
        F.log_softmax(logits_S / temperature, dim=-1),
        F.softmax(soft_T / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return ce + lambda_kl * kl


def _angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    eab = F.normalize(a - b, dim=-1)
    ecb = F.normalize(c - b, dim=-1)
    return (eab * ecb).sum(dim=-1)


def loss_angle(
    soft_T,
    soft_S,
):
    B = soft_T.size(0)
    if B < 3:
        return soft_T.new_tensor(0.0)
    idx = torch.combinations(torch.arange(B, device=soft_T.device), r=3)
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    angle_T = _angle(soft_T[i], soft_T[j], soft_T[k])
    angle_S = _angle(soft_S[i], soft_S[j], soft_S[k])
    return F.huber_loss(angle_S, angle_T)


def loss_HT(
    feat_teacher,   
    feat_student_adapted,
):
    return F.mse_loss(feat_student_adapted, feat_teacher)


def total_loss(
    L_KD,
    L_Angle,
    L_HT,
    alpha= 1.0,
    beta= 2.0,
):
    return L_KD + alpha * L_Angle + beta * L_HT

