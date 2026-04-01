import torch
import torch.nn as nn
import timm
from torchvision.transforms import Resize


def _make_head(embed_dim: int, num_classes: int, dropout_rate: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Dropout(p=dropout_rate),
        nn.Linear(embed_dim, num_classes),
    )

class FitNet(nn.Module):
    """Couche linéaire simple qui projette les features student → teacher dim."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# ImportanceAdapter — poids adaptatifs par teacher  (Eq. 5-6)
# theta_t ∈ R^d  : facteur latent du teacher t
# nu      ∈ R^d  : paramètre global appris
# gamma_{t,i} = nu^T (theta_t ⊙ delta_i)
# w_{t,i}     = softmax_t(gamma_{t,i})
# ---------------------------------------------------------------------------
class ImportanceAdapter(nn.Module):
    """
    Calcule les poids d'importance w_{t,i} pour chaque teacher et chaque image.
    delta_i = features student de l'image i  (B, d)
    """

    def __init__(self, num_teachers: int, embed_dim: int):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(num_teachers, embed_dim) * 0.01)
        self.nu = nn.Parameter(torch.randn(embed_dim) * 0.01)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """
        delta : (B, d)  features student (après MaxPool ou global avg)
        retourne : (B, num_teachers)  poids normalisés (softmax)
        """
        interaction = self.theta.unsqueeze(0) * delta.unsqueeze(1)   
        scores = (interaction * self.nu).sum(dim=-1)                  
        return torch.softmax(scores, dim=-1)                          

class StudentViT(nn.Module):
    """ViT-Base/16 student : expose forward() et forward_features()."""

    EMBED_DIM = 768 

    def __init__(self, num_classes: int = 2, size_picture: int = 224):
        super().__init__()
        self.resize = Resize((size_picture, size_picture))
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0,      
            global_pool="avg",
        )
        self.head = _make_head(self.EMBED_DIM, num_classes, dropout_rate=0.1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.resize(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))