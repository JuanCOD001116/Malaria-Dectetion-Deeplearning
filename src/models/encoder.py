"""
Encoder contrastivo basado en ResNet18 preentrenado en ImageNet.

Arquitectura:
  backbone ResNet18  → 512 features
  encoder_head       → 1024 dims  ← embedding final (downstream)
  proj_head          → 128 dims   ← solo se usa durante SupCon training, se descarta después

La proj_head se descarta post-training porque los embeddings de 1024 de encoder_head
tienen mayor poder representacional al no estar colapsados a 128 dims.
Esto es práctica estándar en SimCLR/SupCon (Chen et al. 2020, Khosla et al. 2020).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class ContrastiveEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 1024,
        proj_dim: int = 128,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # Backbone: ResNet18 sin la capa fc original
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tvm.resnet18(weights=weights)
        backbone_out_dim = resnet.fc.in_features  # 512 para ResNet18
        resnet.fc = nn.Identity()
        self.backbone = resnet

        # Encoder head: 512 → 1024 (embedding final downstream)
        self.encoder_head = nn.Sequential(
            nn.Linear(backbone_out_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Projection head: 1024 → 512 → 128 (solo para pérdida contrastiva)
        self.proj_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.BatchNorm1d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim // 2, proj_dim),
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        h = self.backbone(x)         # (B, 512)
        z = self.encoder_head(h)     # (B, 1024) — embedding final

        if return_embedding:
            return z                 # sin normalizar → para modelos clásicos

        p = self.proj_head(z)        # (B, 128) — proyección para SupCon loss
        return F.normalize(p, dim=1) # l2-normalizado → similitud coseno

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias explícito para extracción de embeddings post-training."""
        with torch.no_grad():
            return self.forward(x, return_embedding=True)
