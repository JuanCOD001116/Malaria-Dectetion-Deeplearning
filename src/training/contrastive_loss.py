"""
Supervised Contrastive Loss (Khosla et al. 2020, eq. 2).

L = -Σ_i (1/|P(i)|) Σ_{p∈P(i)} log(
        exp(z_i · z_p / τ) / Σ_{a∈A(i)} exp(z_i · z_a / τ)
    )

donde:
  P(i) = índices de ejemplos con la misma label que i (positivos)
  A(i) = todos los índices del batch excepto i
  τ    = temperatura (default 0.07)
  z    = vectores l2-normalizados (similitud coseno)

Referencia: https://arxiv.org/abs/2004.11362
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (N, D) — proyecciones l2-normalizadas del batch
            labels:   (N,)   — etiquetas enteras (0 o 1)

        Returns:
            Scalar loss.
        """
        device = features.device
        N = features.shape[0]

        if N < 2:
            raise ValueError("SupConLoss requiere al menos 2 muestras por batch.")

        # Matriz de similitud coseno (N, N) escalada por temperatura
        sim = torch.matmul(features, features.T) / self.temperature  # (N, N)

        # Máscara diagonal: excluir i == i
        diag_mask = ~torch.eye(N, dtype=torch.bool, device=device)

        # Máscara de positivos: i y j tienen la misma label, y i != j
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).bool() & diag_mask  # (N, N)

        # Verificar que hay al menos un par positivo en el batch
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=device)

        # Numerador: exp(sim_ij / τ) para pares positivos
        # Denominador: Σ_{a ≠ i} exp(sim_ia / τ)
        # Estabilización numérica: restar el máximo por fila antes de exp
        sim_max, _ = (sim * diag_mask.float() - (1 - diag_mask.float()) * 1e9).max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim) * diag_mask.float()       # (N, N) — cero en diagonal
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)  # (N, N)

        # Promedio sobre positivos por fila
        n_positives = pos_mask.float().sum(dim=1)          # (N,)
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / (n_positives + 1e-8)

        # Solo promediar filas que tienen al menos un positivo
        valid_rows = n_positives > 0
        loss = -mean_log_prob_pos[valid_rows].mean()

        return loss
