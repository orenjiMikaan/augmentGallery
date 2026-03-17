from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import torch
from torch import nn

from src.models.encoder import build_encoder
from src.models.projector import build_mlp, build_predictor


@torch.no_grad()
def _ema_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)


class BYOL(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet9",
        proj_hidden_dim: int = 512,
        proj_dim: int = 256,
        pred_hidden_dim: int = 512,
    ):
        super().__init__()
        online_enc, feat_dim = build_encoder(backbone)
        self.online_encoder = online_enc
        self.online_projector = build_mlp(feat_dim, proj_hidden_dim, proj_dim)
        self.online_predictor = build_predictor(proj_dim, pred_hidden_dim, proj_dim)

        self.target_encoder = deepcopy(self.online_encoder)
        self.target_projector = deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self, tau: float) -> None:
        _ema_update(self.target_encoder, self.online_encoder, tau)
        _ema_update(self.target_projector, self.online_projector, tau)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Online
        o1 = self.online_encoder(view1)
        o2 = self.online_encoder(view2)
        z1 = self.online_projector(o1)
        z2 = self.online_projector(o2)
        p1 = self.online_predictor(z1)
        p2 = self.online_predictor(z2)

        # Target (stop-grad)
        with torch.no_grad():
            t1 = self.target_encoder(view1)
            t2 = self.target_encoder(view2)
            tz1 = self.target_projector(t1)
            tz2 = self.target_projector(t2)

        return p1, p2, tz1.detach(), tz2.detach()

