"""Bounded parameter mapping for the active-region Ca II example."""

import torch
from torch import nn


class MLSIPINN(nn.Module):

    def __init__(self, n_wavelengths, dropout_rate=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_wavelengths, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 13),
        )

    def forward(self, x):
        raw = self.encoder(x)
        log_Sp = torch.sigmoid(raw[:, 0]) * 1 - 0.7
        log_S2 = -0.6 + 0.45 * torch.sigmoid(raw[:, 1])
        log_S1 = -0.6 + 0.45 * torch.sigmoid(raw[:, 2])
        log_S0_raw = -1.1 + 0.75 * torch.sigmoid(raw[:, 3])
        log_S0 = torch.minimum(log_S0_raw, log_S1 - 0.05)
        v0 = 10 * torch.sigmoid(raw[:, 5]) - 5
        v1 = 13 * torch.sigmoid(raw[:, 6]) - 6.5
        WP_LO, WP_HI = (-1.307, -1.271)
        avg_s = 1 * log_Sp
        AVG_LO, AVG_HI = (-0.31, 0.05)
        u = (avg_s - AVG_LO) / (AVG_HI - AVG_LO + 1e-06)
        u = torch.clamp(u, 0.0, 1.0)
        log_wp = WP_LO + (WP_HI - WP_LO) * u
        log_w0 = 0.6 * torch.sigmoid(raw[:, 8]) - 1.1
        log_w1 = 0.6 * torch.sigmoid(raw[:, 9]) - 0.9
        log_eta = 0.15 * torch.sigmoid(raw[:, 11]) + 4.85
        vp = raw[:, 4]
        log_ap = raw[:, 10]
        tau0 = 10 * torch.ones_like(raw[:, 12])
        return torch.stack(
            [
                log_Sp,
                log_S2,
                log_S1,
                log_S0,
                vp,
                v0,
                v1,
                log_wp,
                log_w0,
                log_w1,
                log_ap,
                log_eta,
                tau0,
            ],
            dim=1,
        )
