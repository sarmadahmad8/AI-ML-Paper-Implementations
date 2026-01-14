import torch
import torch.nn as nn
from typing import Tuple

class FRVSRLoss(nn.Module):

    def __init__(self,
                 lr_weight: float = 1.0,
                 hr_weight: float = 1.0):

        super().__init__()

        self.lr_weight = lr_weight
        self.hr_weight = hr_weight
        self.l2_loss = nn.MSELoss()

    def forward(self,
                preds: Tuple[torch.Tensor, torch.Tensor],
                targets: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:

        lr_preds, hr_preds = preds
        lr_targets, hr_targets = targets

        lr_loss = self.l2_loss(lr_preds, lr_targets)
        hr_loss = self.l2_loss(hr_preds, hr_targets)

        total_loss = self.hr_weight * hr_loss + self.lr_weight * lr_loss

        return total_loss
