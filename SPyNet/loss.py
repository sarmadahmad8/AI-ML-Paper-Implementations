import torch
import torch.nn as nn

class EndPointErrorLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self,
                preds: torch.Tensor,
                targets: torch.Tensor,
                valid_mask: torch.Tensor = None):

        horizontal_difference = preds[:, 0] - targets[:, 0]
        vertical_difference = preds[:, 1] - targets[:, 1]

        epe = (horizontal_difference ** 2 + vertical_difference ** 2) ** 0.5

        if valid_mask is not None:
            epe = epe[valid_mask]

        return epe.mean()
