import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-3,
                 reduction: str = "mean"):
        
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self,
                preds: torch.Tensor,
                targets: torch.Tensor):

        difference_square = (preds - targets) ** 2
        eps_square = self.eps ** 2

        loss = torch.sqrt(difference_square + eps_square)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss.mean()
