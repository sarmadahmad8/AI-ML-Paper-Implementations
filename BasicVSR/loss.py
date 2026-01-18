import torch
import torch.nn as nn
import torch.nn.functional as F

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

class GradientandCharbonnierLoss(nn.Module):
    def __init__(self,
                 eps: float = 1e-3,
                 reduction: str = "mean",
                 charbonnier_weight: float = 0.8,
                 gradient_weight: float = 0.2):
        super().__init__()
        self.reduction = reduction
        self.charbonnier_weight = charbonnier_weight
        self.gradient_weight = gradient_weight
        
        self.charbonnier_loss = CharbonnierLoss(eps = eps,
                                                reduction = reduction)
        # Sobel filters for edge detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, pred, target):
        self.sobel_x = self.sobel_x.to(pred.device)
        self.sobel_y = self.sobel_y.to(pred.device)
        
        # Convert to grayscale if needed
        if pred.shape[1] == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            pred_gray = pred
            target_gray = target
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        
        target_grad_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, self.sobel_y, padding=1)

        charbonnier_loss = self.charbonnier_loss(pred, target)
        # L1 loss on gradients
        gradient_loss = F.l1_loss(pred_grad_x, target_grad_x, reduction = self.reduction) + F.l1_loss(pred_grad_y, target_grad_y, reduction = self.reduction)

        loss = self.charbonnier_weight * charbonnier_loss + self.gradient_weight * gradient_loss
        
        return loss
