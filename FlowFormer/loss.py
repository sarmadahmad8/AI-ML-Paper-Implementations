import torch
import torch.nn as nn
import torch.nn.functional as F

class RAFTLoss(nn.Module):
    def __init__(self, gamma=0.8, max_flow=400):
        super(RAFTLoss, self).__init__()
        self.gamma = gamma
        self.max_flow = max_flow
    
    def forward(self, flow_preds, flow_gt):
    
        n_predictions = len(flow_preds)
        flow_loss = 0.0
        
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (mag < self.max_flow)
        
        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()
        
        epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]
        
        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }
        
        return flow_loss, metrics

class RAFTLossWithGradient(nn.Module):
    def __init__(self, gamma=0.8, max_flow=400, gradient_weight=0.5):
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow
        self.gradient_weight = gradient_weight
        
    def compute_gradient_loss(self, flow_pred, flow_gt, valid):
        # First-order gradients (sharper edges)
        pred_dx = flow_pred[:, :, :, 1:] - flow_pred[:, :, :, :-1]
        pred_dy = flow_pred[:, :, 1:, :] - flow_pred[:, :, :-1, :]
        
        gt_dx = flow_gt[:, :, :, 1:] - flow_gt[:, :, :, :-1]
        gt_dy = flow_gt[:, :, 1:, :] - flow_gt[:, :, :-1, :]
        
        # Crop valid mask
        valid_dx = valid[:, :, :-1]
        valid_dy = valid[:, :-1, :]
        
        grad_loss = (
            (valid_dx[:, None] * (pred_dx - gt_dx).abs()).mean() +
            (valid_dy[:, None] * (pred_dy - gt_dy).abs()).mean()
        )
        
        return grad_loss
    
    def forward(self, flow_preds, flow_gt):
        n_predictions = len(flow_preds)
        flow_loss = 0.0
        
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (mag < self.max_flow)
        
        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            
            # L1 loss
            l1_loss = (flow_preds[i] - flow_gt).abs()
            l1_loss = (valid[:, None] * l1_loss).mean()
            
            # Gradient loss
            grad_loss = self.compute_gradient_loss(
                flow_preds[i], flow_gt, valid
            )
            
            # Combine
            flow_loss += i_weight * (l1_loss + self.gradient_weight * grad_loss)
        
        # Metrics
        epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]
        
        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }
        
        return flow_loss, metrics

class SharpFlowLoss(nn.Module):
    def __init__(self, 
                 gamma=0.8, 
                 max_flow=400,
                 # Loss weights
                 l1_weight=1.0,
                 gradient_weight=0.5,
                 census_weight=0.3,
                 edge_aware_weight=0.2):
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow
        
        # Loss component weights
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.census_weight = census_weight
        self.edge_aware_weight = edge_aware_weight
        
        # Sobel kernels for gradient computation
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).reshape(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).reshape(1, 1, 3, 3))
    
    def compute_gradient(self, flow):
        """
        Compute spatial gradients of flow
        Args:
            flow: [B, 2, H, W]
        Returns:
            grad_x, grad_y: [B, 2, H, W] each
        """
        B, C, H, W = flow.shape
        
        # Apply Sobel filter to each flow channel
        flow_dx = F.conv2d(
            flow.reshape(B * C, 1, H, W),
            self.sobel_x,
            padding=1
        ).reshape(B, C, H, W)
        
        flow_dy = F.conv2d(
            flow.reshape(B * C, 1, H, W),
            self.sobel_y,
            padding=1
        ).reshape(B, C, H, W)
        
        return flow_dx, flow_dy
    
    def gradient_loss(self, flow_pred, flow_gt, valid):
        """
        Encourage sharp gradients where GT has sharp gradients
        """
        # Compute gradients
        pred_dx, pred_dy = self.compute_gradient(flow_pred)
        gt_dx, gt_dy = self.compute_gradient(flow_gt)
        
        # L1 loss on gradients (preserves sharp edges better than L2)
        grad_loss = (pred_dx - gt_dx).abs() + (pred_dy - gt_dy).abs()
        
        # Apply valid mask
        grad_loss = (valid[:, None] * grad_loss).mean()
        
        return grad_loss
    
    def census_transform(self, flow, window_size=3):
        """
        Census transform for better structure preservation
        Compares each pixel to its neighborhood
        """
        B, C, H, W = flow.shape
        pad = window_size // 2
        
        # Pad the flow
        flow_padded = F.pad(flow, (pad, pad, pad, pad), mode='replicate')
        
        # Get center pixel
        center = flow
        
        # Compare with neighbors
        census = []
        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = flow_padded[:, :, 
                                      pad+dy:pad+dy+H, 
                                      pad+dx:pad+dx+W]
                census.append((center < neighbor).float())
        
        return torch.cat(census, dim=1)
    
    def census_loss(self, flow_pred, flow_gt, valid):
        """
        Compare census transforms (captures local structure)
        """
        census_pred = self.census_transform(flow_pred)
        census_gt = self.census_transform(flow_gt)
        
        # Hamming distance
        census_diff = (census_pred != census_gt).float()
        census_loss = (valid[:, None] * census_diff).mean()
        
        return census_loss
    
    def edge_aware_smoothness(self, flow_pred, image):
        """
        Encourage smoothness EXCEPT at image edges
        (Prevents blurring motion boundaries)
        
        Args:
            flow_pred: [B, 2, H, W]
            image: [B, 3, H, W] - source image
        """
        # Compute image gradients
        img_gray = image.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        img_dx = F.conv2d(img_gray, self.sobel_x, padding=1)
        img_dy = F.conv2d(img_gray, self.sobel_y, padding=1)
        
        # Image edge magnitude
        edge_mag = torch.sqrt(img_dx ** 2 + img_dy ** 2 + 1e-8)
        
        # Flow gradients
        flow_dx, flow_dy = self.compute_gradient(flow_pred)
        flow_grad_mag = torch.sqrt(
            flow_dx ** 2 + flow_dy ** 2 + 1e-8
        ).sum(dim=1, keepdim=True)
        
        # Weight: high where image is smooth, low at edges
        # (Allow sharp flow at image edges, penalize elsewhere)
        weight = torch.exp(-edge_mag / 0.1)
        
        # Smoothness loss
        smooth_loss = (weight * flow_grad_mag).mean()
        
        return smooth_loss
    
    def forward(self, flow_preds, flow_gt, source_image=None):
        """
        Args:
            flow_preds: List of [B, 2, H, W] predictions
            flow_gt: [B, 2, H, W] ground truth
            source_image: [B, 3, H, W] source image (for edge-aware loss)
        """
        n_predictions = len(flow_preds)
        
        # Validity mask
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        valid = (mag < self.max_flow)
        
        total_loss = 0.0
        
        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            flow_pred = flow_preds[i]
            
            # 1. Standard L1 loss
            l1_loss = (flow_pred - flow_gt).abs()
            l1_loss = (valid[:, None] * l1_loss).mean()
            
            # 2. Gradient loss (sharpness)
            grad_loss = self.gradient_loss(flow_pred, flow_gt, valid)
            
            # 3. Census loss (structure preservation)
            if self.census_weight > 0:
                c_loss = self.census_loss(flow_pred, flow_gt, valid)
            else:
                c_loss = 0.0
            
            # 4. Edge-aware smoothness (only if image provided)
            if self.edge_aware_weight > 0 and source_image is not None:
                # Only apply to final prediction to avoid over-smoothing
                if i == n_predictions - 1:
                    smooth_loss = self.edge_aware_smoothness(
                        flow_pred, source_image
                    )
                else:
                    smooth_loss = 0.0
            else:
                smooth_loss = 0.0
            
            # Combine losses
            iter_loss = (
                self.l1_weight * l1_loss +
                self.gradient_weight * grad_loss +
                self.census_weight * c_loss +
                self.edge_aware_weight * smooth_loss
            )
            
            total_loss += i_weight * iter_loss
        
        # Compute metrics on final prediction
        epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]
        
        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }
        
        return total_loss, metrics
