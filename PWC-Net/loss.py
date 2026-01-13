import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class PWCNetLoss(nn.Module):
    def __init__(self,
                 weights: Dict[int, float] = None):
        super().__init__()
        
        if weights is None:
            self.weights = {
                2: 0.005,
                3: 0.01,
                4: 0.02,
                5: 0.08,
                6: 0.32
            }
        else:
            self.weights = weights
    
    def forward(self, 
                predicted_flows, 
                gt_flow):
        
        total_loss = 0.0
        gt_flow_scaled = gt_flow / 20.0
        
        flow_2, flow_3, flow_4, flow_5, flow_6 = predicted_flows
        flows = [flow_2, flow_3, flow_4, flow_5, flow_6]
        levels = [2, 3, 4, 5, 6]
        
        for flow, level in zip(flows, levels):
            gt_downsampled = F.interpolate(gt_flow_scaled,
                                           size=flow.shape[2:],
                                           mode="bilinear",
                                           align_corners=True)
            
            level_loss = torch.norm(flow - gt_downsampled, p=2, dim=1).mean()
            total_loss += self.weights[level] * level_loss
        
        return total_loss

class PWCNetRobustLoss(nn.Module):
    
    def __init__(self,
                 weights: Dict[int, float] = None,
                 q: float = 0.4,
                 epsilon: float = 0.01):
        super().__init__()
        
        if weights is None:
            self.weights = {
                2: 0.005,
                3: 0.01,
                4: 0.02,
                5: 0.08,
                6: 0.32
            }
        else:
            self.weights = weights
            
        self.q = q
        self.epsilon = epsilon
    
    def forward(self,
                predicted_flows: torch.Tensor,
                gt_flow: torch.Tensor):

        total_loss = 0.0
        
        gt_flow_scaled = gt_flow / 20.0
        
        flow_2, flow_3, flow_4, flow_5, flow_6 = predicted_flows
        flows = [flow_2, flow_3, flow_4, flow_5, flow_6]
        levels = [2, 3, 4, 5, 6]
        
        for flow, level in zip(flows, levels):
            gt_downsampled = F.interpolate(gt_flow_scaled,
                                          size=flow.shape[2:],
                                          mode='bilinear',
                                          align_corners=True)
            
            level_loss = ((torch.abs(flow - gt_downsampled) + self.epsilon) ** self.q).mean()
            
            total_loss += self.weights[level] * level_loss
        
        return total_loss
