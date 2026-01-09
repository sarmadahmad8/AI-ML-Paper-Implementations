import torch
import torch.nn as nn
import torch.nn.functional as F

class PWCNetLoss(nn.Module):
    def __init__(self, weights=None, gamma=0.0004):
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
            
        self.gamma = gamma
    
    def forward(self, 
                predicted_flows,  # (flow_2, flow_3, flow_4, flow_5, flow_6)
                gt_flow,          # original ground truth
                model_params):
        
        total_loss = 0.0
        
        # Scale ground truth DOWN by 20 (not up!)
        gt_flow_scaled = gt_flow / 20.0
        
        flow_2, flow_3, flow_4, flow_5, flow_6 = predicted_flows
        flows = [flow_2, flow_3, flow_4, flow_5, flow_6]
        levels = [2, 3, 4, 5, 6]
        
        for flow, level in zip(flows, levels):
            # Downsample scaled GT to pyramid level
            downsample_factor = 2 ** level
            gt_downsampled = F.avg_pool2d(gt_flow_scaled, 
                                          kernel_size=downsample_factor, 
                                          stride=downsample_factor)
            
            # Match sizes if needed
            if gt_downsampled.shape != flow.shape:
                gt_downsampled = F.interpolate(gt_downsampled, 
                                              size=flow.shape[2:], 
                                              mode='bilinear', 
                                              align_corners=False)
            
            # L2 loss
            level_loss = torch.norm(flow - gt_downsampled, p=2, dim=1).mean()
            total_loss += self.weights[level] * level_loss
        
        # Regularization
        reg_loss = 0.0
        for param in model_params:
            reg_loss += torch.norm(param, p=2)
        
        total_loss += self.gamma * reg_loss
        
        return total_loss
