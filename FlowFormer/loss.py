import torch
import torch.nn as nn

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
