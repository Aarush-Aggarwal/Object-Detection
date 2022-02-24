import torch
import torch.nn as nn
from utils import IoU

class YoloLoss(nn.Module):
    def __init__(self, nS=7, nB=2, nC=20) -> None:
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.nS = nS
        self.nB = nB
        self.nC = nC
        
        self.lambda_coord = 5
        self.lambda_no_obj = 0.5
        
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.nS, self.nS, self.nC + self.nB*5)
        
        iou_b1 = IoU(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = IoU(predictions[..., 26:30], target[..., 21:25])
        IoUs   = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        
        _, resp_box_idx = torch.max(IoUs, dim=0)
        obj_isExist = target[..., 20].unsqueeze(3) # Iobj_i (Identity of object i). True, if object in cell i
        
        
        # Bounding Box Coordinates
        box_preds = obj_isExist * (resp_box_idx * predictions[..., 26:30] 
                                   + (1-resp_box_idx) * predictions[..., 21:25])
        box_targets = obj_isExist * target[..., 21:25]
        
        box_preds[..., 2:4] = torch.sign(box_preds[..., 2:4]) * torch.sqrt(torch.abs(box_preds[..., 2:4]))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
                            torch.flatten(box_preds, end_dim=-2), 
                            torch.flatten(box_targets, end_dim=-2)
                           )
        
        # Object Loss 
        # (N*S*S)
        predicted_score = resp_box_idx * predictions[..., 20:21] + (1-resp_box_idx) * predictions[..., 25:26]
        obj_loss = self.mse(
                            torch.flatten(obj_isExist * predicted_score), 
                            torch.flatten(obj_isExist * target[25:26])
                           ) 
       
        # No Object Loss
        no_obj_loss = self.mse(
                               torch.flatten((1-obj_isExist) * predictions[..., 20:21], start_dim=1), 
                               torch.flatten((1-obj_isExist) * target[..., 20:21], start_dim=1) 
                              ) 
        no_obj_loss += self.mse(
                                torch.flatten((1-obj_isExist) * predictions[..., 25:26], start_dim=1), 
                                torch.flatten((1-obj_isExist) * target[..., 20:21], start_dim=1) 
                               ) 
        
        # Class Loss
        class_loss = self.mse(
                              torch.flatten(obj_isExist * predictions[..., :20], end_dim=-2), 
                              torch.flatten(obj_isExist * target[..., :20], end_dim=-2)
                             )
        
        loss = self.lambda_coord * box_loss + obj_loss + self.lambda_no_obj * no_obj_loss + class_loss
        
        return loss
