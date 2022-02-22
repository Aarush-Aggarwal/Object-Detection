import torch

from iou import IoU

def NMS(bboxes, iou_threshold, prob, boxes_format="corners"):
    """
    Parameters:
        bboxes (list): list of lists 
        containing all bboxes with each bboxes specified as [class_pred, prob_score, x1, y1, x2, y2]
        
        iou_threshold (float): threshold where predicted bboxes is correct
        
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        
        box_format (str): "midpoint" or "corners" used to specify bboxes
        
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """
    assert type(bboxes) == list
    
    bboxes = [box for box in bboxes if box[1] > prob]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    
    bboxes_after_nms = []
    
    while bboxes:
        highest_prob_bbox = bboxes.pop(0)
        bboxes_after_nms.append(highest_prob_bbox)
        
        bboxes = [box for box in bboxes 
                  if box[0] != highest_prob_bbox[0] 
                  or IoU(torch.tensor(box[2:]), torch.tensor(highest_prob_bbox[2:]))
                  ]
        
    return bboxes_after_nms
