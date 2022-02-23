import torch

def IoU(boxes_preds, boxes_labels, boxes_format="midpoint"):
    """
    Calculates intersection over union
    
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
        
    Returns:
        tensor: Intersection over union for all examples
    """
    
    if boxes_format == "midpoint":
        box1_x1 = (boxes_preds[..., 0:1] - boxes_preds[..., 2:3])/2 # (box1_x1 - box1_x2)/2
        box1_y1 = (boxes_preds[..., 1:2] - boxes_preds[..., 3:4])/2 # (box1_y1 - box1_y2)/2
        box1_x2 = (boxes_preds[..., 0:1] + boxes_preds[..., 2:3])/2 # (box1_x1 + box1_x2)/2
        box1_y2 = (boxes_preds[..., 1:2] + boxes_preds[..., 3:4])/2 # (box1_y1 + box1_y2)/2
        
        box2_x1 = (boxes_labels[..., 0:1] - boxes_labels[..., 2:3])/2 # (box1_x1 - box1_x2)/2
        box2_y1 = (boxes_labels[..., 1:2] - boxes_labels[..., 3:4])/2 # (box1_y1 - box1_y2)/2
        box2_x2 = (boxes_labels[..., 0:1] + boxes_labels[..., 2:3])/2 # (box1_x1 + box1_x2)/2
        box2_y2 = (boxes_labels[..., 1:2] + boxes_labels[..., 3:4])/2 # (box1_y1 + box1_y2)/2
        
    elif boxes_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case boxes do not intersect, then intersection has to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def NMS(bboxes, iou_threshold, prob, boxes_format="corners"):
    """
    Non Max Suppression
    
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
