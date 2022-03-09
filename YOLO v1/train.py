from numpy import append
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Yolov1
from loss import YoloLoss
from utils import IoU, NMS, mAP, load_checkpoint, get_bboxes
from dataset import VOCDataset

seed = 123
torch.manual_seed(seed)

# Hyperparams, other vars, etc.
LEARNING_RATE = 2e-5
DEVICE = "cpu"
BATCH_SIZE = 8
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/label"

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader)
    running_loss = []
    
    for batch_idx, (data, target) in enumerate(loop):
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        loss = loss_fn(output, target)
        running_loss.append(loss.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Upadte the progress bar
        loop.set_postfix(loss=loss.item())
    
    print(f"Mean loss : {sum(running_loss)/len(running_loss)}")
    
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes
        
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()
    
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
        
    train_dataset = VOCDataset("~/Desktop/YOLO/YOLO v1/100examples.csv", IMG_DIR, LABEL_DIR, transform=transform)
    test_dataset  = VOCDataset("~/Desktop/YOLO/YOLO v1/test.csv", IMG_DIR, LABEL_DIR, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True) 
    
    for _ in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)
        meanAP = mAP(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        print(f"Train mAP: {meanAP}")
        train_fn(train_loader, model, optimizer, loss_fn)

if __name__ == "__main__":
    main()
