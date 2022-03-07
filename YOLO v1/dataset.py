import csv
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image

class VOCDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, nS=7, nB=2, nC=20, transform=False) -> None:
       self.annotations = pd.read_csv(csv_file)
       self.img_dir = img_dir
       self.label_dir = label_dir
       self.nS = nS 
       self.nB = nB
       self.nC = nC
       self.transform = transform
       
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        boxes = []
        
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, w, h = [float(x) if float(x) != int(float(x)) else int(x)
                                           for x in label.replace("\n", "").split()
                                          ]
                boxes.append([class_label, x, y, w, h])
        
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        
        boxes = torch.tensor(boxes) # because transformations are usually done on tensors
        if self.transform:
            image, boxes = self.transform(image, boxes)
        
        # Convert to cells
        # As we are doing it for 1 bounding box so size will be 25
        label_matrix = torch.zeros((self.nS, self.nS, self.nC + 5*self.nB)) # last 5 nodes will not be used (only first 25) 
        
        for box in boxes:
            class_label, x, y, w, h = box.tolist()
            class_label = int(class_label) # making sure class_label is an int
            # x, y are relative to whole image, we have to convert to relative to each cell in an image 
            i, j = int(self.S*y), int(self.S*x) # i respresents cell row, j respresents cell column
            x_cell, y_cell = self.S*x - j , self.S*y - i
            w_cell, h_cell = w*self.nS, h*self.nS
            
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] == 1
                box_coords = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                label_matrix[i, j, 21:25] = box_coords
                label_matrix[i, j, class_label] = 1
                
        return image, label_matrix
