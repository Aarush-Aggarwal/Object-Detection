import torch
import torch.nn as nn

""" 
Architecture config:
Tuple is structured as (filters, kernel_size, stride, padding) 
"M" is the maxpooling layer with stride 2x2 and kernel 2x2
List is structured as tuples and an int representing number of repeats
"""

architecture_config = [
    (64, 7, 2, 3),
    "M",
    (192, 3, 1, 1),
    "M",
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    "M",
    [(256, 1, 1, 0), (512, 3, 1, 1), 4],
    (512, 1, 1, 0),
    (1024, 3, 1, 1),
    "M",
    [(512, 1, 1, 0), (1024, 3, 1, 1), 2],
    (1024, 3, 1, 1),
    (1024, 3, 2, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_c, out_c, **kwargs) -> None:
        super(CNNBlock, self).__init__()
        self.Conv      = nn.Conv2d(in_c, out_c, bias=False, **kwargs)
        self.BatchNorm = nn.BatchNorm2d(out_c)
        self.leakyReLU = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.leakyReLU(self.BatchNorm(self.Conv(x)))
    
    
class Yolov1(nn.Module):
    def __init__(self, in_c=3, **kwargs) -> None:
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_c    = in_c
        self.Darknet = self.create_conv_layers(self.architecture)
        self.FCs     = self.create_fcs_layers(**kwargs)
      
    def forward(self, x):
        x = self.Darknet(x)
        return self.FCs(torch.flatten(x, start_dim=1))
    
    def create_conv_layers(self, architecture):
        layers = []
        in_c = self.in_c
        
        for layer in architecture_config:
            if type(layer) == tuple:
                layers += [CNNBlock(in_c, layer[0], kernel_size=layer[1], stride=layer[2], padding=layer[3])]
                in_c = layer[0]
                
            elif type(layer) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                
            elif type(layer) == list:
                conv1 = layer[0]
                conv2 = layer[1]
                num_repeats = layer[2]
                for _ in range(num_repeats):
                    layers += [CNNBlock(in_c, conv1[0], kernel_size=conv1[1], stride=conv1[2], padding=conv1[3])]
                    layers += [CNNBlock(conv1[0], conv2[0], kernel_size=conv2[1], stride=conv2[2], padding=conv2[3])]
                    in_c = conv2[0]
        
        return nn.Sequential(*layers)
    
    def create_fcs_layers(self, split_size, num_boxes, num_classes):
        nS, nB, nC = split_size, num_boxes, num_classes
        return nn.Sequential(nn.Flatten(), nn.Linear(1024 * nS*nS, 4096), nn.Dropout(), nn.LeakyReLU(0.1), nn.Linear(4096, nS*nS * (nC + nB*5)))
