from platform import architecture
from tkinter.tix import IMAGE
from turtle import forward
from numpy import isin
import torch
import torch.nn as nn

"""
Architecture Config:
Tuple : (out_channels, kernel_size, stride) # pretty much every Conv uses "same" padding in YOLOv3
List  : ["B", num_repeats] # B for Residual Block
"S"   : Scale Prediction and computing the YOLO loss
"U"   : Upsampling the Feature map and Concatenating with the previous layer(s)
"""
# discerned the config pattern from GitHub src code 
architecture_config = [
    # starting in_channels = 3
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # Upto this layer is Darknet-53, which is trained on ImageNet, rest 53 layers are added for detection head)
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_c, out_c, use_BN_and_activ=True, **kwargs) -> None:
        super().__init__()
        self.Conv      = nn.Conv2d(in_c, out_c, bias= not use_BN_and_activ, **kwargs)
        self.BN        = nn.BatchNorm2d(out_c)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.use_BN_and_activ = use_BN_and_activ
    
    def forward(self, x):
        # because ScalePrediction is in output and Batch Norm & ReLU won't be used in output, but ScalePrediction will use Conv
        return self.LeakyReLU(self.BN(self.Conv(x))) if self.use_BN_and_activ else self.Conv(x)
        

class ResidualBlock(nn.Module):
    # "use_residual" because sometimes residual connections are used and sometimes not in config architecture
    def __init__(self, channels, use_residual=True, num_repeats=1) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += nn.Sequential(
                nn.Conv2d(channels, channels//2, kernel_size=1),
                nn.Conv2d(channels//2, channels, kernel_size=3, padding=1)
            )
            
        self.use_residual = use_residual
        self.num_repeats  = num_repeats
         
    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = layer(x) + x
            else:
                x = layer(x)
            
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_c, num_classes) -> None:
        super().__init__()
        self.prediction = nn.Sequential(
            CNNBlock(in_c, 2*in_c, kernel_size=3, padding=1), 
            # For each cell -> 3 Anchor Boxes | For each Anchor Box -> num_classes we want to predict + Bounding Box coords
            CNNBlock(2*in_c, 3*(num_classes+5), use_BN_and_activ=False, kernel_size=1) 
        )
        
        self.num_classes = num_classes
    
    
    def forward(self, x):
        # (batch_size x 3 x 13(26|52) x 13(26|52) x num_classes+5)
        return self.prediction(x).reshape(x.shape[0], 3, self.num_classes+5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)

class YOLOv3(nn.Module):
    def __init__(self, in_c=3, num_classes=20) -> None:
        super().__init__()
        self.in_c        = in_c
        self.num_classes = num_classes
        self.layers      = self.create_conv_layers()
    
    def forward(self, x):
        outputs = []
        skip_connections = []
        
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            
            x = layer(x)
            
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8 :
                skip_connections.append(x)
            
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, skip_connections[-1]], dim=1)
                
    
    def create_conv_layers(self):
        layers = nn.ModuleList()
        in_c   = self.in_c
        
        for layer in architecture_config:
            if isinstance(layer, tuple):
                out_c, k_size, stride = layer
                layers.append(CNNBlock(in_c, out_c, kernel_size=k_size, stride=stride, padding = 1 if k_size == 3 else 0))
                in_c = out_c
            
            elif isinstance(layer, list):
                num_repeats = layer[1]
                layers.append(ResidualBlock(in_c, num_repeats=num_repeats))
            
            elif isinstance(layer, str):
                if layer == "S":
                    layers += [
                        ResidualBlock(in_c, use_residual=False, num_repeats=1),
                        CNNBlock(in_c, in_c//2, kernel_size=1),
                        ScalePrediction(in_c, num_classes=self.num_classes)
                    ]
                    in_c = in_c//2 # as we are changing in_c and we want to continue from Conv block and not ScalePrediction
                    
                elif layer == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    # concatenate the channels just after Upsampling which ends up preserving the fine-grained features which help in detecting small objects.
                    in_c = in_c * 3 
                    
        return layers
