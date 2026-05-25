import torch
import torch.nn as nn

class LeNet5( nn.Module ):
    def __init__( self, num_classes: int = 10 ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d( in_channels= 1, out_channels= 6, kernel_size= 5, stride= 1, padding= 0 ),
            nn.ReLU(),
            nn.MaxPool2d( kernel_size= 2, stride= 2),

        )

        self.block2 = nn.Sequential(
            nn.Conv2d( in_channels= 6, out_channels= 16, kernel_size= 5, stride= 1, padding=0 ),
            nn.ReLU(),
            nn.MaxPool2d( kernel_size= 2, stride= 2 ),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )
    

    def forward( self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

