import torch
import torch.nn as nn

class AlexNet( nn.Module ):
    def __init__( self, num_classes: int = 1000, dropout: float = 0.5 ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d( in_channels= 3, out_channels= 96, kernel_size= 11, stride= 4, padding=2  ),
            nn.ReLU( inplace= True ),
            nn.MaxPool2d( kernel_size= 3, stride= 2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d( in_channels= 96, out_channels= 256, kernel_size= 5, stride= 1, padding=2  ),
            nn.ReLU( inplace= True ),
            nn.MaxPool2d( kernel_size= 3, stride= 2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d( in_channels= 256, out_channels= 384, kernel_size= 3, stride= 1, padding=1  ),
            nn.ReLU( inplace= True ),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d( in_channels= 384, out_channels= 384, kernel_size= 3, stride= 1, padding=1  ),
            nn.ReLU( inplace= True ),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d( in_channels= 384, out_channels= 256, kernel_size= 3, stride= 1, padding=1  ),
            nn.ReLU( inplace= True ),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6,6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward( self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
    

        



