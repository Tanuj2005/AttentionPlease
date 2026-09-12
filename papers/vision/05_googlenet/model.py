import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    """
    The core building block. Four parallel branches, concatenated on channel dim.

    Branch 1: 1x1 conv
    Branch 2: 1x1 bottleneck -> 3x3 conv
    Branch 3: 1x1 bottleneck -> 5x5 conv
    Branch 4: 3x3 maxpool    -> 1x1 conv

    All branches preserve spatial size (same padding).
    Output channels = b1 + b2_out + b3_out + b4_out

    Args:
        in_ch:    input channels
        b1:       branch1 out_channels  (1x1 conv)
        b2_red:   branch2 reduction     (1x1 bottleneck channels)
        b2_out:   branch2 out_channels  (3x3 conv)
        b3_red:   branch3 reduction     (1x1 bottleneck channels)
        b3_out:   branch3 out_channels  (5x5 conv)
        b4_out:   branch4 out_channels  (1x1 after pool)
    """

    def __init__(
        self,
        in_ch:  int,
        b1:     int,
        b2_red: int, b2_out: int,
        b3_red: int, b3_out: int,
        b4_out: int,
    ) -> None:
        super().__init__()

        # Branch 1: 1x1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, b1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Branch 2: 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, b2_red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(b2_red, b2_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Branch 3: 1x1 -> 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, b3_red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(b3_red, b3_out, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        # Branch 4: 3x3 maxpool -> 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # same spatial size
            nn.Conv2d(in_ch, b4_out, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)  # concat on channel dim


class AuxClassifier(nn.Module):
    """
    Auxiliary classifier — plugged into inception_4a and inception_4d outputs.
    Only used during training; ignored at inference.

    Purpose: inject gradient signal deeper into the network before the
    vanishing gradient problem kills it (no BatchNorm in 2014).
    Weighted at 0.3 in the total loss.
    """

    def __init__(self, in_ch: int, num_classes: int) -> None:
        super().__init__()

        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)            # (B, C, 4, 4) for 14x14 input
        x = self.conv(x)            # (B, 128, 4, 4)
        x = x.view(x.size(0), -1)  # (B, 2048)
        x = self.fc(x)
        return x


class GoogLeNet(nn.Module):
    """
    Input:  (B, 3, 224, 224)
    Output: (B, num_classes)

    Spatial trace:
      Conv1(k=7,s=2,p=3):    -> (B,  64, 112, 112)
      MaxPool(k=3,s=2,p=1):  -> (B,  64,  56,  56)
      Conv2(k=1):             -> (B,  64,  56,  56)
      Conv3(k=3,p=1):         -> (B, 192,  56,  56)
      MaxPool(k=3,s=2,p=1):  -> (B, 192,  28,  28)
      inception_3a:           -> (B, 256,  28,  28)
      inception_3b:           -> (B, 480,  28,  28)
      MaxPool(k=3,s=2,p=1):  -> (B, 480,  14,  14)
      inception_4a:           -> (B, 512,  14,  14)  <- aux1 here
      inception_4b:           -> (B, 512,  14,  14)
      inception_4c:           -> (B, 512,  14,  14)
      inception_4d:           -> (B, 528,  14,  14)  <- aux2 here
      inception_4e:           -> (B, 832,  14,  14)
      MaxPool(k=3,s=2,p=1):  -> (B, 832,   7,   7)
      inception_5a:           -> (B, 832,   7,   7)
      inception_5b:           -> (B,1024,   7,   7)
      AvgPool(k=7):           -> (B,1024,   1,   1)
      Dropout(0.4)
      Linear(1024, num_classes)
    """

    def __init__(self, num_classes: int = 1000, aux_logits: bool = True) -> None:
        super().__init__()
        self.aux_logits = aux_logits

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),   # -> 112x112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),       # -> 56x56
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),       # -> 28x28
        )

        # Inception blocks — channel args from Table 1 of the paper
        # InceptionModule(in_ch, b1, b2_red, b2_out, b3_red, b3_out, b4_out)
        # Output channels = b1 + b2_out + b3_out + b4_out

        self.inception_3a = InceptionModule(192,  64,  96, 128, 16,  32,  32)  # out: 256
        self.inception_3b = InceptionModule(256, 128, 128, 192, 32,  96,  64)  # out: 480
        self.pool3        = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # -> 14x14

        self.inception_4a = InceptionModule(480, 192,  96, 208, 16,  48,  64)  # out: 512
        self.inception_4b = InceptionModule(512, 160, 112, 224, 24,  64,  64)  # out: 512
        self.inception_4c = InceptionModule(512, 128, 128, 256, 24,  64,  64)  # out: 512
        self.inception_4d = InceptionModule(512, 112, 144, 288, 32,  64,  64)  # out: 528
        self.inception_4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)  # out: 832
        self.pool4        = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # -> 7x7

        self.inception_5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)  # out: 832
        self.inception_5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)  # out: 1024

        # Auxiliary classifiers (training only)
        if aux_logits:
            self.aux1 = AuxClassifier(512, num_classes)   # after inception_4a
            self.aux2 = AuxClassifier(528, num_classes)   # after inception_4d

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(p=0.4),
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...] | torch.Tensor:
        x = self.stem(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.pool3(x)

        x = self.inception_4a(x)
        aux1_out = self.aux1(x) if (self.aux_logits and self.training) else None

        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        aux2_out = self.aux2(x) if (self.aux_logits and self.training) else None

        x = self.inception_4e(x)
        x = self.pool4(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)

        x = self.head(x)
        x = x.view(x.size(0), -1)  # (B, 1024)
        x = self.fc(x)

        if self.aux_logits and self.training:
            return x, aux1_out, aux2_out
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GoogLeNet(num_classes=10, aux_logits=True).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Training mode — returns (main, aux1, aux2)
    model.train()
    x   = torch.randn(2, 3, 224, 224).to(device)
    out = model(x)
    print(f"Training output: main={tuple(out[0].shape)}, aux1={tuple(out[1].shape)}, aux2={tuple(out[2].shape)}")

    # Eval mode — returns main only
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Eval output:     {tuple(out.shape)}")
    print(f"Trainable parameters: {params:,}")   # expect ~6.6M (vs AlexNet's 62M)