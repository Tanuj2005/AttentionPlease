"""
VGGNet — Simonyan & Zisserman (2014)
"Very Deep Convolutional Networks for Large-Scale Image Recognition"
https://arxiv.org/abs/1409.1556

Key insight: replace large kernels (11x11, 7x7, 5x5) with stacks of 3x3 convs.
Two 3x3 convs have the same receptive field as one 5x5, three have the same
as one 7x7 — but with fewer parameters and more non-linearities between them.

The paper defines 6 configurations (A, A-LRN, B, C, D, E).
D = VGG-16 and E = VGG-19 are the ones used everywhere.
"""

import torch
import torch.nn as nn
from typing import Union


# ---------------------------------------------------------------------------
# Config table — directly from Table 1 of the paper.
# int  = Conv2d with that many out_channels (k=3, s=1, p=1 always)
# 'M' = MaxPool2d(k=2, s=2)
# ---------------------------------------------------------------------------

CONFIGS: dict[str, list[Union[int, str]]] = {
    "A":    [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],            # 11 layers
    "B":    [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],   # 13 layers
    "D":    [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],         # VGG-16
    "E":    [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],  # VGG-19
}


def make_features(config: list[Union[int, str]], batch_norm: bool) -> nn.Sequential:
    """
    Build the conv feature extractor from a config list.

    All conv layers share: kernel=3, stride=1, padding=1 (same padding).
    This is the central design decision of VGG — only 3x3 convs, always.

    Args:
        config:     list of ints (out_channels) and 'M' (MaxPool) markers
        batch_norm: if True, insert BN after every conv (before ReLU)

    Returns:
        nn.Sequential of conv/bn/relu/pool layers
    """
    layers: list[nn.Module] = []
    in_channels = 3

    for entry in config:
        if entry == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            out_channels = int(entry)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

    return nn.Sequential(*layers)


class VGG(nn.Module):
    """
    Generic VGG class — instantiate any configuration from the paper.

    Input:  (B, 3, 224, 224)
    Output: (B, num_classes)

    Spatial dimension trace (all configs share the same pooling structure):
      After pool1: 112x112
      After pool2:  56x56
      After pool3:  28x28
      After pool4:  14x14
      After pool5:   7x7
      AdaptiveAvgPool -> 7x7 (already 7x7, no-op for 224 input)
      Flatten: 512 * 7 * 7 = 25088

    The 3x3 convs between pools don't change spatial size (same padding).
    Only the 5 MaxPool(2,2) layers halve dimensions: 224->112->56->28->14->7.
    """

    def __init__(
        self,
        config:      list[Union[int, str]],
        batch_norm:  bool = False,
        num_classes: int  = 1000,
        dropout:     float = 0.5,
    ) -> None:
        super().__init__()

        self.features   = make_features(config, batch_norm)
        self.avgpool    = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)      # (B, 512*7*7)
        x = self.classifier(x)
        return x

    def _init_weights(self) -> None:
        """
        Paper section 3.1: weights initialized from N(0, 0.01), biases to 0.
        For BN variants, gamma=1 and beta=0 by default in PyTorch — no change needed.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Named constructors — the ones people actually use
# ---------------------------------------------------------------------------

def vgg11(num_classes: int = 1000, dropout: float = 0.5) -> VGG:
    return VGG(CONFIGS["A"], batch_norm=False, num_classes=num_classes, dropout=dropout)

def vgg11_bn(num_classes: int = 1000, dropout: float = 0.5) -> VGG:
    return VGG(CONFIGS["A"], batch_norm=True, num_classes=num_classes, dropout=dropout)

def vgg13(num_classes: int = 1000, dropout: float = 0.5) -> VGG:
    return VGG(CONFIGS["B"], batch_norm=False, num_classes=num_classes, dropout=dropout)

def vgg13_bn(num_classes: int = 1000, dropout: float = 0.5) -> VGG:
    return VGG(CONFIGS["B"], batch_norm=True, num_classes=num_classes, dropout=dropout)

def vgg16(num_classes: int = 1000, dropout: float = 0.5) -> VGG:
    return VGG(CONFIGS["D"], batch_norm=False, num_classes=num_classes, dropout=dropout)

def vgg16_bn(num_classes: int = 1000, dropout: float = 0.5) -> VGG:
    return VGG(CONFIGS["D"], batch_norm=True, num_classes=num_classes, dropout=dropout)

def vgg19(num_classes: int = 1000, dropout: float = 0.5) -> VGG:
    return VGG(CONFIGS["E"], batch_norm=False, num_classes=num_classes, dropout=dropout)

def vgg19_bn(num_classes: int = 1000, dropout: float = 0.5) -> VGG:
    return VGG(CONFIGS["E"], batch_norm=True, num_classes=num_classes, dropout=dropout)


# ---------------------------------------------------------------------------
# CIFAR variant — 32x32 input
# ---------------------------------------------------------------------------

class VGGSmall(nn.Module):
    """
    VGG-style network for 32x32 input (CIFAR-10/100).

    Can't use the full VGG configs directly — 5 MaxPools on a 32x32 input
    would reduce spatial dims to 32/2^5 = 1x1 before the classifier,
    leaving no spatial information. We use 3 pools instead.

    Spatial trace (VGG-16 style config):
      Input:          32x32
      After pool1:    16x16
      After pool2:     8x8
      After pool3:     4x4
      AdaptiveAvgPool: 4x4  (no-op)
      Flatten: 512 * 4 * 4 = 8192
    """

    # 3-pool version of VGG-16's conv config
    CIFAR_CONFIG: list[Union[int, str]] = [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512,       # no pool — spatial is already 4x4
        512, 512, 512,       # no pool
    ]

    def __init__(
        self,
        batch_norm:  bool  = True,
        num_classes: int   = 10,
        dropout:     float = 0.5,
    ) -> None:
        super().__init__()

        self.features   = make_features(self.CIFAR_CONFIG, batch_norm)
        self.avgpool    = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs_to_check: list[tuple[str, VGG]] = [
        ("VGG-11",    vgg11()),
        ("VGG-13",    vgg13()),
        ("VGG-16",    vgg16()),
        ("VGG-16-BN", vgg16_bn()),
        ("VGG-19",    vgg19()),
    ]

    x_imgnet = torch.randn(1, 3, 224, 224)
    print(f"{'Model':<12}  {'Params':>12}  {'Output'}")
    print("-" * 50)
    for name, model in configs_to_check:
        model.eval()
        out    = model(x_imgnet)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:<12}  {params:>12,}  {tuple(out.shape)}")

    print()
    small = VGGSmall(batch_norm=True, num_classes=10)
    x_cf  = torch.randn(1, 3, 32, 32)
    out_s = small(x_cf)
    p_s   = sum(p.numel() for p in small.parameters() if p.requires_grad)
    print(f"{'VGGSmall':<12}  {p_s:>12,}  {tuple(out_s.shape)}")