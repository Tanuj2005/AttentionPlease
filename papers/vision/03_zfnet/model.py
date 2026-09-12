
import torch
import torch.nn as nn


class ZFNet(nn.Module):
    """
    Input:  (B, 3, 224, 224)
    Output: (B, num_classes)

    Dimension trace (H only, W is symmetric):
      Conv1(k=7,s=2,p=1): (224+2-7)/2+1 = 110  -> (B, 96,  110, 110)
      MaxPool(k=3,s=2):   (110-3)/2+1   = 54   -> (B, 96,   54,  54)
      Conv2(k=5,s=2,p=0): (54-5)/2+1    = 25   -> (B, 256,  25,  25)
      MaxPool(k=3,s=2):   (25-3)/2+1    = 12   -> (B, 256,  12,  12)
      Conv3(k=3,s=1,p=1): same padding        -> (B, 384,  12,  12)
      Conv4(k=3,s=1,p=1): same padding        -> (B, 384,  12,  12)
      Conv5(k=3,s=1,p=1): same padding        -> (B, 256,  12,  12)
      MaxPool(k=3,s=2):   (12-3)/2+1    = 5   -> (B, 256,   5,   5)
      AdaptiveAvgPool((6,6))                  -> (B, 256,   6,   6)  <- forces 6x6 for classifier
      Flatten: 256*6*6 = 9216
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Conv1 — ZFNet change: 11x11 s=4 -> 7x7 s=2
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),     # -> (B, 96,  110, 110)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),         # -> (B, 96,   54,  54)

            # Conv2 — ZFNet change: s=1 -> s=2
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=0),   # -> (B, 256,  26,  26)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),         # -> (B, 256,  12,  12)

            # Conv3-5 — identical to AlexNet
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # -> (B, 384,  12,  12)
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # -> (B, 384,  12,  12)
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # -> (B, 256,  12,  12)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),         # -> (B, 256,   6,   6)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ZFNetSmall(nn.Module):
    """
    ZFNet adapted for 32x32 input (CIFAR-10/100).
    Applies the same ZFNet philosophy — smaller early kernels + stride
    to preserve more spatial detail in shallow feature maps.

    Dimension trace:
      Conv1(k=5,s=1,p=2): same padding        -> (B, 96,  32, 32)
      MaxPool(k=2,s=2):                        -> (B, 96,  16, 16)
      Conv2(k=3,s=1,p=1): same padding        -> (B, 256, 16, 16)
      MaxPool(k=2,s=2):                        -> (B, 256,  8,  8)
      Conv3(k=3,s=1,p=1): same padding        -> (B, 384,  8,  8)
      Conv4(k=3,s=1,p=1): same padding        -> (B, 384,  8,  8)
      Conv5(k=3,s=1,p=1): same padding        -> (B, 256,  8,  8)
      MaxPool(k=2,s=2):                        -> (B, 256,  4,  4)
      Flatten: 256*4*4 = 4096
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96,  kernel_size=5, stride=1, padding=2),   # -> (B, 96,  32, 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                    # -> (B, 96,  16, 16)

            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),  # -> (B, 256, 16, 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                    # -> (B, 256,  8,  8)

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), # -> (B, 384,  8,  8)
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), # -> (B, 384,  8,  8)
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), # -> (B, 256,  8,  8)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                    # -> (B, 256,  4,  4)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Deconvnet visualization utility
# The actual contribution of the paper — projects feature activations
# back to pixel space to show what a filter has learned.
# ---------------------------------------------------------------------------

class DeconvNet(nn.Module):
    """
    Projects the activation of a chosen conv layer back to input pixel space.

    This is NOT a generative model. It answers:
    "Given that this filter fired strongly here, what input pattern caused it?"

    How it works:
      1. Run a forward pass, keep the feature map at the target layer
      2. Zero out all activations except the one you want to visualize
      3. Pass it backward through: unpooling -> ReLU -> transposed conv
         mirroring the forward path in reverse

    Usage:
        model    = ZFNet()
        deconv   = DeconvNet(model)
        img      = torch.randn(1, 3, 224, 224)
        proj     = deconv.project(img, layer_idx=0, filter_idx=3)
        # proj is (1, 3, 224, 224) — the pixel-space projection
    """

    def __init__(self, model: ZFNet) -> None:
        super().__init__()

        # Mirror the conv layers as transposed convs using the same weights
        self.deconv_layers = nn.ModuleList()
        self.switches: list[torch.Tensor] = []   # stores MaxPool switch positions

        for layer in model.features:
            if isinstance(layer, nn.Conv2d):
                deconv = nn.ConvTranspose2d(
                    in_channels=layer.out_channels,
                    out_channels=layer.in_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                )
                # Tie weights — transposed conv uses same weights as forward conv
                deconv.weight = layer.weight
                self.deconv_layers.append(deconv)

    def project(
        self,
        x: torch.Tensor,
        layer_idx: int,
        filter_idx: int,
    ) -> torch.Tensor:
        """
        Args:
            x:          input image tensor (1, 3, H, W)
            layer_idx:  which conv layer to visualize (0-4 for ZFNet's 5 convs)
            filter_idx: which filter within that layer

        Returns:
            projection: (1, 3, H, W) — pixel-space reconstruction
        """
        raise NotImplementedError(
            "Full deconvnet visualization requires storing switch positions "
            "during forward pass (MaxPool indices). Implement by registering "
            "forward hooks on each MaxPool2d with return_indices=True, then "
            "unpool with nn.MaxUnpool2d in reverse order. "
            "See: https://github.com/hukkelas/DeepVisualization for reference."
        )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Full ZFNet — 224x224
    model = ZFNet(num_classes=1000).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    out = model(x)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ZFNet        input: {tuple(x.shape)} -> output: {tuple(out.shape)}")
    print(f"Trainable parameters: {params:,}")

    # Small variant — 32x32
    model_s = ZFNetSmall(num_classes=10).to(device)
    xs = torch.randn(2, 3, 32, 32).to(device)
    outs = model_s(xs)
    params_s = sum(p.numel() for p in model_s.parameters() if p.requires_grad)
    print(f"\nZFNetSmall   input: {tuple(xs.shape)} -> output: {tuple(outs.shape)}")
    print(f"Trainable parameters: {params_s:,}")

    # Diff vs AlexNet — the only architectural change
    print("\n--- ZFNet vs AlexNet changes ---")
    print("Conv1: AlexNet(k=11, s=4, p=2) -> ZFNet(k=7, s=2, p=1)")
    print("Conv2: AlexNet(k=5,  s=1, p=2) -> ZFNet(k=5, s=2, p=0)")
    print("Conv3-5, Classifier: identical")