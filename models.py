"""
In this script you'll find all models for this mini-Deep Learning project.

This file implements two ResNet-18 based models:

1) Pretrained ResNet-18 (transfer learning)
2) Custom ResNet-18 (built manually using BasicBlock and residual connections)

"""


import torch
from torch import nn
from torchvision import models
import config


#________________________________________________________________________________________________________________________________
#
# Helper: Count parameters (used in documentation)
#________________________________________________________________________________________________________________________________

def count_trainable_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#________________________________________________________________________________________________________________________________
#
# MODEL 1 — Pretrained ResNet-18 (Transfer Learning)
#________________________________________________________________________________________________________________________________

def build_pretrained_model() -> nn.Module:
    """
    Pretrained ResNet-18 from ImageNet, adapted to our NUM_CLASSES.
    Backbone is kept trainable since the model uses fine-tuning
    """

    # Load pretrained weights  (Transfer Learning concept)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Backbone uses pretrained weights; gradients are enabled so it can be fine-tuned
    freeze_backbone = False
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final FC layer  (Adjust classifier head)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, config.NUM_CLASSES)

    # Ensure head is trainable: new classification layer must learn task specific features,
    # backbone may be frozen to keep pretrained representations stable,
    # optimizer will only update layers where requires_grad is True
    for param in model.fc.parameters():
        param.requires_grad = True

    # Move model to device  (GPU acceleration)
    model = model.to(config.DEVICE)

    # Parameter count (Model complexity)
    print(f"[Pretrained ResNet-18] Trainable parameters: {count_trainable_parameters(model)}")

    return model


#________________________________________________________________________________________________________________________________
#
# MODEL 2 — Custom ResNet-18 (Manual Implementation)
#________________________________________________________________________________________________________________________________

class BasicBlock(nn.Module):
    """
    Basic residual block as in ResNet-18.

    KEY IDEAS:
    - Two 3×3 convolutions with BatchNorm + ReLU
    - Skip connection (identity or 1×1 conv)
    - Residual learning (solves vanishing gradients)
    """

    expansion = 1  # ResNet-18 use expansion=1 (no channel expansion)

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: nn.Module | None = None) -> None:
        super().__init__()

        # First 3×3 conv   Local receptive field)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,           # keep spatial size
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)  # Non-linearity

        # Second 3×3 conv   Feature refinement)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample path   Match dimensions for skip)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # identity skip path

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  # activation after BN

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection  preserves gradient flow)
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity  # residual addition
        out = self.relu(out)

        return out


class CustomResNet18(nn.Module):
    """
    Manual implementation of ResNet-18  Architecture from CV pensum)

    KEY STRUCTURE:
    - Conv7×7 stem + MaxPool  aggressive spatial reduction)
    - 4 stages with [2,2,2,2] BasicBlocks
    - Channels: 64 128 256 512
    - Stride=2 on first block of each stage (except stage1)
    - Global Average Pooling  reduces spatial dims to 1×1)
    - Fully connected classifier
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        self.in_channels = 64

        # ----- STEM LAYER -----
        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=7,       # large kernel (historical ResNet choice)
            stride=2,            # downsampling
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # MaxPool (further spatial downsampling)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ----- 4 RESIDUAL STAGES -----
        self.layer1 = self._make_layer(64, 2, stride=1)  # no downsampling
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # ----- CLASSIFIER -----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # Weight initialization (He initialization)
        self._init_weights()

    def _make_layer(self,
                    out_channels: int,
                    num_blocks: int,
                    stride: int) -> nn.Sequential:
        """
        Build one ResNet stage.
        """
        downsample = None

        # If size or channel depth changes apply 1×1 conv in skip
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        layers = []

        # First block (may downsample)
        layers.append(
            BasicBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample
            )
        )

        # Update channel tracker
        self.in_channels = out_channels

        # Remaining blocks (stride=1)
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1,
                    downsample=None
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """He initialization  good for ReLU networks)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----- STEM -----
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ----- RESIDUAL STAGES -----
        x = self.layer1(x)   # 64 channels
        x = self.layer2(x)   # 128 channels
        x = self.layer3(x)   # 256 channels
        x = self.layer4(x)   # 512 channels

        # ----- CLASSIFICATION HEAD -----
        x = self.avgpool(x)          # GAP
        x = torch.flatten(x, 1)      # (batch, features)
        x = self.fc(x)               # logits

        return x


#________________________________________________________________________________________________________________________________
#
# MODEL FACTORY — Based on config.MODEL_TYPE
#________________________________________________________________________________________________________________________________

def build_model_from_config() -> nn.Module:
    """Return either pretrained model or custom ResNet-18."""
    if config.MODEL_TYPE == "resnet_pretrained":
        return build_pretrained_model()
    elif config.MODEL_TYPE == "resnet_custom":
        model = CustomResNet18(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        print(f"[Custom ResNet-18] Trainable parameters: {count_trainable_parameters(model)}")
        return model
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {config.MODEL_TYPE}")
