import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet9(nn.Module):

    def __init__(self, in_channels=3, num_classes=8):
        super().__init__()

        # Convolution Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Convolution Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Residual Block 1
        self.res1 = nn.Sequential(
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128)
        )

        # Convolution Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Convolution Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                256,
                512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Residual Block 2
        self.res2 = nn.Sequential(
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512)
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):

        out = self.conv1(xb)

        out = self.conv2(out)

        # Residual connection
        out = self.res1(out) + out
        out = F.relu(out)

        out = self.conv3(out)

        out = self.conv4(out)

        # Residual connection
        out = self.res2(out) + out
        out = F.relu(out)

        return self.classifier(out)