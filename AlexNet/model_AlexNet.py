"""
'ImageNet Classification with Deep Convolutional Neural Networks'

- Activation function : ReLU
- cross-GPU parallelization
- Local Response Normalization
- Overlapping Pooling

"""

import torch.nn as nn
import torch.nn.functional as F


# Paper ver
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(3, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(3, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fclayer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(-1, 256 * 6 * 6)

        x = self.fclayer(x)

        return x

        
# CIFAR10 ver
class CustomAlexNet(nn.Module):
    def __init__(self):
        super(CustomAlexNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 48, 4, 2),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(3, 2),
        )

        # (input - F_size + 2*pad)/Stride + 1

        """
        (32 - 4)/2 + 1 = 15 * 15 * 48
        (15 - 3)/2 + 1 = 7 * 7 * 48
        """
        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 100, 2, 1, 2),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(2, 1)
        )
        """
        (7 - 2 + 2*2)/1 + 1 = 10 * 10 * 100
        (10 - 2)/1  + 1= 9 * 9 * 100
        """
        self.conv3 = nn.Sequential(
            nn.Conv2d(100, 150, 3, 1, 1),
            nn.ReLU()
        )
        """
        (9 - 3 + 2*1)/1 + 1 = 9 * 9 * 150
        """
        self.conv4 = nn.Sequential(
            nn.Conv2d(150, 150, 3, 1, 1),
            nn.ReLU()
        )
        """
        (9 - 3 + 2*1)/1 + 1 = 9 * 9 * 150
        """
        self.conv5 = nn.Sequential(
            nn.Conv2d(150, 48, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 1)
        )
        """
        (9 - 3 + 2*1)/1 + 1 = 9 * 9 * 48
        (9 - 2)/1 + 1 = 8 * 8 * 48 = 3072
        """

        self.fclayer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(8 * 8 * 48, 864),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(864, 864),
            nn.ReLU(),
            nn.Linear(864, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(-1, 8 * 8 * 48)

        x = self.fclayer(x)

        return x


