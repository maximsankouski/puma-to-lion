import torch
import torch.nn as nn
import config


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ConvTransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.convtransp = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convtransp(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.resblock = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.resblock(x)


class Generator(nn.Module):
    def __init__(self, img_channels=3, num_features=64, num_residuals=config.NUM_RESIDUALS):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(
                img_channels, num_features, kernel_size=7, stride=1, padding=3
            ),
            ConvBlock(
                num_features, num_features * 2, kernel_size=3, stride=2, padding=1
            ),
            ConvBlock(
                num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1
            ),

            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)],

            ConvTransBlock(
                num_features * 4, num_features * 2, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            ConvTransBlock(
                num_features * 2, num_features, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Conv2d(
                num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect",
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


def test():
    img_channels = 3
    x = torch.randn((16, img_channels, config.IMAGE_SIZE, config.IMAGE_SIZE))
    gen = Generator()
    print(gen(x).shape)
    print(gen)


if __name__ == "__main__":
    test()
