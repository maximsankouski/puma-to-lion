import torch
import torch.nn as nn
import config


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        features = [64, 128, 256, 512, 1]
        strides = [2, 2, 2, 1, 1]
        instancenorm2ds = [0, 1, 1, 1, 0]
        leakyrelus = [1, 1, 1, 1, 0]

        layers = []
        for indx in range(5):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels=features[indx],
                    kernel_size=4,
                    stride=strides[indx],
                    padding=1,
                    padding_mode="reflect",)
            )
            if instancenorm2ds[indx]:
                layers.append(nn.InstanceNorm2d(features[indx]))
            if leakyrelus[indx]:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = features[indx]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = nn.functional.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x


def test():
    x = torch.randn((16, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
    disc = Discriminator()
    print(disc(x).shape)
    print(disc)
    print(disc(x))


if __name__ == "__main__":
    test()
