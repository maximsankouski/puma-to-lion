import config
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def denorm(img_tensors):
    return img_tensors * config.STATS[1][0] + config.STATS[0][0]


def show_images(images, nmax=16):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=4).permute(1, 2, 0))


def get_images(dl):
    images_puma = []
    images_lion = []
    for _ in range(int(16 / config.BATCH_SIZE)):
        images_puma.append(next(iter(dl))[0])
        images_lion.append(next(iter(dl))[1])
    images_puma = torch.cat(images_puma)
    images_lion = torch.cat(images_lion)
    return images_puma, images_lion


def get_images_test(dl):
    images_puma = []
    images_lion = []
    for puma, lion in dl:
        images_puma.append(puma)
        images_lion.append(lion)
    images_puma = torch.cat(images_puma)
    images_lion = torch.cat(images_lion)
    return images_puma, images_lion
