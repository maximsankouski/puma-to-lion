from PIL import Image
import os
from torch.utils.data import Dataset


class PumaLionDataset(Dataset):
    def __init__(self, root_puma, root_lion, transform):
        self.root_puma = root_puma
        self.root_lion = root_lion
        self.transform = transform

        self.puma_images = os.listdir(root_puma)
        self.lion_images = os.listdir(root_lion)
        self.length_dataset = max(len(self.puma_images), len(self.lion_images))
        self.puma_len = len(self.puma_images)
        self.lion_len = len(self.lion_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        puma_img = self.puma_images[index % self.puma_len]
        lion_img = self.lion_images[index % self.lion_len]

        puma_path = os.path.join(self.root_puma, puma_img)
        lion_path = os.path.join(self.root_lion, lion_img)

        puma_img = Image.open(puma_path).convert("RGB")
        lion_img = Image.open(lion_path).convert("RGB")

        puma_img = self.transform(puma_img)
        lion_img = self.transform(lion_img)

        return puma_img, lion_img
