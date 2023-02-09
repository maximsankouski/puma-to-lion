import config
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import PumaLionDataset


def get_size_of_dataset(first_root, second_root):
    images1 = os.listdir(first_root)
    images2 = os.listdir(second_root)
    return min([len(images1), len(images2)])


def get_datasets(first_root, second_root):
    size_of_dataset = get_size_of_dataset(first_root, second_root)
    dataset = PumaLionDataset(first_root, second_root, transform=config.image_transform)
    dataset_train, dataset_test = train_test_split(dataset, train_size=(size_of_dataset-config.TEST_LEN), shuffle=True)
    return dataset_train, dataset_test


if __name__ == "__main__":
    pumalion_train, pumalion_test = get_datasets(config.PUMA_ROOT, config.LION_ROOT)
    dataloader = DataLoader(pumalion_train, batch_size=config.BATCH_SIZE, shuffle=True)
    dataloader_test = DataLoader(pumalion_test, batch_size=config.BATCH_SIZE, shuffle=False)
