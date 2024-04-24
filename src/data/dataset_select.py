from torchvision import transforms
from torch.utils.data import random_split

from .datasets import CelebADataset


def get_dataset(config, transform=None):
    if config.data_name == 'celeba':
        return get_celeba_dataset(config, transform)


def get_celeba_dataset(config, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((config.data_size, config.data_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    dataset_base = CelebADataset(root_dir=config.data_path, transform=transform, filename=config.data_filename)
    if config.data_split_ratio > 0.0:
        train_len = int(config.data_split_ratio * len(dataset_base))  # 取整
        test_len = len(dataset_base) - train_len
        train_dataset, test_dataset = random_split(dataset_base, [train_len, test_len])
        return train_dataset, test_dataset
    else:
        return dataset_base, None
