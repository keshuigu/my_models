import random

from torchvision import transforms


# 从图像的中心位置开始，加上一个随机的偏移量，然后裁剪出一个指定大小的区域。这种操作在数据增强中常常被用到，可以增加模型的泛化能力。
# data_size - 2*offset_range >= size
class CenterCropWithRandomOffset:
    def __init__(self, size, offset_range=10):
        self.size = size
        self.offset_range = offset_range

    def __call__(self, image):
        # Get the original image size
        c, h, w = image.shape

        # Calculate the center coordinates
        center_h = h // 2
        center_w = w // 2

        # Generate random offsets
        offset_h = random.randint(-self.offset_range, self.offset_range)
        offset_w = random.randint(-self.offset_range, self.offset_range)

        # Calculate the crop coordinates
        start_h = center_h + offset_h - self.size // 2
        start_w = center_w + offset_w - self.size // 2
        end_h = start_h + self.size
        end_w = start_w + self.size

        # Crop the image
        cropped_image = image[:, start_h:end_h, start_w:end_w]

        return cropped_image


def get_transforms(config):
    transform = None
    if config.data_crop == True and config.data_name == 'celeba':
        transform = transforms.Compose([
            transforms.ToTensor(),
            CenterCropWithRandomOffset(config.data_crop_size, config.data_crop_offset),
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=config.data_h_flip_p),
            transforms.Resize((config.data_size, config.data_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    return transform
