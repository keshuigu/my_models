import os

import PIL.Image
from loguru import logger as log
from torch.utils.data import Dataset
from torchvision import transforms


def get_default_transforms():
    return transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])  # 标准化处理


class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None, filename='celeba.txt'):
        self.root_dir = root_dir
        self.transform = transform if transform is not None else get_default_transforms()
        self.list_imgs_path = os.path.join(self.root_dir, filename)
        self.list_imgs = []
        if not os.path.exists(self.list_imgs_path):
            log.error(f'{self.list_imgs_path} does not exist')
            raise FileExistsError(self.list_imgs_path)
        with open(self.list_imgs_path, 'r') as f:  # 遍历文件每一行
            for index, context in enumerate(f.readlines()):
                context = context.strip()
                one_line = context.split(' ')
                class_id = int(one_line[1])
                img_path = one_line[0]
                self.list_imgs.append({'cid': class_id, 'img_path': img_path})
                if index % 10000 == 0:
                    log.info(f'Processed {index}')

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, idx):
        info = self.list_imgs[idx]
        img_path = os.path.join(self.root_dir, 'data/', info['img_path'])
        img = PIL.Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, info['cid']


if __name__ == '__main__':
    dataset = CelebADataset(root_dir='/mnt/share/workspace/generate_models/dataset/celeba_prepared',
                            filename='mini1000_1.txt')
    log.info(f'len(dataset) = {len(dataset)}')
    img, class_id = dataset[0]
    # 给图像处理后到文件test.png
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # 反归一化
        transforms.ToPILImage(),
    ])
    # 将张量应用预处理转换
    image = preprocess(img)
    image.save("test.png")
