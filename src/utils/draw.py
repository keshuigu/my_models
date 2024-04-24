
import numpy as np
from matplotlib import pyplot as plt


def save_labeled_imgs(imgs, labels, save_path, rows=3, cols=3):
    # imgs: [N, 3, H, W] tensor
    # labels: [N] tensor
    # save_path: save file path
    # rows: number of rows in the subplot grid
    # cols: number of columns in the subplot grid

    N, C, H, W = imgs.shape

    # start figure
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))

    for i, ax in enumerate(axes.flat):
        # calculate subplot indices
        ax.imshow(((imgs[i].permute(1, 2, 0) + 1.) * 127.5).numpy().astype(np.uint8))
        ax.set_title(labels[i].item())
        ax.axis("off")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


from PIL import Image, ImageDraw


def save_labeled_PIL_imgs(imgs, labels, save_path, rows=3, cols=3):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5)

    for i, ax in enumerate(axes.flat):
        if i < len(imgs):
            img = imgs[i]
            label = labels[i]
            ax.imshow(img)
            ax.set_title(f"{label}")
            ax.axis('off')
        else:
            ax.axis('off')

    plt.savefig(save_path)
    plt.close()
