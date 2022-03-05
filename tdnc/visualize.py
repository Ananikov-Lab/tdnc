import matplotlib.pyplot as plt
import math


def plot_image(image, mask, rows=1, current_index=0):
    plt.subplot(rows, 2, current_index * 2 + 1)
    plt.imshow(image, cmap='gray')

    plt.subplot(rows, 2, current_index * 2 + 2)
    plt.imshow(mask, cmap='gray')


def create_figure(figsize, rows, columns=2):
    if figsize is None:
        plt.figure(figsize=(5 * columns, rows * 5))
    else:
        plt.figure(figsize=figsize)


def plot_images(images, masks, path=None, figsize=None, show=True):
    n_of_imgs = len(images)

    create_figure(figsize, n_of_imgs)

    for img_index in range(n_of_imgs):
        plot_image(images[img_index], masks[img_index], n_of_imgs, img_index)

    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    if not show:
        plt.close()


def plot_with_aug(images, masks, dset_class, augs, image_idx=1, n_examples=5, path=None, figsize=None, show=True):
    augmented_dataset = dset_class(
        images, masks,
        augmentation=augs
    )

    create_figure(figsize, n_examples)

    for example_idx in range(n_examples):
        image, mask = augmented_dataset[image_idx]
        plot_image(image.transpose(1, 2, 0)[:, :, 0], mask.transpose(1, 2, 0)[:, :, 0], n_examples, example_idx)

    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    if not show:
        plt.close()


def plot_logs(logs, columns=3, figsize=None, path=None, show=True):
    rows = math.ceil(len(logs) // columns)
    create_figure(figsize, rows, columns)

    metric = 'iou_score'
    for i, row in logs.iterrows():
        plt.subplot(rows, columns, i + 1)
        plt.title(f"{row['arch']} / {row['encoder']}")
        plt.plot([log[metric] for log in row['train']])
        plt.plot([log[metric] for log in row['valid']])
        plt.ylim(0, 1)

    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    if not show:
        plt.close()
