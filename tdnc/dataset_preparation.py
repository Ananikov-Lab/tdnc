import imageio
import numpy as np
import pandas as pd
import albumentations as albu

from psd_tools import PSDImage
from os import path, walk, listdir
from torch.utils.data import Dataset, DataLoader
from datetime import datetime


class SegmentedImages(Dataset):
    def __init__(self, images, masks, augmentation):
        self.images = images
        self.masks = masks

        self.len = len(images)
        self.size = (images[0].shape[0], images[0].shape[1], 1)

        self.augmentation = augmentation

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = self.images[index].reshape(self.size).astype('float32') / 255
        mask = self.masks[index].reshape(self.size).astype('float32')

        augmented = self.augmentation()(image=image, mask=mask)

        return augmented['image'].transpose(2, 0, 1).astype('float32'), augmented['mask'].transpose(2, 0, 1).astype(
            'float32')


class SegmentedImagesMultiLabel(Dataset):
    def __init__(self, images, magnifications, classes, augmentation):
        self.images = images
        self.magnifications = magnifications
        self.classes = classes

        self.len = len(images)

        self.size = (images[0].shape[0], images[0].shape[1], 1)
        self.masks_size = (images[0].shape[0], images[0].shape[1], len(classes))

        self.augmentation = augmentation

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = self.images[index][:, :, 0].reshape(self.size).astype('float32')
        mask = self.images[index][:, :, self.classes].reshape(self.masks_size).astype('float32')

        augmented = self.augmentation()(image=image, mask=mask)

        return augmented['image'].transpose(2, 0, 1).astype('float32'), augmented['mask'].transpose(2, 0, 1).astype(
            'float32')


def get_loader(dset_class, images, masks, indexes, aug, batch_size):
    """Creates corresponding dataloader
    """
    loader = DataLoader(
        dset_class(images[indexes], masks[indexes], aug),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    return loader


def get_loader_multilabel(dset_class, data, classes, indexes, aug, batch_size):
    """Creates corresponding dataloader
    """
    loader = DataLoader(
        dset_class([data[idx][0] for idx in indexes],
                   [data[idx][1] for idx in indexes],
                   classes, aug),
        batch_size=batch_size,
        shuffle=True,
        num_workers=5
    )
    return loader


def process_psd(path, threshold=100, bottom_line_cutoff=890):
    """Opens PSD image, the last layer is the image layer
    """
    psd = PSDImage.open(path)
    image = np.array(psd[0].composite((0, 0, psd.size[0], psd.size[1])))
    mask = np.array(psd[1].composite((0, 0, psd.size[0], psd.size[1])))

    image = image[0:bottom_line_cutoff, :, 0]
    mask = mask[0:bottom_line_cutoff, :, 3] > threshold

    return image, mask


def process_psd_folder(folder, threshold=100, verbose=True):
    """Processes all PSD in a folder
    """
    images = []
    masks = []

    for subdir, dirs, files in walk(folder):
        for file in files:
            if verbose:
                print(path.join(subdir, file))
            image, mask = process_psd(path.join(subdir, file), threshold)
            images.append(image)
            masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    return images, masks


def process_sem_image(path):
    """Get image and metadata
    .txt metadata file must be in path
    """
    data = dict()

    image = imageio.imread(path, "TIFF")
    with open(f"{path[:-4]}.txt", "r") as f:
        lines = f.readlines()
        data['SampleName'] = lines[4][11:-1]
        data['Date'] = lines[8][5:-1]
        data['Time'] = lines[9][5:-1]
        data['DateTime'] = datetime.strptime(lines[8][5:-1] + ' ' + lines[9][5:-1], "%m/%d/%Y %H:%M:%S")
        data['AcceleratingVoltage'] = int(lines[15][20:-6])
        data['Magnification'] = int(lines[17][14:-1])
        data['WorkingDistance'] = int(lines[18][16:-4])
        data['EmissionCurrent'] = int(lines[19][16:-4])
        data['LensMode'] = lines[20][9:-1]
        data['PixelSize'] = float(lines[12].split('=')[1].strip())
        data['DataSize'] = [int(dim) for dim in lines[11].strip().split('=')[1].split('x')]
        data['Path'] = path
        data['X'] = int(lines[39][15:-1])
        data['Y'] = int(lines[40][15:-1])
        data['R'] = float(lines[41][15:-1])
        data['Z'] = int(lines[42][15:-1])

    return image, data


def process_sem_folder_wo_items(folder):
    images = []
    metadata = []

    for filename in listdir(folder):
        if not filename.endswith(".tif"):
            continue

        image, meta_data_item = process_sem_image(path.join(folder, filename))

        images.append(image)
        metadata.append(meta_data_item)

    images = np.array(images)
    metadata = pd.DataFrame(metadata)

    return images, metadata


def process_sem_folder(folder, subfolder=''):
    """Processes a folder of SEM images
    Each image with corresponding metadata file should be in {folder}/{id}/{subfolder}/{image}
    """
    images = []
    metadata = []
    labels = []

    for item in [item for item in listdir(folder) if path.isdir(path.join(folder, item))]:
        for filename in listdir(path.join(folder, item, subfolder)):
            if not filename.endswith(".tif"):
                continue

            image, meta_data_item = process_sem_image(path.join(folder, item, subfolder, filename))

            images.append(image)
            metadata.append(meta_data_item)
            labels.append(int(item))

    images = np.array(images)
    metadata = pd.DataFrame(metadata)
    labels = np.array(labels)

    return images, metadata, labels
