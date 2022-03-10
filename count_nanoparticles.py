import argparse
import pickle as pkl

import imageio
import cv2
import numpy as np
from tqdm import tqdm

from scipy import ndimage as ndi
import scipy.ndimage.filters as filters
from skimage.segmentation import watershed


def mark_local_maxima(image, neighborhood_size=25, threshold=1500):
    data_max = filters.maximum_filter(image, neighborhood_size)
    maxima = (image == data_max)
    data_min = filters.minimum_filter(image, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndi.label(maxima)
    slices = ndi.find_objects(labeled)
    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2
        y.append(y_center)

    return x, y


def dist2(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def fuse(x, y, d):
    points = np.array([x, y]).T
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i + 1, n):
                if dist2(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count += 1
                    taken[j] = True

            point[0] /= count
            point[1] /= count
            ret.append((point[0], point[1]))

    ret = np.array(ret)
    return ret[:, 0], ret[:, 1]


def distance_transform(pic):
    dt = cv2.distanceTransform(pic, cv2.DIST_L2, 5)
    return dt


def find_max(dt):
    x, y = mark_local_maxima(dt, 10, 5)
    x, y = fuse(x, y, 10)
    return x, y


def process_watershed(dt, mask):
    x, y = find_max(dt)

    local_maxi = np.zeros(dt.shape)
    for i, x_ in enumerate(tqdm(x)):

        if (x_ < 1) or (y[i] < 1) or ((-x_ + dt.shape[1]) < 1) or ((-y[i] + dt.shape[0]) < 1):
            continue

        local_maxi[int(y[i]), int(x_)] = True

    markers = ndi.label(local_maxi)[0]
    labels = watershed(-dt, markers, mask=mask)

    return labels


def process_frame(frame):
    mask_frame = frame.astype(np.uint8)
    dt = distance_transform(mask_frame)

    labels = process_watershed(dt, mask_frame)
    uniq = np.unique(labels)

    output = np.zeros((len(uniq), 5))

    for i, label in enumerate(tqdm(np.unique(labels)[1:])):
        temp_image = np.zeros(labels.shape)
        temp_image[labels == label] = 1

        ccws = cv2.connectedComponentsWithStats(temp_image.astype(np.uint8), 4, cv2.CV_32S)
        output[i] = ccws[2][1]

    return output, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary_image', type=str, help="Binary image, containing nanoparticles")
    parser.add_argument('--output_path', type=str, help="Path to the output pickle-file with nanoparticle data")

    args = parser.parse_args()

    mask = imageio.imread(args.binary_image)
    coordinates, labels = process_frame(mask)

    with open(args.output_path, 'wb') as f:
        pkl.dump({
            'coordinates': coordinates,
            'labels': labels
        }, f)
