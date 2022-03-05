import os
import yaml
import pickle as pkl
from argparse import ArgumentParser

import imageio
import numpy as np
import torch
import segmentation_models_pytorch as smp

from tdnc.dataset_preparation import SegmentedImages, get_loader
from tdnc.utils import MODEL_MAPPER, module_from_file

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the yaml config file")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        params = yaml.load(f)

    images, masks = [], []
    for i in range(int(len(os.listdir(params['dset_path']))/2)):
        images.append(np.array(imageio.imread(os.path.join(params['dset_path'], f'{i}_image.png'))))
        masks.append(np.array(imageio.imread(os.path.join(params['dset_path'], f'{i}_mask.png'))/255))

    images = np.array(images)
    masks = np.array(masks)

    aug = module_from_file("aug", params['augmentation_path'])

    os.mkdir(params['output_path'])

    train = get_loader(SegmentedImages, images, masks, params['train_index'],
                       aug.get_training_augmentation, params['bs']['train'])

    valid = get_loader(SegmentedImages, images, masks, params['test_index'],
                       aug.get_validation_augmentation, params['bs']['test'])

    model = MODEL_MAPPER[params['architecture']](
        params['encoder'], in_channels=1, classes=1, activation=params['activation']
    )

    loss = smp.utils.losses.BCELoss() + smp.utils.losses.DiceLoss()

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=params['lr']),
    ])

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=params['device'],
        verbose=False,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=params['device'],
        verbose=False,
    )

    max_score = 0

    training = []
    validation = []

    for i in range(params['n_epochs']):
        train_logs = train_epoch.run(train)
        valid_logs = valid_epoch.run(valid)

        training.append(train_logs)
        validation.append(valid_logs)

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, os.path.join(params['output_path'], 'best.pth'))

        if i % params['print_every'] == 0:
            print(i)

        if i in params['lr_reduce_epochs']:
            optimizer.param_groups[0]['lr'] /= params['lr_reduce_by']
            print('Decrease learning rate to ' + str(optimizer.param_groups[0]['lr']))

    torch.save(model, os.path.join(params['output_path'], 'last.pth'))

    print("Max score is", max_score)

    with open(os.path.join(params['output_path'], 'logs.pkl'), 'wb') as f:
        pkl.dump(
            {
                'train': training,
                'valid': validation,
                'max_score': max_score
            },
            f
        )
