import argparse

import imageio
import cv2
import numpy as np
import ttach as tta

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model file")
    parser.add_argument("--target_image_path", type=str, help="Path to the target image")
    parser.add_argument("--output_path", type=str, help="Path to the output file")
    args = parser.parse_args()

    model = torch.load(args.model_path)

    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 180]),
            tta.Scale(scales=[1, 2]),
            tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )

    tta_model = tta.SegmentationTTAWrapper(model, transforms)
    resize = (864, 1280)

    image = imageio.imread(args.target_image_path)[:890] / 255

    with torch.no_grad():
        resized = torch.cuda.FloatTensor(cv2.resize(image, (resize[1], resize[0])
                                                    ).reshape((1, 1, resize[0], resize[1])))
        mask = tta_model(resized).detach().cpu().numpy()[0, 0] > 0.5

    imageio.imwrite(args.output_path, cv2.resize((mask * 255).astype(np.uint8), (image.shape[1], image.shape[0])))
