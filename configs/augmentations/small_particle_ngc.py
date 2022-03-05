import cv2
import albumentations as albu


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.VerticalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=(-0.5, 3), rotate_limit=45,
                              shift_limit=0.6, p=1, border_mode=cv2.BORDER_REFLECT),

        albu.GridDistortion(),

        albu.RandomBrightness(p=0.9),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.RandomContrast(p=0.9),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
    ]
    return albu.Compose(test_transform)
