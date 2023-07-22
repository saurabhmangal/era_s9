from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomResnetTransforms:

    def train_transforms(means,stds):
        transforms = A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                A.HorizontalFlip(),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, p=0.1),
                A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means),
                ToTensorV2(),
            ]
        )
        return transforms

    def test_transforms(means,stds):
        transforms = A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                ToTensorV2(),
            ]
        )
        return transforms


