from typing import Union
import torch.nn as nn


from torchvision.transforms import functional as F
from PIL import Image
import torch


class CropBottom(torch.nn.Module):
    def __init__(self, output_size: Union[tuple, int]):
        """
        Args:
            output_size (tuple or int): Desired output size of the crop. If int, square crop is made.
        """
        super(CropBottom, self).__init__()
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def forward(self, img: Union[Image, torch.Tensor]):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        crop_height, crop_width = self.output_size
        img_width, img_height = F.get_image_size(img)
        top = max(0, img_height - crop_height)

        return F.crop(img, top, 0, crop_height, crop_width)

    def __repr__(self):
        return self.__class__.__name__ + '(output_size={0})'.format(self.output_size)


if __name__ == "__main__":
    import numpy as np

    image = Image.new('RGB', (100, 200), color='red')
    bottom_half = Image.new('RGB', (100, 100), color='blue')
    image.paste(bottom_half, (0, 100))
    output_size = (50, 100)
    transform = CropBottom(output_size)
    cropped_image = transform(image)
    assert cropped_image.size == output_size[::-1]

    cropped_image_np = np.array(cropped_image)
    unique_colors = np.unique(
        cropped_image_np.reshape(-1, cropped_image_np.shape[2]), axis=0)
    assert np.any(np.all(unique_colors == [0, 0, 255], axis=1))
    assert len(unique_colors) == 1
