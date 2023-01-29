import os
import config
import torchvision.transforms as tt
from PIL import Image
import pathlib


def prepare_images(original_root, root, name: str):
    folder = pathlib.Path(original_root)
    i = 0

    os.makedirs(root, exist_ok=True)

    for item in folder.iterdir():
        img = Image.open(item)

        size = img.size
        size0 = size[0]
        size1 = size[1]

        if size0 >= size1:
            size0 = int(size0 / size1 * config.IMAGE_SIZE)
            size1 = config.IMAGE_SIZE
        else:
            size1 = int(size1 / size0 * config.IMAGE_SIZE)
            size0 = config.IMAGE_SIZE

        transform = tt.Compose(
            [
                # Different order of the sides of the rectangle in torchvision.transforms
                tt.Resize(size=(size1, size0)),
                tt.CenterCrop(size=config.IMAGE_SIZE),
            ]
        )

        img = transform(img)
        img.save(root + '/' + name + str(i) + '.jpg')
        i += 1


if __name__ == "__main__":
    prepare_images(config.ORIGINAL_PUMA_ROOT, config.PUMA_ROOT, 'puma')
    prepare_images(config.ORIGINAL_LION_ROOT, config.LION_ROOT, 'lion')
