import config
import pathlib
import cloudinary

# Config
cloudinary.config(
    cloud_name="",
    api_key="",
    api_secret="",
    secure=True
)

from cloudinary.uploader import upload
import cloudinary.uploader
import cloudinary.api


def prepare_images2(original_root, name):
    folder = pathlib.Path(original_root)
    i = 0

    for item in folder.iterdir():
        cloudinary.uploader \
            .upload(str(item),
                    transformation={
                        "gravity": "auto",
                        "aspect_ratio": "1",
                        "width": 128,
                        "height": 128,
                        "crop": "fill"},
                    public_id=str(name + str(i) + '.jpg'),
                    folder=name,
                    overwrite=True)
        i += 1


if __name__ == "__main__":
    prepare_images2(config.ORIGINAL_PUMA_ROOT, 'puma')
    prepare_images2(config.ORIGINAL_LION_ROOT, 'lion')
    print(cloudinary.utils.download_folder('puma'))
    print(cloudinary.utils.download_folder('lion'))
