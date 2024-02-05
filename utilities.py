from PIL import Image
import PIL
from nsd_access import NSDAccess


def load_img_from_arr(img_arr, resolution):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = resolution, resolution
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    return image


def load_img_from_nsd(image_index, resolution):
    nsda = NSDAccess('../nsd/')
    image = nsda.read_images(image_index)
    return load_img_from_arr(image, resolution)
