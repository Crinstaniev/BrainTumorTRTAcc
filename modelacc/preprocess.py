# preprocess image
from PIL import Image
import numpy as np
import os


def preprocess_image(
    image_path: str, target_size: int, transpose: bool = False
) -> np.ndarray:
    """
    Preprocess image for model prediction.
    Target size is (3, target_size, target_size)

    :param image_path: str: path to the image
    :param target_size: tuple: target size for the image
    :return: np.ndarray: preprocessed image
    """
    # open image
    image = Image.open(image_path)

    # resize image
    image = image.resize((target_size, target_size))

    # convert image to array
    image = np.array(image)

    # normalize image
    image = image / 255.0

    if transpose:
        image = np.transpose(image, (2, 0, 1))

    # add batch dimension
    image = np.expand_dims(image, axis=0)

    return image
