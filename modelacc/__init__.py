# get current directory
import os
import sys
import random

# get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# get the parent directory
parent_dir = os.path.dirname(current_dir)

"""
HELPER FUNCTIONS FOR DEBUGGING
"""


def get_root_dir() -> str:
    return parent_dir


def get_test_img_dir() -> str:
    return os.path.join(get_root_dir(), "data/test/images")


def get_sample_test_image_path() -> str:
    test_images = os.listdir(get_test_img_dir())
    random_image = random.choice(test_images)
    return os.path.join(get_test_img_dir(), random_image)


# import everything from preprocess.py
from .preprocess import *
from .model_convert import *
from .postprocess import *
