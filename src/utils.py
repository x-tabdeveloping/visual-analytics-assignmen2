import numpy as np
from PIL import Image

class_names = [
    "aquatic",
    "fish",
    "flowers",
    "food_containers_bottle",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man_made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_sized_mammals",
    "non_insect_invertebrates",
    "people",
    "reptiles",
    "small",
    "trees",
    "vehicles_1",
    "vehicles_2",
]


def greyscale_images(images: np.ndarray) -> np.ndarray:
    res = []
    for image in images:
        image = Image.fromarray(image)
        res.append(np.array(image.convert("L")))
    return np.stack(res)


def flatten_images(images: np.ndarray) -> np.ndarray:
    n, w, h = images.shape
    return np.reshape(images, (n, w * h))
