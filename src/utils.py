import numpy as np
from PIL import Image
from tensorflow.keras.datasets import cifar10

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
    """Greyscales images in the array."""
    res = []
    for image in images:
        image = Image.fromarray(image)
        res.append(np.array(image.convert("L")))
    return np.stack(res)


def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flattens images in the given image array."""
    n, w, h = images.shape
    return np.reshape(images, (n, w * h))


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads CIFAR-10 data, greyscales and flattens images."""
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = flatten_images(greyscale_images(X_train))
    X_test = flatten_images(greyscale_images(X_test))
    y_train = np.array([class_names[y] for y in np.ravel(y_train)])
    y_test = np.array([class_names[y] for y in np.ravel(y_test)])
    return X_train, y_train, X_test, y_test
