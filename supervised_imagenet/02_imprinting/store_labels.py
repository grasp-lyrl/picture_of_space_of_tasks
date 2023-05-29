import numpy as np
import os

from utils.data import create_loader


def store_imagenet_val_labels():
    """
    Store the true labels for the ImageNet dataset
    """
    path = "../data/img_all_val.ffcv"
    loader = create_loader(path, deterministic=True)
    labels = []
    for x, y in loader:
        labels.append(y.to('cpu').numpy())
    labels = np.concatenate(labels)

    os.makedirs("../../predictions/labels", exist_ok=True)
    np.save("../../predictions/labels/imagenet_val_labels.npy", labels)


if __name__ == '__main__':
    store_imagenet_val_labels()
