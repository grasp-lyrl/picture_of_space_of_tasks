import torch as ch
import numpy as np
from pathlib import Path


from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder


def create_loader(dataset_path, deterministic=False):
    """
    Create a data-loader
    """
    num_workers = 4
    batch_size = 256

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    DEFAULT_CROP_RATIO = 224/256

    this_device = "cuda:0"
    dpath = Path(dataset_path)
    assert dpath.is_file()

    res_tuple = (256, 256)
    cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(ch.device(this_device) ,non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(ch.device(this_device),
        non_blocking=True)
    ]

    if deterministic:
        ordr = OrderOption.SEQUENTIAL
    else:
        ordr = OrderOption.QUASI_RANDOM

    loader = Loader(dataset_path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=ordr,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=False)
    return loader
