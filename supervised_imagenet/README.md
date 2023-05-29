## Prediction space of ImageNet

The ImageNet models were trained using a modified version of the [FFCV ImageNet training scripts](https://github.com/libffcv/ffcv-imagenet). A single training run (1 dataset and 1 seed) takes about 4 hours using 4 Nvidia A10G GPUs.

**Step 1**: Download the [ImageNet-1k](http://www.image-net.org/) dataset and place it into `./data`. The directory structure should look like:
```bash
.
└── supervised_imagenet
    └── data
        ├── img1k.tar
        ├── train
        └── val
```


**Step 2**: Train models on tasks created from ImageNet. The code to train most tasks are commented out in `write_imagenet.sh` and `train_imagenet.sh`. Note that this requires a lot of storage (around 1 TB) since it creates a different FFCV object for each task. You can make do with less disk storage by deleting the FFCV objects after training.


```bash
cd 01_training
bash write_imagenet.sh    # Create FFCV objects for fast data-loading
bash train_imagenet.sh    # Train on ImageNet and store checkpoints
```


**Step 3**: Imprint models and store the predictions. Finally, we convert the trajectories in weight space to trajectories in prediction space.

```bash
cd 02_imprinting
bash store_predictions.sh
```
