## Prediction space of ImageNet

The ImageNet models were trained using a modified version of the [FFCV ImageNet training scripts](https://github.com/libffcv/ffcv-imagenet). A single training run (1 dataset and 1 seed) takes about 4 hours using 4 Nvidia A10G GPUs.

**Step 1**: [Install FFCV](https://github.com/libffcv/ffcv#install-with-anaconda) and pytorch. I usually prefer [micromamba](https://mamba.readthedocs.io/en/latest/installation.html) as a package manager over anaconda and highly recommend.

```bash
micromamba create -y -f env.yml
micromamba activate picture
```

**Step 2**: Download the [ImageNet-1k](http://www.image-net.org/) dataset and place it into `./data`. The directory structure should look like this.
```bash
.
└── supervised_imagenet
    └── data
        ├── img1k.tar
        ├── train
        └── val
```


**Step 3**: Train models on tasks created from ImageNet. The code to train most tasks are commented out `write_imagenet.sh` and `train_imagenet.sh`. Note that this requires a lot of storage (around 1 TB) since it creates a different FFCV object for each task.

```bash
cd 01_imagenet_training
bash write_imagenet.sh    # Create FFCV objects for fast data-loading
bash train_imagenet.sh    # Train on ImageNet and store checkpoints
```

You can download predictions from https://mega.nz/folder/lAU2EBTT#6NXdRnL2RUoZL06e2baP7A


