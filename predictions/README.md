#  Trajectory Data

**ImageNet predictions**: https://mega.nz/folder/lAU2EBTT#6NXdRnL2RUoZL06e2baP7A

Download and place the files into this directory. The directory should look like:

```
.
└── predictions
    ├── imagenet_all                  # Trajectories of models trained on all ImageNet classes
    ├── imagenet_3rd                  # Trajectories of models trained on a subset of ImageNet
    ├── imagenet_vert                 # Trajectories of models trained on all Vertebrates
    ├── imagenet_instr                # Trajectories of models trained on all Instrumentality
    ├── embedding
    │   ├── fig_1a.pkl                 # InPCA Embedding for ImageNet trajectories
    │   ├── fig_wordnet.pkl           # InPCA Embedding for ImageNet-Wordnet trajectories
    │   └── imagenet_inpca.pkl        # InPCA Embedding from code in this repo
    └── labels
        └── imagenet_val_labels.npy   # ImageNet validation labels
```

Please do reach out if you would like access to more trajectories.
