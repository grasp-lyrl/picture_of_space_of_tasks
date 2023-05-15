# Picture of the Space of Learnable Tasks

[This paper](https://arxiv.org/abs/2210.17011) (ICML 23) develops techniques to
analyze the training trajectories of representations learnt by deep networks.
Our techniques allow us to compare representations learnt using different
datasets and using different methods. We study phenomena relating to
fine-tuning, supervised, meta- and contrastive learning by studying networks in
prediction space. 

**Understanding the techniques from Information Geometry**: A good place to start would be [this jupyter notebook](https://colab.research.google.com/github/grasp-lyrl/picture_of_space_of_tasks/blob/main/picture_of_tasks_tutorial.ipynb). It contains a brief overview of the techniques and only takes a few minutes to run on Google Colab.

Our techniques help us build a qualitative and quantitative picture of the space of tasks. For example in the below figure, we plot the visual representations learnt by networks trained on different subsets of ImageNet. Interestingly, the trajectories of the representations resemble the Wordnet phylogenetic tree which was built using only natural language-based semantics. 

<p align="center">
<img src="./plots/imagenet/tree.png" width="600">
</p>



# Directory Structure


```bash

├── picure_of_tasks_tutorial.ipynb  # Tutorial summarizing the techniques
└── superivsed_imagenet:           
    ├── 01_imagenet_training        # Trains models in ImageNet
    ├── 02_imagenet_predictions     # Converts checkpoints into predictions
    └── 03_imagenet_stats           # Analyze prediction space trajectories
```


