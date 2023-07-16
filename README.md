# Picture of the Space of Learnable Tasks

Many representation learning algorithms like supervised, transfer, meta, semi,
self-supervised learning work unreasonably well---but why is this the case? We
find that *all* of these trajectories seem to be lie in a low dimensional
space; Even 10-dimensional can capture most of the trajectories for ImageNet
(high explained stress). This seems to indicate that the underlying space of
typical tasks are very low-dimensional.

This repository includes code to reproduce the results from [our paper](https://arxiv.org/abs/2210.17011) (ICML 23). This work builds off of techniques developed in our [other paper](https://arxiv.org/abs/2305.01604) (*The Training Process of Many Deep Networks Explores the Same Low-Dimensional Manifold*) by [Jialin Mao](https://www.linkedin.com/in/jialin-mao-339346182), [Itay Griniasty](https://scholar.google.co.il/citations?user=a3Uhp58AAAAJ&hl=en), [Han Kheng Teoh](https://www.linkedin.com/in/han-kheng-teoh-09392a70), [Rubing Yang](https://www.amcs.upenn.edu/people/rubing-yang), [Mark K. Transtrum](https://physics.byu.edu/faculty/transtrum/index), [James P. Sethna](https://sethna.lassp.cornell.edu) and [Pratik Chaudhari](https://pratikac.github.io/).

**Understanding the techniques from information geometry**: A good place to start would be [this jupyter notebook](https://colab.research.google.com/github/grasp-lyrl/picture_of_space_of_tasks/blob/main/picture_of_tasks_tutorial.ipynb). It contains a brief overview of the techniques and only takes a few minutes to run on Google Colab.


We build a qualitative and quantitative picture of the trajectories of representations. For example in the below figure, we plot the visual representations learnt by networks trained on different subsets of ImageNet. Interestingly, the trajectories of the representations resemble the Wordnet phylogenetic tree, which was built using only natural language-based semantics. 

<p align="center">
<img src="./plots/imagenet/tree.png" width="600">
</p>

We can use these techniques to compare representations learnt using
different *datasets* and using different *methods*---making them usable across
many different settings. We study phenomena relating to supervised, meta- and
contrastive learning and fine-tuning by studying networks in prediction space. 


## Setup

I usually prefer [micromamba](https://mamba.readthedocs.io/en/latest/installation.html) as a package manager over anaconda and highly recommend it.
To install the packages run:

```
micromamba create -y -f env.yml
micromamba activate picture
```

## Usage

The below steps can be used to reproduce the supervised learning results on ImageNet. Feel free to send us an email if you want code to reproduce the other results.

**Step 1**: Generate network trajectories. We first train networks and store predictions at different points on the training trajectory. The folder [supervised_imagenet](./supervised_imagenet), describes how to train a network on ImageNet. 

You can skip this step and download the ImageNet trajectories from [this link](https://mega.nz/folder/lAU2EBTT#6NXdRnL2RUoZL06e2baP7A); Move the downloaded files to the `predictions/` folder.

**Step 2**: Analyze the trajectories. The folder [info_geometry](./info_geometry) contains code to generate InPCA embeddings and compare different trajectories.

```bash
cd info_geometry
python inpca.py
python trajectory.py
```


## Directory Structure

```
.
├── picture_of_tasks_tutorial.ipynb
├── README.md
├── LICENSE
├── env.yml
├── info_geometry          
│   ├── inpca.py           # Compute InPCA embedding on data
│   └── trajectory.py      # Analyze training trajectories
├── supervised_imagenet    
│   ├── 01_training        # Train models and store trajectory weights
│   └── 02_imprinting      # Imprint models and store predictions
├── predictions            # Folder to store trajectories
└── plots
```

