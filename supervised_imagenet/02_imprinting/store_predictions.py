import random
import os
import numpy as np
import torch
import json
import torch.nn as nn
import hydra
import h5py
import hdf5plugin

from glob import glob
from collections import OrderedDict
from torchvision.models import resnet50

import torch as ch
import torch.nn.functional as F

from utils.net import BlurPoolConv2d, get_model
from utils.data import create_loader
from utils.runner import compute_prototype, store_outputs


def set_seed(seed=0):
    """
    Don't set true seed to be nearby values. Doesn't give best randomness
    """
    rng = np.random.default_rng(seed)
    true_seed = int(rng.integers(2**30))

    random.seed(true_seed)
    np.random.seed(true_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(true_seed)
    torch.cuda.manual_seed_all(true_seed)


def get_dirs(seed_dir):
    """
    Get list of checkpoints in sorted order
    """
    ck_dir = os.path.join(seed_dir, "ckpt/*.pth")
    ckpts = list(glob(ck_dir))
    get_dir_ep = lambda x: int(x.split("/")[-1].split("_")[0])
    get_dir_bn = lambda x: int(x.split("_")[-1].split(".")[0])

    ckpts = [(get_dir_ep(d), get_dir_bn(d), d) for d in ckpts]

    ckpt_count = np.zeros(500)
    ckpts = sorted(ckpts)
    ckpts_reduced = []

    for dinfo in ckpts:
        ckpt_count[dinfo[0]] += 1
        if ckpt_count[dinfo[0]] > 5:
            continue
        if dinfo[0] >= 5 and dinfo[1] > 0:
            continue
        ckpts_reduced.append(dinfo[2])

    return ckpts_reduced


@hydra.main(config_path="./config", config_name="conf.yaml", version_base="1.2")
def main(cfg):
    print("Imprinting Models:")
    set_seed(cfg.seed)

    # Assumes atleast one GPU exists
    assert(torch.cuda.is_available())
    dataloaders = []
    dataloaders.append(create_loader(cfg.eval.imprint_data, True))
    dataloaders.append(create_loader(cfg.eval.predict_data, True))

    # Create directory for storing predictions
    tag = cfg.eval.ckpt_dir.split("/")[-1]
    fdir = '../../predictions/' + tag
    os.makedirs(fdir, exist_ok=True)

    metrics = {}

    # Get predictions for each seed
    ckpt_dir = os.path.join(cfg.eval.ckpt_dir, "seed*")
    for cdir in glob(ckpt_dir):
        dirs_reduced = get_dirs(cdir)
        seed = int(cdir.split("/")[-1].split("_")[-1])

        metrics[seed] = {}
        all_probs = []

        for idx, ckpt in enumerate(dirs_reduced):
            ckpt_name = ckpt.split("/")[-1]
            ckpt_ep = int(ckpt_name.split("_")[0])

            if ("_0.pth" not in ckpt) and (ckpt_ep >= 5):
                continue
            print("\t- %s" % ckpt)

            # Get network and compute imprinted weights
            net = get_model(ckpt)
            proto = compute_prototype(cfg, dataloaders[0], net)

            # Store outputs of the network
            ret, probs = store_outputs(cfg, dataloaders[1], net)
            if ckpt_ep not in metrics[seed]:
                metrics[seed][ckpt_ep] = ret
            all_probs.append(probs)

            with open(fdir + '/metrics.json', 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=4)


        all_probs = np.array(all_probs)

        # We use np.round to reduce the size of the file. We do not round
        # in the paper and store the data as npy files but this increases
        # the disk space by 10x
        all_probs = np.round(all_probs, 5).astype(np.float16)

        fname = fdir + '/preds_' + str(seed) + '.h5'
        with h5py.File(fname, 'w') as fp:
            fp.create_dataset('data', data=all_probs, **hdf5plugin.LZ4())


if __name__ == '__main__':
    main()
