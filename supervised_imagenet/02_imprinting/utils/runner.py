import os
import torch
import torch.nn as nn
import numpy as np
import time

import torch.optim as optim
import torch.nn as nn
import torch.cuda.amp as amp


def compute_prototype(cfg, dataloader, net):

    hid_dim = net.fc.weight.shape[1]
    net.fc = nn.Sequential()

    num_cls = cfg.eval.imprint_ncls
    prototypes = torch.zeros(num_cls, hid_dim)
    lab_count = torch.zeros(num_cls)

    prototypes = prototypes.cuda(0)
    lab_count = lab_count.cuda(0)

    net.eval()
    with torch.no_grad():
        for idx, (dat, labels) in enumerate(dataloader):
            with amp.autocast():
                rep = net(dat).float()

            if (~(rep.isfinite())).any():
                dat = dat.float()
                rep = net(dat)
                rep = torch.nan_to_num(rep, nan=0.0, posinf=0, neginf=0) 

            prototypes.index_add_(0, labels, rep)
            lab_count += torch.bincount(labels, minlength=num_cls)

    prototypes = prototypes / lab_count.reshape(-1, 1)
    net.prototypes = prototypes

    return prototypes


def store_outputs(cfg, dataloader, net):

    criterion = nn.CrossEntropyLoss()
     
    def euclid_dist(proto, rep):
        n = rep.shape[0]
        k = proto.shape[0]
        rep = rep.unsqueeze(1).expand(n, k, -1)
        proto = proto.unsqueeze(0).expand(n, k, -1)
        logits = -((rep - proto)**2).sum(dim=2)
        return logits

    all_out = []
    loss, acc, count = 0.0, 0.0, 0.0
    net.eval()

    prototypes = net.prototypes
    with torch.no_grad():
        for dat, labels in dataloader:
            batch_size = int(labels.size()[0])

            with amp.autocast():
                rep = net(dat).float()

            if (~(rep.isfinite())).any():
                dat = dat.float()
                rep = net(dat)
                rep = torch.nan_to_num(rep, nan=0.0, posinf=0, neginf=0) 

            out = euclid_dist(prototypes, rep).float()

            loss += (criterion(out, labels).item()) * batch_size

            labels = labels.cpu().numpy()
            out = out.cpu().detach()
            all_out.append(torch.nn.functional.softmax(out, dim=1))
            out = out.numpy()

            acc += np.sum(labels == (np.argmax(out, axis=1)))
            count += batch_size

    ret = {"accuracy": acc/count, "loss": loss/count}
    all_out = np.concatenate(all_out)

    return ret, all_out
