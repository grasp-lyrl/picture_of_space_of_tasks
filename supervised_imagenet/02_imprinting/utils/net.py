from torchvision.models import resnet50
import numpy as np
import torch
import torch.nn.functional as F

from collections import OrderedDict


def get_model(ckpt):
    net = resnet50(weights=None)
    ckpt_file = torch.load(ckpt)

    ncls = ckpt_file['module.fc.weight'].shape[0]
    hdim = ckpt_file['module.fc.weight'].shape[1]
    hid_dim = net.fc.in_features
    net.fc = torch.nn.Linear(hid_dim, ncls)

    def apply_blurpool(mod: torch.nn.Module):
        for (name, child) in mod.named_children():
            if isinstance(child, torch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                setattr(mod, name, BlurPoolConv2d(child))
            else: apply_blurpool(child)
    apply_blurpool(net)

    net = net.to(memory_format=torch.channels_last)
    net = net.to("cuda:0")

    nckpt = OrderedDict()
    for key in ckpt_file:

        if key.startswith("module."):
            nkey = key[key.find(".")+1:]
            nckpt[nkey] = ckpt_file[key]
    net.load_state_dict(nckpt)

    return net


class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)
