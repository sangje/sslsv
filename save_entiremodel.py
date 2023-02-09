import argparse
import os
import torch

from sslsv.utils.helpers import load_config, load_model
from collections import OrderedDict

import warnings

warnings.filterwarnings('ignore')


def make(args):
    config, checkpoint_dir = load_config(args.config)

    #model = load_model(config).to(device)
    model = load_model(config)

    state_dict = torch.load('checkpoints/mfa_1024_vox1/model.pt')['model']
    temp = OrderedDict()
    for i, j in state_dict.items():   # search all key from model
        name = i.replace("head.","")  # change key that doesn't match
        temp[name] = j
    model.load_state_dict(temp, strict=False)

    torch.save(model,'model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    make(args)