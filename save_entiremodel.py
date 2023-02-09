import argparse
import os
import torch

from sslsv.utils.helpers import load_config, load_model

import warnings

warnings.filterwarnings('ignore')


def make(args):
    config, checkpoint_dir = load_config(args.config)

    #model = load_model(config).to(device)
    model = load_model(config)
    torch.save(model,'model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    make(args)