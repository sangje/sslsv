import os
import numpy as np

import torch

import argparse
from sslsv.utils.helpers import load_config
from sslsv.data.AudioDataset1 import save

import warnings

warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()
    
    #print("--------------------------------------")
    configs, checkpoint_dir = load_config(args.config)
    save(configs.data)

