import argparse
import os
import torch

from sslsv.Trainer import Trainer
from sslsv.utils.helpers import load_config, load_dataloader, load_model

import warnings

warnings.filterwarnings('ignore')

import torch.distributed as dist

def train(args):
    config, checkpoint_dir = load_config(args.config)
    train_dataloader = load_dataloader(config)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"
    #device = torch.device('cuda:0,1,2') if torch.cuda.is_available() else torch.device('cpu')
    #GPU_NUM = 0,1,2
    #device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(config)
    #model = load_model(config).to("cuda")
    model = torch.nn.DataParallel(model)
    model.to(device)
    print("Model Loaded!!  Let's use", torch.cuda.device_count(), "GPUs!")

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=config,
        checkpoint_dir=checkpoint_dir,
        device=device
    )
    trainer.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to model config file.')
    args = parser.parse_args()

    train(args)
