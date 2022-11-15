import sys
import os
from model import Net

import collections
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
from joblib import Parallel, delayed

import numpy as np
import yaml
import argparse

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import torch.nn.functional as F

from tqdm import tqdm



def pad(x, max_len = 64600):
    
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    
    return padded_x


def evaluate(data, model, device):

    data = data.to(device)
    data = data.view(1, data.shape[0])

    out = model(data,is_test=True)
    _, pred = out.max(dim=1)
    print ('bonafide' if pred == 1 else 'spoof')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_trained_model_path', type=str, required=True)
    parser.add_argument('--AUDIO_path', type=str, help='Change it to the testing audio file', required=True)
    args = parser.parse_args()
    
    np.random.seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Parameter
    config = yaml.safe_load(open('model_config.yaml'))

    # Model Initialization
    model = Net(config['model'],device).to(device)

    # Load pre-trained model 
    if args.pre_trained_model_path:
        model.load_state_dict(torch.load(args.pre_trained_model_path,map_location=device))


    audio_data, sample_rate = sf.read(args.AUDIO_path)
    audio_data = Tensor(pad(audio_data))

    evaluate(audio_data, model, device)
