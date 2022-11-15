import sys
import os
import data_preprocessing
from model import Net

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


def evaluate(dataset, model, device, eval_output):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    true_y = []
    y_pred = []
    
    fname_list = []
    key_list = []
    sys_id_list = []
    key_list = []
    score_list = []

    for batch_x, batch_y, batch_meta in tqdm(data_loader):
        true_y.extend(batch_y.numpy())
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x,batch_y,is_test=True)
        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        _, batch_pred = batch_out.max(dim=1)

        num_correct += (batch_pred == batch_y).sum(dim=0).item() 
        y_pred.extend(batch_pred.cpu().detach().numpy())
        
        
        # add outputs
        fname_list.extend(list(batch_meta[1]))
        key_list.extend(
          ['bonafide' if key == 1 else 'spoof' for key in list(batch_meta[4])])
        sys_id_list.extend([dataset.sysid_dict_inv[s.item()]
                            for s in list(batch_meta[3])])
        score_list.extend(batch_score.tolist())
   
    print ('Testing Accuracy: {}'.format(100 * (num_correct / num_total)))
    
    with open(eval_output, 'w') as fh:
        for f, s, k, cm in zip(fname_list, sys_id_list, key_list, score_list):
            if dataset.is_eval:
                fh.write('{} {} {} {}\n'.format(f, s, k, cm))
            else:
                fh.write('{} {}\n'.format(f, cm))
    print('Result saved to {}'.format(eval_output))

    return true_y, y_pred, num_total,score_list



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_trained_model_path', type=str, required=True)
    parser.add_argument('--database_path', type=str, help='Change it to the directory of ASVSPOOF2019 database', required=True)
    parser.add_argument('--protocols_path', type=str, help='Change it to the directory of ASVSPOOF2019 (LA) protocols', required=True)
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


    # Testing Data loading
    transform = transforms.Compose([
        lambda x: pad(x, max_len = config['samp_len']),
        lambda x: Tensor(x)
    ])

    database_path = args.database_path
    label_path = args.protocols_path

    is_eval = True
    test_set = data_preprocessing.ASVDataset(data_path=database_path,label_path=label_path,is_train=False,is_eval=is_eval,transform=transform)


    # Load pre-trained model 
    if args.pre_trained_model_path:
        model.load_state_dict(torch.load(args.pre_trained_model_path,map_location=device))
    eval_output = 'eval_scores.txt'
    evaluate(test_set, model, device, eval_output)
