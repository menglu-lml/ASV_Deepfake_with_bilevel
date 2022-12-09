import sys
import os
import data_preprocessing
from model import Net

import numpy as np
import yaml

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter
from tqdm import tqdm


def pad(x, max_len = 64600):
    
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    
    return padded_x

def idx_preprocessing(train_set): 
    idx0 = torch.tensor(train_set.data_sysid) == 0    # get index for each generation method
    idx1 = torch.tensor(train_set.data_sysid) == 1
    idx2 = torch.tensor(train_set.data_sysid) == 2
    idx3 = torch.tensor(train_set.data_sysid) == 3
    idx4 = torch.tensor(train_set.data_sysid) == 4
    idx5 = torch.tensor(train_set.data_sysid) == 5
    idx6 = torch.tensor(train_set.data_sysid) == 6

    train_mask_0 = idx0.nonzero().reshape(-1)
    real_idx = torch.split(train_mask_0, int(len(train_mask_0)/6)) # equally divide real data set  

    train_idx_1 = torch.cat((real_idx[0],idx1.nonzero().reshape(-1)))
    train_idx_2 = torch.cat((real_idx[1],idx2.nonzero().reshape(-1)))
    train_idx_3 = torch.cat((real_idx[2],idx3.nonzero().reshape(-1)))
    train_idx_4 = torch.cat((real_idx[3],idx4.nonzero().reshape(-1)))
    train_idx_5 = torch.cat((real_idx[4],idx5.nonzero().reshape(-1)))
    train_idx_6 = torch.cat((real_idx[5],idx6.nonzero().reshape(-1)))
    
    train_idx = torch.stack((train_idx_1,train_idx_2,train_idx_3,train_idx_4,train_idx_5,train_idx_6))
    
    return train_idx

## For Optimizer
def rate_A(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def rate_B(current_step,  num_training_steps, num_cycles):
    num_warmup_steps = int(0.04 * num_training_steps)
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))


## For Loss function
class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha]).cuda()
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha).cuda()

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)


# get validation accuracy
def validate(data_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y, batch_meta in tqdm(data_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x,batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)


def train_epoch_with_swap(train_A_loader, train_B_loader, idx, model_A, model_B,
                          optim_A, optim_B, device, select_param_vec, scheduler_A = None, scheduler_B = None):  
    running_loss = 0
    num_correct = 0.0
    model_A.train()
    
    weight = torch.FloatTensor([1.0, 1.0]).to(device)
    CEloss = nn.CrossEntropyLoss(weight=weight)
    
    for train_A_set, train_B_set in tqdm(zip(train_A_loader,train_B_loader), total = len(train_A_loader)):
        
        A_x = train_A_set[0]
        A_y = train_A_set[1]
        batch_size = A_x.size(0)
        idx += 1
        
        A_x = A_x.to(device)
        A_y = A_y.view(-1).type(torch.int64).to(device)
        A_out = model_A(A_x,A_y)
        
        CE_Loss_from_A = CEloss(A_out, A_y)
        Focal_Loss_from_A = FocalLoss()(A_out, A_y)

        batch_loss_from_A = CE_Loss_from_A + Focal_Loss_from_A
        _, batch_pred_from_A = A_out.max(dim=1)

        batch_acc = ((batch_pred_from_A == A_y).sum(dim=0).item() / batch_size)*100
        batch_loss = batch_loss_from_A.item() 
  
        
        optim_A.zero_grad()
        batch_loss_from_A.backward()
        optim_A.step()
        if scheduler_A !=None:
            scheduler_A.step()  # get new learning rate
        
        writer.add_scalar('train_accuracy', batch_acc, idx)
        writer.add_scalar('loss', batch_loss, idx)
        
      
        # update model_B with the parameter value from model_A
        model_A_dict = model_A.state_dict()
        model_B_dict = model_B.state_dict()
        param_dict = {k: v for k, v in model_A_dict.items() if k not in select_param_vec}
        model_B_dict.update(param_dict)
        model_B.load_state_dict(model_B_dict)
        model_B.train()
        
        B_x = train_B_set[0]
        B_y = train_B_set[1]   
        B_x = B_x.to(device)
        B_y = B_y.view(-1).type(torch.int64).to(device)
        B_out = model_B(B_x,B_y)
        
        CE_Loss_from_B = CEloss(B_out, B_y)
        Focal_Loss_from_B = FocalLoss()(B_out, B_y)

        batch_loss_from_B = CE_Loss_from_B + Focal_Loss_from_B
        _, batch_pred_from_B = B_out.max(dim=1)
        
        optim_B.zero_grad()
        batch_loss_from_B.backward()
        optim_B.step()
        if scheduler_B !=None:
            scheduler_B.step() 

    return idx


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
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
    num_epochs = config['epoch']
    lr = config['lr']
    warmup = config['warmup']  

    # Model Initialization
    model_A = Net(config['model'],device).to(device)
    model_B = Net(config['model'],device).to(device)

    # Adam optimizer
    optim_A = torch.optim.Adam(model_A.parameters(), lr = lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler_A = LambdaLR(optimizer=optim_A,
                              lr_lambda=lambda step: rate_A(step, d_model, factor=1, warmup=warmup),)

    optim_B = torch.optim.Adam(model_B.parameters(), lr = lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler_B = LambdaLR(optimizer=optim_B,
                              lr_lambda=lambda step: rate_B(step, num_training_steps=55000, num_cycles=0.5),)

    select_param_vec = [k for k, v in model_A.state_dict().items() if k.startswith("featureEncoder")]
    
    # To save model
    tag = '{}_{}_{}_{}'.format(config['batch_size'], config['model']['num_filter'], config['model']['patch_embed'],lr)
    model_save_path = os.path.join('SAVED_MODELS', tag)
    if os.path.exists(model_save_path)==False:
        os.makedirs(model_save_path)
    writer = SummaryWriter('logs/{}'.format(tag))


    # Data loading
    transform = transforms.Compose([
        lambda x: pad(x, max_len = config['samp_len']),
        lambda x: Tensor(x)
    ])

    database_path = args.database_path
    label_path = args.protocols_path

    train_set = data_preprocessing.ASVDataset(data_path=database_path,label_path=label_path,is_train=True,transform=transform)
    train_idx = idx_preprocessing(train_set)

    validate_set = data_preprocessing.ASVDataset(data_path = database_path,label_path = label_path,is_train=False, transform=transform)
    validate_loader = DataLoader(validate_set, batch_size=config['batch_size'], shuffle=True)


    # Start training
    best_valid_acc = 40
    index = 0

    for epoch in range(num_epochs):
        train_fold_idx = epoch % config['data']['k_fold']
        
        for i in range(1,len(train_fold_list[train_fold_idx][0])):  # get index from train_A set
            if i == 1:
                idx = train_fold_list[train_fold_idx][0][i]
                train_A_indices = torch.concat((train_idx[0],train_idx[1]))
            else:
                idx = train_fold_list[train_fold_idx][0][i]
                train_A_indices = torch.concat((train_A_indices,train_idx[idx]))
                
        train_B_set_idx1 = train_fold_list[train_fold_idx][1][0]
        train_B_set_idx2 = train_fold_list[train_fold_idx][1][1]
        train_B_indices = torch.concat((train_idx[train_B_set_idx1],train_idx[train_B_set_idx2]))
        
        train_A_set = Subset(train_set, train_A_indices)
        train_B_set = Subset(train_set, train_B_indices)
        
        # get two sets of training data for each model
        train_A_loader = DataLoader(train_A_set, batch_size=batch_size, shuffle=True, drop_last=True)
        train_B_loader = DataLoader(train_B_set, batch_size=batch_size, shuffle=True, drop_last=True)
        
        index = train_epoch_with_swap(train_A_loader, train_B_loader, index,model_A, model_B, optim_A, optim_B, 
                                        device, select_param_vec, scheduler_A = lr_scheduler_A, scheduler_B = lr_scheduler_B)
        
        if epoch < num_epochs-1:
            model_A_dict = model_A.state_dict()
            model_B_dict = model_B.state_dict()
            param_dict = {k: v for k, v in model_B_dict.items() if k in select_param_vec}
            model_A_dict.update(param_dict)
            model_A.load_state_dict(model_A_dict)
            
        valid_acc = validate(validate_loader, model_A, device)
        writer.add_scalar('valid_accuracy', valid_acc, epoch)
        print('\n{} - {:.4f}'.format(epoch, valid_acc))
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model_A.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))

    writer.close()    
