import collections
import librosa, librosa.display
import soundfile as sf

import torch
from torch import Tensor
from torchvision import transforms
from torchaudio import transforms as audioTran
from torch.utils.data import Subset
from torch import nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import os
import math
import yaml
import copy


    ######### helper function / class ############
## For PatchEmbed
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)

## For Transformer
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta






class SincConv(nn.Module): 
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size=80, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv,self).__init__()

        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel) 

        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) 
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);

        n = (self.kernel_size - 1) / 2.0
        self.band_pass = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate 

 


    def forward(self, x):
        self.band_pass = self.band_pass.to(x.device)

        self.window_ = self.window_.to(x.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)
        
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.band_pass)
        f_times_t_high = torch.matmul(high, self.band_pass)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.band_pass/2))*self.window_ 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)
 
        band_pass = band_pass / (2*band[:,None])
        

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1) 




class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, feature_size, patch_size, embed_dim, in_chans=1):
        super(PatchEmbed,self).__init__()
        patch_size = to_2tuple(patch_size)
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, H, W = x.shape
        x = x.view(B, 1, H, W)
        x = self.proj(x)
        x = x.flatten(2)   # (Batch, Channel, Feature)
        return x


class EmbedReduce(nn.Module):  # reduce the number of features
    def __init__(self, current_len, seq_size):
        
        super(EmbedReduce,self).__init__()
        self.linear1=nn.Linear(current_len, seq_size[0])
        self.lin_ln1 = nn.LayerNorm(seq_size[0])
        self.linear2=nn.Linear(seq_size[0], seq_size[1])
        self.lin_ln2 = nn.LayerNorm(seq_size[1])
        self.linear3=nn.Linear(seq_size[1], seq_size[2])
        self.lin_ln3 = nn.LayerNorm(seq_size[2])
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.lin_ln1(x)
        x = self.linear2(x)
        x = self.lin_ln2(x)
        x = self.linear3(x)
        x = self.lin_ln3(x)

        return x.transpose(1, 2)  # (Batch, Feature, Channel)

    
class FeatureEncoder(nn.Module):    # extract embedding for melspectrogram
    def __init__(self, encoder_channel, kernel, stride, padding):
        super(FeatureEncoder, self).__init__()
        
        self.mel_transform = audioTran.MelSpectrogram(16000)
        
        self.conv1 = nn.Conv2d(1, encoder_channel[0], kernel, stride=stride, padding=padding)
        nn.init.kaiming_uniform_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        self.encode1 = nn.Sequential(self.conv1, nn.BatchNorm2d(encoder_channel[0]), nn.LeakyReLU(0.2))
        
        
        self.conv2 = nn.Conv2d(encoder_channel[0], encoder_channel[1], kernel,stride=stride, padding=padding)
        nn.init.kaiming_uniform_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        self.encode2 = nn.Sequential(self.conv2,nn.BatchNorm2d(encoder_channel[1]), nn.LeakyReLU(0.2))
        
        
        self.conv3 = nn.Conv2d(encoder_channel[1], encoder_channel[2], kernel,stride=stride, padding=padding)
        nn.init.kaiming_uniform_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        self.encode3 = nn.Sequential(self.conv3, nn.BatchNorm2d(encoder_channel[2]), nn.LeakyReLU(0.2))
        
        self.conv4 = nn.Conv2d(encoder_channel[2], encoder_channel[3], kernel, stride=stride, padding=padding)
        nn.init.kaiming_uniform_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        self.encode4 = nn.Sequential(self.conv4, nn.BatchNorm2d(encoder_channel[3]), nn.LeakyReLU(0.2))
        

    def forward(self, x):
        x = self.mel_transform(x)
        
        batch = x.shape[0]
        freq = x.shape[1]
        frame = x.shape[2]
        x = x.view(batch,1,freq,frame)
        
        x = self.encode1(x)
        x = self.encode2(x)
        x = self.encode3(x)
        x = self.encode4(x)

        x = x.flatten(2)
        return x.transpose(1, 2) 
    
  


##########      implementation of Transformer      ##########   

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)  # wont be trained by the model wont update
        

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False) #freeze parameters for training
        return self.dropout(x)  
    
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    
    
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
    
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0     
        self.d_k = d_model // h  # assume d_v always equals d_k
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
    
    
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, h, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_hidden,)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
    
class Encoder(nn.Module):
    def __init__(self, d_model, d_hidden, N, h, dropout):
        super(Encoder, self).__init__()
        self.layers = clones(EncoderLayer(d_model, d_hidden, h, dropout), N)
        #self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        #return self.norm(x)   
        return x
    
class Transformer(nn.Module):
    def __init__(self, d_model, d_hidden, N, h,dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        
        # apply encoder only for classification task
        self.encoder = Encoder(self.d_model, self.d_hidden, N, h, dropout)
        
        print('initialization: xavier')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x):
        src_mask = torch.cat((torch.zeros(x.shape[0],1,1),torch.ones(x.shape[0],1,x.shape[1]-1)),dim=2).cuda()
        x = self.encoder(x,src_mask)
        return x


class Net(nn.Module):
    def __init__(self, model_args, device):
        super(Net, self).__init__()
        
        self.device = device

        self.Sinc_conv = SincConv(out_channels = model_args['num_filter'],
                                  kernel_size = model_args['filt_len'],
                                  in_channels = model_args['in_channels'])
        self.feature_dim = int((model_args['samp_len']-model_args['filt_len'])/model_args['max_pool_len']) 
        self.lnorm1 = LayerNorm([model_args['num_filter'],self.feature_dim])
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.d_embed = model_args['patch_embed']
        self.patchEmbed = PatchEmbed(feature_size = (model_args['num_filter'], self.feature_dim),
                                     patch_size = model_args['patch_size'],
                                     embed_dim = self.d_embed)
        self.seq_len = (model_args['num_filter']//model_args['patch_size'])*(self.feature_dim//model_args['patch_size'])
        self.EmbedReduce = EmbedReduce(current_len = self.seq_len, seq_size = model_args['seq_size'])
        
        self.FeatureEncoder = FeatureEncoder(model_args['encoder_channel'], model_args['kernel'], 
                                             model_args['stride'], model_args['padding'])
        
        
        self.posEncode = PositionalEncoding(d_model = self.d_embed, dropout = model_args['drop_out'])
        
        self.transformer = Transformer(d_model = self.d_embed, 
                                       d_hidden = model_args['encoder_hidden'], 
                                       N = model_args['num_block'],
                                       h = model_args['num_head'],
                                       dropout = model_args['drop_out'])
        
        self.gru = nn.GRU(input_size = self.d_embed, hidden_size = model_args['gru_hidden'],
                          num_layers = model_args['gru_layer'], batch_first = True)
        
        self.mlp = nn.Sequential(nn.LayerNorm(model_args['mlp_size'][0]),
                                 nn.Linear(in_features = model_args['mlp_size'][0], out_features = model_args['mlp_size'][1]),
                                 nn.Linear(in_features = model_args['mlp_size'][1], out_features = model_args['nb_classes']))
        
       
    def forward(self, x, y = None,is_test=False):
        batch = x.shape[0]
        len_seq = x.shape[1]

        # extracting SincNet features
        x_1 = x.view(batch,1,len_seq)
        
        x_1 = self.Sinc_conv(x_1)   
        x_1 = F.max_pool1d(torch.abs(x_1), 3)
        x_1 = self.lnorm1(x_1)
        x_1 = self.leaky_relu(x_1)
        
        x_1 = self.patchEmbed(x_1)
        x_1 = self.EmbedReduce(x_1)
        

        # extracting embedding features from Melspectrogram
        x_2 = self.FeatureEncoder(x)

        
        # combine two sets of features and pass to transformer for classification
        x = torch.cat((x_1, x_2), 1)
        x = self.posEncode(x)
        
        x = self.transformer(x)
        
        self.gru.flatten_parameters()
        x,h_n = self.gru(x)
        
        x = x.transpose(1, 2) 
        
        x = self.mlp(x.mean(dim=1))
        output=F.softmax(x,dim=1)
        
        return output
