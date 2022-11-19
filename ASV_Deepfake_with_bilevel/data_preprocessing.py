import torch
import collections
import os
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
import numpy as np
from joblib import Parallel, delayed



###### For ASV DATSET  ########
ASVFile = collections.namedtuple('ASVFile',['speaker_id', 'file_name', 'path', 'sys_id', 'key'])


class ASVDataset(Dataset):
    def __init__(self, data_path=None, label_path=None,transform=None, is_train=True,is_eval=False,feature=None,track=None):
        self.data_path_root = data_path
        self.label_path = label_path
        self.track = track
        self.feature = feature
        self.is_eval = is_eval
        self.transform = transform
        
        if self.is_eval:
            self.sysid_dict = {
            '-': 0,  # bonafide speech
            'A07': 1,
            'A08': 2, 
            'A09': 3, 
            'A10': 4, 
            'A11': 5, 
            'A12': 6,
            'A13': 7, 
            'A14': 8, 
            'A15': 9, 
            'A16': 10, 
            'A17': 11, 
            'A18': 12,
            'A19': 13,    
        }
        else:
            self.sysid_dict = {
            '-': 0,  # bonafide speech         
            'A01': 1, 
            'A02': 2, 
            'A03': 3, 
            'A04': 4, 
            'A05': 5, 
            'A06': 6,        
        }
        
        self.sysid_dict_inv = {v:k for k,v in self.sysid_dict.items()}
        print('sysid_dict_inv',self.sysid_dict_inv)
        
        self.dset_name = 'eval' if is_eval else 'train' if is_train else 'dev'
        print('dset_name',self.dset_name)
        
        self.label_fname = 'eval.trl' if is_eval else 'train.trn' if is_train else 'dev.trl'
        print('label_fname',self.label_fname)
        
        self.label_dir = os.path.join(self.label_path)
        print('protocols_dir',self.label_dir)
        
        track = 'LA' 
        self.prefix = 'ASVspoof2019_{}'.format(track)
        self.audio_files_dir = os.path.join(self.data_path_root, '{}_{}'.format(
            self.prefix, self.dset_name), 'flac')
        print('audio_files_dir',self.audio_files_dir)
        
        self.label_fname = os.path.join(self.label_dir,
            'ASVspoof2019.{}.cm.{}.txt'.format(track, self.label_fname))
        print('label_file',self.label_fname)
        
        if (self.dset_name == 'eval'):
            cache_fname = 'cache_ASV_{}.npy'.format(self.dset_name)
            self.cache_fname = os.path.join(cache_fname)
        else:
            cache_fname = 'cache_ASV_{}.npy'.format(self.dset_name)
            self.cache_fname = os.path.join(cache_fname)
            
        if os.path.exists(self.cache_fname):
            self.data_x, self.data_y, self.data_sysid, self.files_meta = torch.load(self.cache_fname)
            print('Dataset loaded from cache', self.cache_fname)
        else: 
            self.files_meta = self.parse_protocols_file(self.label_fname)
            data = list(map(self.read_file, self.files_meta))
            self.data_x, self.data_y, self.data_sysid = map(list, zip(*data))
            if self.transform:
                self.data_x = Parallel(n_jobs=5, prefer='threads')(delayed(self.transform)(x) for x in self.data_x)                          
            torch.save((self.data_x, self.data_y, self.data_sysid, self.files_meta), self.cache_fname)
        
    def __len__(self):
        self.length = len(self.data_x)
        return self.length
   
    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y, self.files_meta[idx]
            
    def read_file(self, meta):   
        #data_x, sample_rate = librosa.load(meta.path,sr=16000)  
        data_x, sample_rate = sf.read(meta.path)
        data_y = meta.key
        return data_x, float(data_y) ,meta.sys_id   

    def parse_line(self, line):
        tokens = line.strip().split(' ')
        if self.is_eval:
            return ASVFile(speaker_id=tokens[0],
                file_name=tokens[1],
                path=os.path.join(self.audio_files_dir, tokens[1] + '.flac'),
                sys_id=self.sysid_dict[tokens[3]],
                key=int(tokens[4] == 'bonafide'))
        return ASVFile(speaker_id=tokens[0],
            file_name=tokens[1],
            path=os.path.join(self.audio_files_dir, tokens[1] + '.flac'),
            sys_id=self.sysid_dict[tokens[3]],
            key=int(tokens[4] == 'bonafide'))

    def parse_protocols_file(self, label_fname):
        lines = open(label_fname).readlines()
        files_meta = map(self.parse_line, lines)
        return list(files_meta)
