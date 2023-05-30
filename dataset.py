import os
import json
import torch
from torch.utils.data import Dataset

class VideoTrafficDataset(Dataset):
    def __init__(self, src_dir, seq_len=30, normalize=True):
        cat_data = []
        file_list = []
        for root, _dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".json"):
                    file_list.append(os.path.join(root, file))
        for file in file_list:
            st = file.index("-") + 1
            end = file.index("-", st)
            vt = (int(file[st:end])/ 100) - 1 # video type
            vt = [1 if i == vt else 0 for i in range(4)] # one-hot encoding
            file_data = []
            with open(file, "r") as f:
                content = json.load(f)
                for c in content[10:]: # 초기 10개의 데이터는 너무 불안정해서 삭제
                    row = []
                    try:
                        row.append(c["stats"]["bitrate"]["video"]["upload"])
                    except:
                        row.append(0)
                    try:
                        row.append(c["stats"]["bitrate"]["video"]["download"])
                    except:
                        row.append(0)
                    try:
                        row.append(c["stats"]["packetLoss"]["total"])
                    except:
                        row.append(0)
                    row.extend(vt)
                    file_data.append(row)
            file_data = torch.tensor(file_data)
            tot_len = file_data.size(0)
            stk_data = []
            for i in range(tot_len - seq_len + 1):
                stk_data.append(file_data[i:i+seq_len])
            file_data = torch.stack(stk_data, dim=0)
            cat_data.append(file_data)
        self.data = torch.cat(cat_data, dim=0)
        self.delta = self.data[:, -1, :3] - self.data[:, 0, :3] # sequence 내에서 값 변화량
        self.delta_max, self.delta_min = self.delta.max(dim=0)[0], self.delta.min(dim=0)[0]
       
        if normalize:
            self.data = self.normalize(self.data)
            self.orig_delta = self.delta[:]
            self.orig_delta_max, self.orig_delta_min = self.delta_max, self.delta_min
            self.delta = self.data[:, -1, :3] - self.data[:, 0, :3]
            self.delta_max, self.delta_min = self.delta.max(dim=0)[0], self.delta.min(dim=0)[0]
        
        self.delta_mean, self.delta_std = self.delta.mean(dim=0), self.delta.std(dim=0)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def normalize(self, x):
        '''
        Normalize each variable in x to [-1, 1] and save x.max and x.min to class
        '''
        self.max = x.max()
        self.min = x.min()
        return (2 * (x - x.min()) / (x.max() - x.min()) - 1)
    
    def denormalize(self, x):
        '''
        Revert
        '''
        if not hasattr(self, 'max') or not hasattr(self, 'min'):
            raise Exception('You must normalize first')
        return 0.5 * (x * self.max - x * self.min + self.max + self.min)
    
    def sample_deltas(self, number):
        return (torch.randn(number, 3) + self.delta_mean) * self.delta_std
    
    def normalize_deltas(self, x):
        return ((self.delta_max - self.delta_min) * (x - self.or_delta_min)/(self.or_delta_max - self.or_delta_min) + self.delta_min)