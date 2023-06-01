import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F

class VideoTrafficDataset(Dataset):
    norm = {}
    def __init__(self, src_dir, seq_len=30, normalize=True, type='train'):
        # type = ['train', 'test', 'validation']
        self.columns = ['throughput', 'packetLoss', 'delay', 'jitter', 'availableBandwidth', 'layerIndex']
        self.available_layer_index = [0, 1, 2, 8, 9, 10, 16, 17, 18]
        self.data_num = 4
        if os.path.exists(f'{src_dir}/data_seq={seq_len}_type={type}.ts'):
            self.data = torch.load(f'{src_dir}/data_seq={seq_len}_type={type}.ts')
        else:
            data = []
            df = pd.read_csv(f'{src_dir}/data.csv')
            tot_len = len(df)
            if type == 'train':
                df = df.iloc[:(tot_len // 10) * 8]
            elif type == 'test':
                df = df[(tot_len // 10) * 8:(tot_len // 10) * 9]
            elif type == 'validation':
                df = df[(tot_len // 10) * 9:]
            else:
                df = df
            # Sequence Data 추출
            # 1. source, target 쌍으로 splitting
            # 2. 각 splitted result 마다 sequence 추출
            df = df.groupby(['source', 'target'])
            groups = df.groups
            for idx, group_columns in enumerate(groups):
                group = df.get_group(group_columns)
                group = group[self.columns]
                if len(group) < seq_len:
                    continue
                tot_len = len(group)
                for i in range(tot_len - seq_len + 1):
                    d = torch.tensor(group.iloc[i:i+seq_len].values)
                    data.append(d)
                print(f'extracting data [{idx}/{len(groups)}]')
            self.data = torch.stack(data, dim=0)
            torch.save(self.data, f'{src_dir}/data_seq={seq_len}_type={type}.ts')
        batch_size = self.data.size(0)
        availableBandwidth = self.data[:,:,self.columns.index('availableBandwidth')].view(batch_size, seq_len, 1)
        layerIdx = self.data[:,:,self.columns.index('layerIndex')]
        layerIdx = layerIdx.view(batch_size, seq_len, 1) == torch.tensor(self.available_layer_index).repeat(batch_size, seq_len, 1) # one hot encoding
        self.condition = torch.cat([availableBandwidth, layerIdx], dim=2)
        self.data = self.data[:,:,:self.data_num]

        self.delta = self.data[:, -1, :] - self.data[:, 0, :]  #sequence 내에서 값 변화량
        self.delta_max, self.delta_min = self.delta.max(dim=0)[0], self.delta.min(dim=0)[0]

        if normalize:
            self.data = self.normalize(self.data, 'data')
            self.condition[:,:, 0] = self.normalize(self.condition[:,:,0].view(-1, seq_len, 1), 'condition').view(-1, seq_len)
            self.orig_delta = self.delta[:]
            self.orig_delta_max, self.orig_delta_min = self.delta_max.clone(), self.delta_min.clone()
            self.delta = self.data[:, -1, :] - self.data[:, 0, :]
            self.delta_max, self.delta_min = self.delta.max(dim=0)[0], self.delta.min(dim=0)[0]
        
        self.delta_mean, self.delta_std = self.delta.mean(dim=0), self.delta.std(dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx],    # (throughput, packetLoss, delay, jitter)
            self.condition[idx], # (Available Bandwidth, Layer Index(one hot))
        )

    def normalize(self, x, label):
        x_max = (x.max(dim=1)[0]).max(dim=0)[0]
        x_min = (x.min(dim=1)[0]).min(dim=0)[0]
        self.norm[f'{label}_max'] = x_max
        self.norm[f'{label}_min'] = x_min
        return (2 * (x - x_min)) / (x_max - x_min) - 1

    def denormalize(self, x, label):
        x_max = self.norm[f'{label}_max']
        x_min = self.norm[f'{label}_min']
        return 0.5 * (x * x_max - x * x_min + x_max + x_min)

    def sample_deltas(self, number):
        return (torch.randn(number, self.data_num) + self.delta_mean) * self.delta_std

    def normalize_deltas(self, x):
        return ((self.delta_max - self.delta_min) * (x - self.orig_delta_min)/(self.orig_delta_max - self.orig_delta_min) + self.delta_min)