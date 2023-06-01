import torch
import torch.nn as nn

class LSTMGenerator(nn.Module):
    def __init__(self, device, in_dim, out_dim, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.Tanh())
    
    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        h, _ = self.lstm(input, (h_0, c_0))
        h = h.contiguous().view(batch_size * seq_len, self.hidden_dim)
        output = self.linear(h)
        output = output.view(batch_size, seq_len, self.out_dim)
        return output

class LSTMDiscriminator(nn.Module):
    def __init__(self, device, in_dim, out_dim, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.Sigmoid())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        h, _ = self.lstm(input, (h_0, c_0))
        h = h.contiguous().view(batch_size * seq_len, self.hidden_dim)
        output = self.linear(h)
        output = output.view(batch_size, seq_len, self.out_dim)
        return output
    
class LSTMPredictor(nn.Module):
    def __init__(self, device, in_dim, condition_dim, out_dim, n_layers=5):
        super(LSTMPredictor, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.condition_dim = condition_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(in_dim, in_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(in_dim + condition_dim, out_dim)

    def forward(self, prev_seq, next_condition):
        batch_size = prev_seq.size(0)
        h_0 = torch.zeros(self.n_layers, batch_size, self.in_dim).to(self.device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.in_dim).to(self.device)
        h, _ = self.lstm(prev_seq, (h_0, c_0))
        h = h[:, -1, :]
        h = torch.concat([h, next_condition], dim=1)
        output = self.fc(h)
        return output