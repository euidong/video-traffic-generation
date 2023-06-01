import os
import torch
import random
import datetime
import torch.nn as nn
from utils import Option
import torch.optim as optim
from model import LSTMPredictor
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import VideoTrafficDataset

def main():
    opt = Option()
    opt.dataset_dir = "data"
    opt.batch_size = (2 ** 8)
    opt.seq_len = 30
    opt.epochs = 50
    opt.lr = 0.001
    opt.out_dir = "predictor/output_dir"
    opt.manual_seed = 9999
    opt.log_dir = "predictor/log_dir"
    opt.run_tag = "test"
    opt.checkpoint_every = 5
    opt.tensorboard_image_every = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cudnn.is_available():
        print('cudnn is available')
        cudnn.benchmark = True
    
    date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")

    run_name = f"{opt.run_tag}_{date}"
    log_dir_name = os.path.join(opt.log_dir, run_name)
    writer = SummaryWriter(log_dir_name)
    writer.add_text('Options', str(opt), 0)
    print(opt)

    try:
        os.makedirs(opt.out_dir)
    except OSError:
        pass
    output_dir = f"{opt.out_dir}/{run_name}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if opt.manual_seed is None:
        opt.manual_seed = random.randint(1, 10000)
    print(f"Random Seed: {opt.manual_seed}")
    torch.manual_seed(opt.manual_seed)

    train_dataset = VideoTrafficDataset(src_dir=opt.dataset_dir, seq_len=opt.seq_len, type='train')
    validation_dataset = VideoTrafficDataset(src_dir=opt.dataset_dir, seq_len=opt.seq_len, type='validation')
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=opt.batch_size, shuffle=True)

    seq_len = opt.seq_len
    data_dim = train_dataset[0][0].size(1)
    delta_dim = data_dim
    condition_dim = train_dataset[0][1].size(1)
    in_dim = data_dim + delta_dim + condition_dim
    out_dim = data_dim
    n_layers = 5

    model = LSTMPredictor(device, in_dim=in_dim, condition_dim=condition_dim, out_dim=out_dim, n_layers=n_layers).to(device)
    
    print('|Predictor Architecture|\n', model)

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    min_validation_loss = torch.inf
    patience = 3
    cnt = 0
    for epoch in range(opt.epochs):
        for i, (data, condition) in enumerate(train_dataloader):
            batch_size, seq_len = data.size(0), data.size(1)
            prev_seq = data[:, :-1, :]
            true_next = data[:,-1,:].to(device).to(torch.float32)
            delta = (prev_seq[:, -1] - prev_seq[:, 0]).view(batch_size, 1, delta_dim).repeat(1, seq_len -1, 1)
            prev_condition = condition[:,:-1, :]
            next_condition = condition[:, -1, :].to(device).to(torch.float32)
            prev_seq = torch.concat([prev_seq, delta, prev_condition], dim=2).to(device).to(torch.float32)

            model.zero_grad()
            output = model(prev_seq, next_condition)
            pred_next = output

            loss = criterion(pred_next, true_next)
            loss.backward()
            optimizer.step()


            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.epochs, i, len(train_dataloader), loss.item()))
            niter = epoch * len(train_dataloader) + i
            for name, param in model.named_parameters():
                writer.add_histogram("Gradients/{}".format(name), param.grad, niter)
            writer.add_scalar('Loss', loss.item(), niter)
        
        validation_loss = 0
        for i, (data, condition) in enumerate(validation_dataloader):
            batch_size, seq_len = data.size(0), data.size(1)
            prev_seq = data[:, :-1, :]
            true_next = data[:,-1,:].to(device).to(torch.float32)
            delta = (prev_seq[:, -1] - prev_seq[:, 0]).view(batch_size, 1, delta_dim).repeat(1, seq_len -1, 1)
            prev_condition = condition[:,:-1, :]
            next_condition = condition[:, -1, :].to(device).to(torch.float32)
            prev_seq = torch.concat([prev_seq, delta, prev_condition], dim=2).to(device).to(torch.float32)

            with torch.no_grad():
                output = model(prev_seq, next_condition)
                pred_next = output
                loss = criterion(pred_next, true_next)
                validation_loss += loss.item()
        if validation_loss < min_validation_loss:
            min_validation_loss = validation_loss
            cnt += 1
            torch.save(model.state_dict(), f"{output_dir}/predictor.pt")
        else:
            cnt = 0
        if patience <= cnt:
            break
    
    


if __name__ == "__main__":
    main()