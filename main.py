import os
import json
import torch
import random
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from dataset import VideoTrafficDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import LSTMDiscriminator, LSTMGenerator


def time_series_to_plot(data, dpi=35, feature_idx=0, n_images_per_row=4, titles=None):
    """Convert a batch of time series to a tensor with a grid of their plots
    
    Args:
        data (Tensor): (batch_size, seq_len, feature)
        feature_idx (int): index of the feature that goes in the plots (the first one by default)
        n_images_per_row (int): number of images per row in the plot
        titles (list of strings): list of titles for the plots

    Output:
        single (channels, width, height)-shaped tensor representing an image
    """
    images = []
    for i, seq in enumerate(data.detach()):
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(1,1,1)
        if titles:
            ax.set_title(titles[i])
        ax.plot(seq[:, feature_idx].numpy())
        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(img)
        plt.close(fig)
    
    images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
    grid_image = vutils.make_grid(images.detach(), nrow=n_images_per_row)
    return grid_image

class Option:
    def __str__(self):
        return str(vars(self))

opt = Option() # make empty object
opt.dataset_dir = "data"
opt.workers = 2
opt.batchSize = 16
opt.nz = 100
opt.epochs = 50
opt.lr = 0.0002
opt.outf = "output_folder"
opt.imf = "image_folder"
opt.manualSeed = 927
opt.logdir = "log_dir"
opt.run_tag = "test"
opt.checkpoint_every = 5
opt.tensorboard_image_every = 5
opt.dis_type = "cnn"
opt.gen_type = "lstm"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if cudnn.is_available():
    print('cudnn is available')
    cudnn.benchmark = True

date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")

run_name = f"{opt.run_tag}_{date}"
log_dir_name = os.path.join(opt.logdir, run_name)
writer = SummaryWriter(log_dir_name)
writer.add_text('Options', str(opt), 0)
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass
try:
    os.makedirs(opt.imf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print(f"Random Seed: {opt.manualSeed}")
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = VideoTrafficDataset(opt.dataset_dir)
dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

nz = int(opt.nz)
seq_len = dataset[0].size(0)
data_dim = 3
deleta_dim = data_dim
condition_dim = 4
in_dim_D = data_dim + deleta_dim + condition_dim
in_dim_G = opt.nz + deleta_dim + condition_dim
hidden_dim = 256
n_layers = 2

netD = LSTMDiscriminator(device, in_dim=in_dim_D, out_dim=data_dim, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
netG = LSTMGenerator(device, in_dim=in_dim_G, out_dim=data_dim, hidden_dim=hidden_dim, n_layers=n_layers).to(device)

print("|Discriminator Architecture|\n", netD)
print("|Generator Architecture|\n", netG)

criterion = nn.BCELoss().to(device)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr)


# Training

validation_noise = torch.randn(opt.batchSize, seq_len, nz)
deltas = dataset.sample_deltas(opt.batchSize).view(opt.batchSize, 1, deleta_dim).repeat(1, seq_len, 1)
condition = torch.tensor([0,0,0,1]).repeat(opt.batchSize, seq_len, 1)
validation_noise = torch.cat((validation_noise, condition, deltas), dim=2).to(device)


real = 1.
fake = 0.


for epoch in range(opt.epochs):
    for i, data in enumerate(dataloader):
        # 1. construct real data
        batch_size, seq_len = data.size(0), data.size(1)
        real_data = data.to(device)
        condition_data = real_data[:,:,3:]
        real_deltas = (real_data[:, -1, :3] - real_data[:, 0, :3]).view(batch_size, 1, deleta_dim).repeat(1, seq_len, 1).to(device)
        real_data = torch.cat((real_data, real_deltas), dim=2).to(device).type(torch.float)
        real_label = torch.full((batch_size, seq_len, data_dim), real, device=device)
        
        # 2. construct fake data
        train_noise = torch.randn(batch_size, seq_len, nz).to(device)
        train_deltas = dataset.sample_deltas(batch_size).view(batch_size, 1, deleta_dim).repeat(1, seq_len, 1).to(device)
        train_noise = torch.cat((train_noise, condition_data, train_deltas), dim=2).to(device)
        fake_data = netG(train_noise)
        fake_data = torch.cat((fake_data, condition_data, real_deltas), dim=2).to(device).type(torch.float) # TODO: 이게 reasonable한지 확인해보기.
        fake_label = torch.full((batch_size, seq_len, data_dim), fake, device=device)

        # 3. Train netD
        # -1. with real data
        netD.zero_grad()
        outputD_real = netD(real_data)
        errD_real = criterion(outputD_real, real_label)
        errD_real.backward()
        D_x = outputD_real.mean().item()
        # -2. with fake data
        outputD_fake = netD(fake_data)
        errD_fake = criterion(outputD_fake, fake_label)
        errD_fake.backward()
        D_G_z1 = outputD_fake.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # 4. Train netG
        netG.zero_grad()
        fake_data = netG(train_noise)
        fake_data = torch.cat((fake_data, condition_data, real_deltas), dim=2).to(device).type(torch.float) # TODO: 이게 reasonable한지 확인해보기.
        fake_label = torch.full((batch_size, seq_len, data_dim), fake, device=device)
        real_label = torch.full((batch_size, seq_len, data_dim), real, device=device)
        outputD2_fake = netD(fake_data)
        errG = criterion(outputD2_fake, real_label)
        errG.backward()
        D_G_z2 = outputD2_fake.mean().item()
        optimizerG.step()

        niter = epoch * len(dataloader) + i
        for name, param in netG.named_parameters():
            writer.add_histogram("GeneratorGradients/{}".format(name), param.grad, niter)
        
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' 
              % (epoch, opt.epochs, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), end='')
        print()
        writer.add_scalar('DiscriminatorLoss', errD.item(), niter)
        writer.add_scalar('GeneratorLoss', errG.item(), niter)
        writer.add_scalar('D of X', D_x, niter)
        writer.add_scalar('D of G of z', D_G_z1, niter)
    
    if (epoch % opt.tensorboard_image_every == 0) or (epoch == (opt.epochs - 1)):
        label = ["upload-bitrate", "download-bitrate", "packet-loss"]
        for i, l in enumerate(label):
            real_plot = time_series_to_plot(dataset.denormalize(data), feature_idx=i)
            writer.add_image(f"[Real]{l}", real_plot, epoch)
            fake_data = netG(validation_noise)
            fake_plot = time_series_to_plot(dataset.denormalize(fake_data.cpu()), feature_idx=i)
            writer.add_image(f"[Fake]{l}", fake_plot, epoch)
            # torchvision.utils.save_image(fake_plot, os.path.join(opt.imf, f'{opt.run_tag}_epoch{epoch}.jpg'))


torch.save(netG.state_dict(), f"{opt.outf}/generator.pt")
torch.save(netD.state_dict(), f"{opt.outf}/discriminator.pt")
