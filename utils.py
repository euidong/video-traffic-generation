import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

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
