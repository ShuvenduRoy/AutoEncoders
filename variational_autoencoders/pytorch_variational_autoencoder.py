import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tnrange
from pathlib import Path

batch_size = 128

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root=str(Path.home()) + '/Documents/datasets/mnist',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root=str(Path.home()) + '/Documents/datasets/mnist',
                           train=False,
                           transform=transforms.ToTensor())

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# parameters
input_dim = 784


# Defining the model
class VAE(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=500):
        super(VAE, self).__init__()
        self.fc_e = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_d1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_d2 = nn.Linear(hidden_dim, 784)

    def encoder(self, x_in):
        x = F.relu(self.fc_e(x_in.view(-1, 784)))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

    def decoder(self, z):
        z = F.relu(self.fc_d1(z))
        x_out = F.sigmoid(self.fc_d2(z))
        return x_out.view(-1, 1, 28, 28)

    def sample_normal(self, mean, logvar):
        # Using torch.normal(means,sds) returns a stochastic tensor which we cannot backpropogate through.
        # Instead we utilize the 'reparameterization trick'.
        # http://stats.stackexchange.com/a/205336
        # http://dpkingma.com/wordpress/wp-content/uploads/2015/12/talk_nips_workshop_2015.pdf
        sd = torch.exp(logvar * 0.5)
        e = Variable(torch.randn(sd.size()))  # sample from standard normal
        z = e.mul(sd).add_(mean)
        return z

    def forward(self, x_in):
        z_mean, z_logvar = self.encoder(x_in)
        z = self.sample_normal(z_mean, z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mean, z_logvar


model = VAE()


# Loss function
def criterion(x_out, x_in, z_mu, z_logvar):
    bce_loss = F.binary_cross_entropy(x_out, x_in, size_average=False)
    kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
    loss = (bce_loss + kld_loss) / x_out.size(0)  # normalize by batch size
    return loss


# optimizer
optimizer = torch.optim.Adam(model.parameters())


# Training
def train(model, optimizer, dataloader, epochs=5):
    losses = []
    for epoch in tnrange(epochs, desc='Epochs'):
        for iter, (images, _) in enumerate(dataloader):
            x_in = Variable(images)
            optimizer.zero_grad()
            x_out, z_mu, z_logvar = model(x_in)
            loss = criterion(x_out, x_in, z_mu, z_logvar)
            loss.backward()
            optimizer.step()
            losses.append(loss.data[0])

            if(iter+1)%100 ==0:
                print("Epoch: [{0}/{1}], Iter: [{2}/{3}] loss: {4}".format(epoch+1, epochs, iter+1, len(dataloader), loss.data[0]))
    return losses


if __name__ == '__main__':
    train_losses = train(model, optimizer, train_loader)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.show()

    # save the model
    torch.save(model.state_dict(), 'model.pkl')
