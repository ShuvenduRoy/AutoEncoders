import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable

torch.manual_seed(1)
batch_size = 64

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='C:\datasets\mnist',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='C:\datasets\mnist',
                           train=False,
                           transform=transforms.ToTensor())

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# define the model
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            # nn.Tanh(),
            # nn.Linear(128, 64),
            # nn.Tanh(),
            # nn.Linear(64, 12),
            # nn.Tanh(),
            # nn.Linear(12, 3),  # 3 feattures will be visualized in plt
        )

        self.decoder = nn.Sequential(
            # nn.Linear(3, 12),
            # nn.Tanh(),
            # nn.Linear(12, 64),
            # nn.Tanh(),
            # nn.Linear(64, 128),
            # nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# creating instance of model
model = AutoEncoder()

# Hyper Parameters
num_epochs = 10
learning_rate = 0.005

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# initialize figure
f, a = plt.subplots(2, 5, figsize=(5, 2))
plt.ion()  # continuously plot

# original data (first row) for viewing
view_data = Variable(train_dataset.train_data[:5].view(-1, 28 * 28).type(torch.FloatTensor) / 255.)
for i in range(5):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray');
    a[0][i].set_xticks(());
    a[0][i].set_yticks(())

# train the model
for epoch in range(2):
    for step, (images, labels) in enumerate(train_loader):
        inputs = Variable(images.view(-1, 28 * 28))
        targets = Variable(images.view(-1, 28 * 28))

        # Forward
        optimizer.zero_grad()
        encoded, decoded = model(inputs)

        # backward
        loss = criterion(decoded, targets)
        loss.backward()

        # optimize
        optimizer.step()

        if (step + 1) % 100 == 0:
            print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (epoch + 1, 80, step + 1, (60000)/batch_size, loss.data[0]))

            # plotting decoded image (second row)
            _, decoded_data = model(view_data)
            for i in range(5):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(());
                a[1][i].set_yticks(())
            plt.draw();
            plt.pause(0.05)

plt.ioff()
plt.show()


# save the model
torch.save(model.state_dict(), 'model.pkl')
