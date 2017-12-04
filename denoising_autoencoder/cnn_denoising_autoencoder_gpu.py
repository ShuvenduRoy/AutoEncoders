import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from pathlib import Path

batch_size = 64

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root=str(Path.home())+'/datasets/mnist',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root=str(Path.home())+'/datasets/mnist',
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
class CNN_Autoencoder(nn.Module):
    def __init__(self):
        super(CNN_Autoencoder, self).__init__()
        self.encoded = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.decoded = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=5, padding=2),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

    def forward(self, x):
        encoded = self.encoded(x)
        decoded = self.decoded(encoded)
        return encoded, decoded


# creating instance of model
model = CNN_Autoencoder()
model.cuda()

# Hyper Parameters
num_epochs = 50
learning_rate = 0.005

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# initialize figure
f, a = plt.subplots(2, 5, figsize=(5, 2))
plt.ion()  # continuously plot

# original data (first row) for viewing

x_test = test_dataset.test_data.numpy()
x_test_noisy = x_test + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
x_test_noisy = torch.FloatTensor(x_test_noisy)
view_data = Variable(x_test_noisy[:5].view(5, 1, 28, 28).type(torch.FloatTensor) / 255.)

for i in range(5):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray');
    a[0][i].set_xticks(());
    a[0][i].set_yticks(())

# train the model
for epoch in range(num_epochs):
    for step, (images, labels) in enumerate(train_loader):
        x_train = images.numpy()
        x_train_noisy = x_train + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        inputs = Variable(torch.FloatTensor(x_train_noisy)).cuda()

        targets = Variable(images).cuda()

        # Forward
        optimizer.zero_grad()
        encoded, decoded = model(inputs)

        # backward
        loss = criterion(decoded, targets)
        loss.backward()

        # optimize
        optimizer.step()

        if (step + 1) % 100 == 0:
            print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (
                epoch + 1, batch_size, step + 1, 60000 / batch_size, loss.data[0]))

            # # plotting decoded image (second row)
            _, decoded_data = model(view_data.cuda())
            for i in range(5):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.cpu().data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(());
                a[1][i].set_yticks(())
            plt.draw();
            plt.pause(0.05)

plt.ioff()
plt.show()

# save the model
torch.save(model.state_dict(), 'model.pkl')
