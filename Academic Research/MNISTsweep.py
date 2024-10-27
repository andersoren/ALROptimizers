import sys
sys.path.append("..")
import Custom_Optimizers
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm import trange, tqdm
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.login()

random_seed = 3
torch.manual_seed(random_seed)

sweep_config = {
    "name": "SGD+M-Random",
    "method": "random",
    "metric": {"goal": "minimize", "name": "Loss over latest 3000 patterns"},
    "parameters": {
        "optimizer": {"value": "SGD+M"},
        "learning_rate": {"values": [1e-4, 1e-3, 1e-2, 1e-1]},
        "minibatch_size": {"values": [5, 25, 100, 500]},
        #"lr_batch_size": {"values": [3000, 6000, 12000, 15000, 30000]},
        "epochs": {"value": 5}
    },
}

eta_m, eta_p = 0.5, 1.2   
min_lr, max_lr = 1e-10, 1

sweep_id = wandb.sweep(sweep=sweep_config, project="Updated-Adam-&-SGD-Tests")

def build_dataset(minibatch_size):
    loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                            batch_size=minibatch_size, shuffle=True)
    return loader

def build_network():
    network = Custom_Optimizers.CNN()
    return network.to(device)

def build_optimizer(network, optimizer, learning_rate, minibatch_size, lr_batch_size, eta_m, eta_p, min_lr, max_lr):
    if optimizer == "S-Rprop":
        optimizer = Custom_Optimizers.SRPROP(network.parameters(), M=minibatch_size, L=lr_batch_size, lr=learning_rate, etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr))
    elif optimizer == "SGDUpd":
        optimizer = Custom_Optimizers.SGDUPD(network.parameters(), M=minibatch_size, L=lr_batch_size, lr=learning_rate, etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr))
    elif optimizer == "AdamUpd":
        optimizer = Custom_Optimizers.ADAMUPD(network.parameters(), M=minibatch_size, L=lr_batch_size, lr=learning_rate, etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr))
    elif optimizer == "Adam":
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    elif optimizer == "SGD+M":
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    return optimizer

def train_epoch(network, loader, optimizer, minibatch_size, epoch, min_loss):
    network.train()
    loss_3000 = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss_3000 += loss
        loss.backward()
        optimizer.step()
        if ((batch_idx+1)*len(data))%3000==0:
            if minibatch_size<3000:
                loss_3000 /= (3000/len(data))
            if loss_3000 < min_loss:
                min_loss = loss_3000
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx+1) * len(data),
            len(loader.dataset), 100. * batch_idx / len(loader), loss_3000.item()))
            wandb.log({"Loss over latest 3000 patterns": loss_3000.item()})
            loss_3000 = 0
    return min_loss

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        loader = build_dataset(config.minibatch_size)
        network = build_network()
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate, config.minibatch_size, \
                                    0, eta_m, eta_p, min_lr, max_lr)

        min_loss = float('inf')

        for epoch in range(1, config.epochs+1):
            min_loss = train_epoch(network, loader, optimizer,  config.minibatch_size, epoch, min_loss)
            loader = Custom_Optimizers.Shuffle_MNIST_Data(loader, random_seed)

        # Ensure the wandb sweep uses the minimum loss and not final loss
        wandb.log({"Loss over latest 3000 patterns": min_loss.item()})


wandb.agent(sweep_id, train, count=20)