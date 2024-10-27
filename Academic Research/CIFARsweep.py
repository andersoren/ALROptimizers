import sys
sys.path.append("..")
import Custom_Optimizers
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import wandb
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.login()

random_seed = random.randint(0, 2**32 - 1)
torch.manual_seed(random_seed)

eta_m, eta_p = 0.7375, 1.2
min_lr, max_lr = 1e-6, 0.01

# Data augmentation transform
mean = (0.5071, 0.4866, 0.4409)  # (129.3, 124.1, 112.4) 
std = (0.2673, 0.2564, 0.2762)  # (68.2, 65.4, 70.4)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # Convert images to tensors
    torchvision.transforms.Normalize(mean, std),  # Normalize all pixel values
    torchvision.transforms.RandomHorizontalFlip(p=0.5),    # Horizontally flip augmented data with 50% chance of flipping
    torchvision.transforms.Pad(padding=4, padding_mode='reflect'),    # Pad with 4 pixels using reflection
    torchvision.transforms.RandomCrop(size=(32, 32))])    # Randomly centered crop back to original size

def build_dataset(minibatch_size):
    loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(root='./data',train=True, download=False,
                                        transform=transform), batch_size=minibatch_size, shuffle=True)
    return loader

def build_network():
    network = Custom_Optimizers.ResNet9(in_channels=3, num_classes=100, drop_rate=0) # Default Dropout is 0.2

    if torch.cuda.device_count() > 1:
        network = torch.nn.DataParallel(network)  # Parallelise data with two GPUs
        print("Using", torch.cuda.device_count(), "GPUs.")
    
    return network.to(device)

def build_optimizer(network, config, eta_m, eta_p, min_lr, max_lr):
    if config["optimizer"] == "S-Rprop":
        optimizer = Custom_Optimizers.SRPROP(network.parameters(), M=config["minibatch_size"], L=config["lr_batch_size"], \
                                             lr=config["learning_rate"], etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), \
                                              weight_decay=0, track_lr=False)
    elif config["optimizer"] == "SGD-Upd":
        optimizer = Custom_Optimizers.SGDUPD(network.parameters(), M=config["minibatch_size"], L=config["lr_batch_size"], \
                                             lr=config["learning_rate"], etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), \
                                             weight_decay=0, track_lr=False)
    elif config["optimizer"] == "Adam-Upd":
        optimizer = Custom_Optimizers.ADAMUPD(network.parameters(), M=config["minibatch_size"], L=config["lr_batch_size"], \
                                              lr=config["learning_rate"], etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), \
                                                weight_decay=0, track_lr=False)
    elif config["optimizer"] == "Adam":
        optimizer = optim.Adam(network.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "SGD+M":
        optimizer = optim.SGD(network.parameters(), lr=config["learning_rate"], momentum=0.9, weight_decay=0, dampening=0)
    else:
        print("Optimizer type not recognised")
    return optimizer

def train_epoch(network, loader, optimizer, epoch, min_loss, max_acc):
    network.train()
    training_loss=0
    correct=0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        loss.backward()
        training_loss += loss.item()
        optimizer.step()
        if (batch_idx+1)%100==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx+1) * len(data),
                  len(loader.dataset), 100. * batch_idx / len(loader), loss.item()))
    training_loss /= (len(loader.dataset)/len(data))
    training_acc = 100. * correct / len(loader.dataset)
    if training_loss < min_loss:
                min_loss = training_loss
    if training_acc > max_acc:
        max_acc = training_acc
    wandb.log({"Training loss": training_loss, "Training accuracy": 100. * correct / len(loader.dataset)})
    return min_loss, max_acc

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
    
        network = build_network()

        optimizer = build_optimizer(network, config, eta_m, eta_p, min_lr, max_lr)

        min_loss = float('inf')
        max_acc = -float('inf')

        for epoch in range(1, config["epochs"]+1):
            loader = build_dataset(config["minibatch_size"])
            min_loss, max_acc = train_epoch(network, loader, optimizer, epoch, min_loss, max_acc)  

        wandb.log({"Training loss": min_loss, "Training accuracy": max_acc})

###### -----------------------------------------------------------------------
######                   For hyper-parameter sweeps using training only

sweep_config = {
    "name": "Adam-Upd",
    "method": "grid",
    "metric": {"goal": "maximize", "name": "Training accuracy"},
    "parameters": {
        "optimizer": {"value": "Adam-Upd"},
        "learning_rate": {"values": [1e-4, 1e-3, 1e-2]},
        "minibatch_size": {"values": [20, 100, 500, 2500]},
        "lr_batch_size": {"values": [25000, 50000]},                          # Change to = 0 if algorithm does not use lr_batch_size
        "epochs": {"value": 50},
    },
}

sweep_id = wandb.sweep(sweep=sweep_config, project="CIFAR-100")

wandb.agent(sweep_id, train)