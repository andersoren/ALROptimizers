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
import random


device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.login()

random_seed = random.randint(0, 2**32 - 1)
torch.manual_seed(random_seed)

eta_m, eta_p = 0.5, 1.2   
min_lr, max_lr = 1e-10, 1

def build_dataset(minibatch_size):
    loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                            batch_size=minibatch_size, shuffle=True)
    return loader

def build_test_dataset(batch_size_test):
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))])), 
                               batch_size=batch_size_test, shuffle=True)
    return test_loader

def build_network():
    network = Custom_Optimizers.CNN()
    return network.to(device)

def build_optimizer(network, optimizer, learning_rate, minibatch_size, lr_batch_size, eta_m, eta_p, min_lr, max_lr, track_lr):
    if optimizer == "S-Rprop":
        optimizer = Custom_Optimizers.SRPROP(network.parameters(), M=minibatch_size, L=lr_batch_size, lr=learning_rate, etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), track_lr=track_lr)
    elif optimizer == "SGDUpd":
        optimizer = Custom_Optimizers.SGDUPD(network.parameters(), M=minibatch_size, L=lr_batch_size, lr=learning_rate, etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), track_lr=track_lr)
    elif optimizer == "AdamUpd":
        optimizer = Custom_Optimizers.ADAMUPD(network.parameters(), M=minibatch_size, L=lr_batch_size, lr=learning_rate, etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), track_lr=track_lr)
    elif optimizer == "Adam":
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    elif optimizer == "SGD+M":
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == "SGD":
        optimizer = optim.SGD(network.parameters(), lr=learning_rate)
    return optimizer

def train_epoch(network, loader, optimizer, minibatch_size, epoch, lr_batch_size, track_lr, counter_3000, val, test_loader=None):
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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx+1) * len(data),
            len(loader.dataset), 100. * batch_idx / len(loader), loss_3000.item()))
            counter_3000 +=  1
            wandb.log({"Loss over latest 3000 patterns": loss_3000.item(), "steps_3000": counter_3000})
            loss_3000 = 0
            if val:
                network.eval()
                test(network, test_loader, counter_3000)
                network.train()
        if track_lr:
            if ((batch_idx+1)*len(data))%lr_batch_size==0:
                wandb.log({"Indiv. learning rate mean": optimizer.lr_mean[-1], \
                        "Indiv. learning rate std": optimizer.lr_std[-1], "LR_steps": optimizer.lr_counter})

# Default is that learning rates are tracked inside updated algorithms, and no scheduler used nor validation set
def train(config=None, schedule=False, track_lr=True, val=False):
    # Initialize a new wandb run
    run = wandb.init(project="MNIST2",
                    config=config)
    wandb.define_metric("steps_3000")
    wandb.define_metric("Loss over latest 3000 patterns", step_metric="steps_3000")

    if val:
        wandb.define_metric("val_steps")
        wandb.define_metric("Val loss", step_metric="val_steps")
        wandb.define_metric("Val accuracy", step_metric="val_steps")
        test_loader = build_test_dataset(config["test_batch_size"])
    else:
        test_loader = None
    
    loader = build_dataset(config["minibatch_size"])
    network = build_network()
    if config["optimizer"] in ["S-Rprop", "AdamUpd", "SGDUpd"]:
        pass
    else:
        config["lr_batch_size"] = "null"
    optimizer = build_optimizer(network, config["optimizer"], config["learning_rate"], config["minibatch_size"], \
                                config["lr_batch_size"], eta_m, eta_p, min_lr, max_lr, track_lr)
    if schedule:
        steps = config["epochs"]
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        wandb.define_metric("scheduler_steps")
        wandb.log({"Global learning rate": scheduler.get_last_lr()[0], "scheduler_steps": 0})
    elif track_lr:
        wandb.define_metric("LR_steps")
        wandb.define_metric("Indiv. learning rate mean", step_metric="LR_steps")
        wandb.define_metric("Indiv. learning rate std", step_metric="LR_steps")
    else:
        pass

    counter_3000 = 0  # Needed to make custom steps in wandb
    if val:
        network.eval()
        test(network, test_loader, counter_3000)
    for epoch in range(1, config["epochs"]+1):
        train_epoch(network, loader, optimizer,  config["minibatch_size"], epoch, \
                    config["lr_batch_size"], track_lr, counter_3000, val, test_loader)
        loader = Custom_Optimizers.Shuffle_MNIST_Data(loader, random_seed)
        counter_3000+=20
        if schedule:
            scheduler.step()
            wandb.log({"Global learning rate": scheduler.get_last_lr()[0], "scheduler_steps": epoch})
        #if val:
        #    network.eval()
        #    test(network, test_loader, counter_3000)   # This validates model after every epoch
        
    wandb.finish()

def test(network, test_loader, counter_3000):
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = network(data)
      test_loss += F.nll_loss(output, target).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= (len(test_loader.dataset)/len(data))
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
  wandb.log({"Validation loss": test_loss, "Val accuracy": 100. * correct / len(test_loader.dataset), \
             "val_steps": counter_3000})

learning_rates = [0.05]
lr_batch_size = 0
minibatch_sizes = [25]

test_batch_size = 5000

for learning_rate in learning_rates:
    for minibatch_size in minibatch_sizes:
            config = {"epochs": 5,
                    "optimizer": "SGD",
                    "learning_rate": learning_rate,
                    "minibatch_size": minibatch_size,
                    "lr_batch_size": lr_batch_size,
                    "test_batch_size": test_batch_size,
                    "method": "grid"}
            for j in range(4):
                ### schedule is only used for Adam and SGD/SGD+M with LR schedulers, also tracks learning rates
                ### track_lr is only used for S-Rprop, Adam-Upd, SGD-Upd
                ### Vanilla SGD+M and Adam have constant learning rates and therefore we do not track them
                ### validation can be used for all of them
                train(config, schedule=False, track_lr=False, val=False)