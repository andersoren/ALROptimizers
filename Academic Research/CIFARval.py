import sys
sys.path.append("..")
import Custom_Optimizers
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

def build_val_dataset(batch_size_val):
    val_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(root='./data',train=False, download=False,
                                            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean, std)])),
                                            batch_size=batch_size_val, shuffle=False)
    return val_loader      

def build_network(arch, Dropout):
    if arch=="ResNet":
        network = Custom_Optimizers.ResNet9(in_channels=3, num_classes=100, drop_rate=Dropout) # Default Dropout is 0.2
    elif arch=="DenseNet":
        network = Custom_Optimizers.DenseNet(growth_rate=12, block_config=(12,12,12), num_init_features=16, bn_size=1, \
                                          drop_rate=Dropout, num_classes=100)
    else:
        print("Architecture not recognised")

    if torch.cuda.device_count() > 1:
        network = torch.nn.DataParallel(network)  # Parallelise data with two GPUs
        print("Using", torch.cuda.device_count(), "GPUs.")
    
    return network.to(device)

def build_optimizer(network, config, eta_m, eta_p, min_lr, max_lr, track_lr):
    if config["optimizer"] == "S-Rprop":
        optimizer = Custom_Optimizers.SRPROP(network.parameters(), M=config["minibatch_size"], L=config["lr_batch_size"], \
                                             lr=config["learning_rate"], etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), \
                                              weight_decay=config["weight_decay"], track_lr=track_lr)
    elif config["optimizer"] == "SGDUpd2":
        optimizer = Custom_Optimizers.SGDUPD(network.parameters(), M=config["minibatch_size"], L=config["lr_batch_size"], \
                                             lr=config["learning_rate"], etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), \
                                              momentum=config["momentum"], weight_decay=config["weight_decay"], track_lr=track_lr)
    elif config["optimizer"] == "AdamUpd":
        optimizer = Custom_Optimizers.ADAMUPD(network.parameters(), M=config["minibatch_size"], L=config["lr_batch_size"], \
                                              lr=config["learning_rate"], etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), \
                                                weight_decay=config["weight_decay"], track_lr=track_lr)
    elif config["optimizer"] == "Adam":
        optimizer = optim.Adam(network.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "SGD+M":
        optimizer = optim.SGD(network.parameters(), lr=config["learning_rate"], momentum=config["momentum"], \
                              weight_decay=config["weight_decay"], dampening=0)
    else:
        print("Optimizer type not recognised")
    return optimizer

def train_epoch(network, loader, optimizer, epoch, lr_batch_size, track_lr, schedule, scheduler, grad_clip=False):
    network.train()
    training_loss = 0
    correct=0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        loss.backward()
        if grad_clip: 
            torch.nn.utils.clip_grad_value_(network.parameters(), 0.1)
        training_loss += loss.item()
        optimizer.step()
        if schedule == "OneCycleLR":
            scheduler.step()  
            wandb.log({"Global learning rate": scheduler.get_last_lr()[0], "scheduler_steps": int(len(loader)*(epoch-1)+(batch_idx+1))})
        if (batch_idx+1)%100==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx+1) * len(data),
                  len(loader.dataset), 100. * batch_idx / len(loader), loss.item()))
    training_loss /= (len(loader.dataset)/len(data))
    wandb.log({"Training loss": training_loss, "Training accuracy": 100. * correct / len(loader.dataset), \
             "epoch_steps": epoch})
    if track_lr:
        if ((batch_idx+1)*len(data))%lr_batch_size==0:
            wandb.log({"Indiv. learning rate mean": optimizer.lr_mean[-1], \
                    "Indiv. learning rate std": optimizer.lr_std[-1], "LR_steps": optimizer.lr_counter})
        

# Default is that learning rates are tracked inside updated algorithms, and no scheduler used nor validation set
def train(config=None, track_lr=False, val=False):
    # Initialize a new wandb run
    run = wandb.init(project="CIFAR-100",
                    config=config)
    wandb.define_metric("steps_3000")
    wandb.define_metric("epoch_steps")
    wandb.define_metric("Loss over latest 3000 patterns", step_metric="steps_3000")
    wandb.define_metric("Training loss per epoch", step_metric="epoch_steps")

    if val:
        wandb.define_metric("val_steps")
        wandb.define_metric("Val loss", step_metric="val_steps")
        wandb.define_metric("Val accuracy", step_metric="val_steps")
        val_loader = build_val_dataset(config["val_batch_size"])
    else:
        val_loader = None
    
    network = build_network(config["architecture"], config["Dropout"])
    optimizer = build_optimizer(network, config, eta_m, eta_p, min_lr, max_lr, track_lr)
    
    if config["schedule"] == "None":
        scheduler = config["schedule"]
    else:
        # steps = config["epochs"]
        # Cosine lr scheduling
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        # Multi-step lr scheduling from Huang et al 2018
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(config["epochs"]*0.5), int(config["epochs"]*0.75)], gamma=0.1)
        #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, config["learning_rate"], epochs=config["epochs"],  \
        #                                        steps_per_epoch=int(50000/config["minibatch_size"]))
        wandb.define_metric("scheduler_steps")
        wandb.log({"Global learning rate": scheduler.get_last_lr()[0], "scheduler_steps": 0})

    if track_lr:
        wandb.define_metric("LR_steps")
        wandb.define_metric("Indiv. learning rate mean", step_metric="LR_steps")
        wandb.define_metric("Indiv. learning rate std", step_metric="LR_steps")
    else:
        pass

    if val:
        network.eval()
        validate(network, val_loader, 0)
    for epoch in range(1, config["epochs"]+1):
        loader = build_dataset(config["minibatch_size"])
        train_epoch(network, loader, optimizer, epoch, \
                    config["lr_batch_size"], track_lr, config["schedule"], scheduler, grad_clip=config["grad_clip"])
        if config["schedule"] in ["MultiStep", "CosineAnn"]:
            scheduler.step()  
            wandb.log({"Global learning rate": scheduler.get_last_lr()[0], "scheduler_steps": epoch})
        if val:
            network.eval()
            validate(network, val_loader, epoch)   # This validates model after every epoch
    wandb.finish()

def validate(network, val_loader, epoch):
  val_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in val_loader:
      data, target = data.to(device), target.to(device)
      output = network(data)
      val_loss += F.cross_entropy(output, target).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  val_loss /= (len(val_loader.dataset)/len(data))
  print('Val set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset), 100. * correct / len(val_loader.dataset)))
  wandb.log({"Validation loss": val_loss, "Val accuracy": 100. * correct / len(val_loader.dataset), \
             "val_steps": epoch})

###### -----------------------------------------------------------------------
######                   For making custom runs including validating

learning_rates = [0.001]
minibatch_sizes = [500]
lr_batch_size = 25000
optimizer = "AdamUpd"
#SGDUpd2 is without momentum
val_batch_size = 5000

for learning_rate in learning_rates:
    for minibatch_size in minibatch_sizes:
        config = {"epochs": 50,
                "optimizer": optimizer,
                "learning_rate": learning_rate,
                "minibatch_size": minibatch_size,
                "lr_batch_size": lr_batch_size,
                "val_batch_size": val_batch_size,
                "architecture": "ResNet",
                "schedule": "None",    # Must be one of: ["MultiStep", "CosineAnn", "OneCycleLR", "None"]
                "grad_clip": False,
                "weight_decay": 0.0,
                "eta_m": eta_m,
                "min_lr": min_lr,
                "Dropout": 0.0,
                "momentum": 0.0,
                "Validating best runs": True}

        for j in range(2):
            ### if a scheduler other than None is used, learning rates automatically tracked
            ### track_lr is only used for S-Rprop, Adam-Upd, SGD-Upd (must make =True/False manually)
            ### Vanilla SGD+M and Adam have constant learning rates and therefore we do not track them
            ### validation can be used for all of them
            train(config, track_lr=True, val=True)
