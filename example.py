import Custom_Optimizers
import torch
import torchvision
import torch.nn.functional as F
import random

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

def validate(network, test_loader, device):
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

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(random_seed)

    eta_m, eta_p = 0.5, 1.2   
    min_lr, max_lr = 1e-10, 1

    mb_size = 600
    lrb_size=12000
    learning_rate=0.005
    test_batch_size = 10000
    epochs = 2

    loader = build_dataset(mb_size)
    test_loader = build_test_dataset(test_batch_size)

    network = Custom_Optimizers.CNN()
    network.to(device)

    optimizer = Custom_Optimizers.SRPROP(network.parameters(), M=mb_size, L=lrb_size, lr=learning_rate, etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), track_lr=True)

    for epoch in range(1, epochs+1):
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
                if mb_size<3000:
                    loss_3000 /= (3000/len(data))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx+1) * len(data),
                len(loader.dataset), 100. * batch_idx / len(loader), loss_3000.item()))
                loss_3000 = 0
        network.eval()
        validate(network, test_loader, device)
  
    lr_mean = optimizer.lr_mean
    lr_std = optimizer.lr_mean
    lr_steps = optimizer.lr_counter

if __name__ == '__main__':
    main()