import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # ============================================
        #               Encoder Architecture
        # ============================================
        self.hidden_dim = 10

        self.conv1 = nn.Conv2d(1, 64, 3, 1, padding=1)

        # Residual block
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.residual_conv = nn.Conv2d(64, 64, 1)

        # MLP
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(3136, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc_mu = nn.Linear(128, self.hidden_dim)
        self.fc_log_var = nn.Linear(128, self.hidden_dim)

        # ============================================
        #               Decoder Architecture
        # ============================================
        self.fc4 = nn.Linear(self.hidden_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

    def encoder(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # Residual connection
        residual = self.residual_conv(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x + residual)

        # MLP
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        # Output mu and log_var
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def decoder(self, x):
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = torch.sigmoid(x)
        return x

    def sampling(self, mu, log_var):
        # Sample a epsilon from the standardized normal distribution keeping the same shape with log_var
        eps = torch.randn_like(log_var)
        # Obtain the standard deviation from variance
        std = torch.exp(log_var / 2)
        # Reparameterization trick
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        x = self.decoder(z)
        return x, mu, log_var


def vae_loss(x, y, mu, log_var):
    # Binary cross entropy loss of reconstraction data and original data
    bce_loss = F.binary_cross_entropy(y, x.view(-1, 784), reduction='sum')
    # KL divergence loss
    kld_loss = 0.5 * torch.sum(log_var.exp() + mu.pow(2) - 1 - log_var)
    return bce_loss + kld_loss


def train(args, model, device, train_loader, optimizer, epoch):
    epoch_train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        recon_data, mu, log_var = model(data)
        loss = vae_loss(data, recon_data, mu, log_var)

        # backpropagation
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / len(data)))
            if args.dry_run:
                break
    return epoch_train_loss / len(train_loader.dataset)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            recon, mu, log_var = model(data)
            test_loss += vae_loss(data, recon, mu, log_var).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    return test_loss


def save_test_image(model, device, epoch):
    with torch.no_grad():
        # Sample a latent vector as the input of the decoder
        z = torch.randn(64, model.hidden_dim).to(device)
        sample = model.decoder(z)
        if not os.path.exists('./samples'):
            os.makedirs('./samples')
        save_image(sample.view(64, 1, 28, 28), './samples/sample_%03d.png' % epoch)


def draw_learning_curve(epoch_train_loss_list, epoch_test_loss_list, epochs):
    plt.figure(figsize=(10, 6))

    # Training loss curve
    plt.plot(range(2, epochs + 1), epoch_train_loss_list[1:], label='Train Loss', marker='o')
    plt.text(2, epoch_train_loss_list[1], f"{epoch_train_loss_list[1]:.2f}", fontsize=10, ha='center', va='bottom')
    plt.text(epochs, epoch_train_loss_list[-1], f"{epoch_train_loss_list[-1]:.2f}", fontsize=10, ha='center', va='bottom')
    # Test loss curve
    plt.plot(range(2, epochs + 1), epoch_test_loss_list[1:], label='Test Loss', marker='s')
    plt.text(2, epoch_test_loss_list[1], f"{epoch_test_loss_list[1]:.2f}", fontsize=10, ha='center', va='bottom')
    plt.text(epochs, epoch_test_loss_list[-1], f"{epoch_test_loss_list[-1]:.2f}", fontsize=10, ha='center', va='bottom')
    plt.xticks(range(2, epochs + 1))
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Learning Curve', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device('cuda')
    elif use_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # Training data
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    # Test data
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    epoch_train_loss_list, epoch_test_loss_list = [], []
    for epoch in range(1, args.epochs + 1):
        epoch_train_loss_list.append(train(args, model, device, train_loader, optimizer, epoch))
        epoch_test_loss_list.append(test(model, device, test_loader))
        save_test_image(model, device, epoch)
        scheduler.step()
    draw_learning_curve(epoch_train_loss_list, epoch_test_loss_list, args.epochs)
    if args.save_model:
        torch.save(model.state_dict(), 'mnist_vae.pt')


if __name__ == '__main__':
    main()
