import numpy as np

import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import ConcreteVAE


def main(args):

    def train(epoch):
        model.train()
        temp = 1.0
        train_loss = 0
        for i, (data, _) in enumerate(train_loader, 1):
            data = data.to(device)  # shape (batch, 1, 28, 28)
            x = data
            x = x.view(x.size(0), -1)
            if i % args.temp_interval == 1:
                n_updates = epoch * len(train_loader) + i
                temp = max(np.exp(-n_updates*args.temp_anneal), args.min_temp)

            x_probs, alpha = model(x, temp=temp)

            bce, kl = model.loss(x, x_probs, alpha)
            elbo = -bce - kl
            loss = -elbo
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if i % args.log_interval == 0:
                print('| Epoch: {} [{}/{} ({:.0f}%)] | loss: {:.6f} | temp {:.3f}'.format(
                    epoch, i * len(data), len(train_loader.dataset),
                    100. * i / len(train_loader),
                    loss.item() / len(data),
                    temp))

            if i == 10:
                save_image(x[0].view(28, 28),
                    f'results/{epoch}_data_x.png')
                save_image(x_probs[0].view(28, 28),
                    f'results/{epoch}_data_x_.png')
                save_image(alpha,
                    f'results/{epoch}_data_alpha.png')
                sample = alpha > 0.5
                sample = torch.distributions.OneHotCategorical(logits=alpha).sample()
                save_image(sample,
                    f'results/{epoch}_data_sample.png')


        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))


    def test(epoch):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)  # shape (batch, 1, 28, 28)
                x = data
                x = x.view(x.size(0), -1)
                x_probs, alpha = model(x)

                bce, kl = model.loss(x, x_probs, alpha)
                elbo = -bce - kl
                loss = -elbo
                test_loss += loss
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                          x_probs.view(args.batch_size, 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                             'results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    torch.manual_seed(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data, train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    model = ConcreteVAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            # sample = torch.distributions.Bernoulli(
                # torch.ones(64, args.latent_dim) / 2).sample()
            sample = torch.distributions.OneHotCategorical(
                torch.ones(64, args.latent_dim) / 2).sample()
            onehot = torch.zeros(args.latent_dim, args.latent_dim)
            locs = torch.arange(args.latent_dim)
            onehot[locs, locs] = 1
            sample_decoded = model.decode(sample).cpu()
            onehot_decoded = model.decode(onehot).cpu()
            save_image(sample,
                'results/latent_' + str(epoch) + '.png')
            save_image(onehot,
                'results/onehot_' + str(epoch) + '.png')
            save_image(sample_decoded.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
            save_image(onehot_decoded.view(args.latent_dim, 1, 28, 28),
                       'results/onehot_decoded_' + str(epoch) + '.png')
