import torch
import warnings
import numpy as np
import torch.nn as nn
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

torch.random.manual_seed(42)
warnings.simplefilter("ignore")

parser = ArgumentParser()

parser.add_argument("-dsp", "--dataset_path", type=str, help="Dataset Path")
parser.add_argument(
    "-d", "--download_dataset", type=bool, help="Download Dataset", default=False
)
parser.add_argument("-bs", "--batch_size", type=int, help="Batch Size", default=128)
parser.add_argument("-e", "--epochs", type=int, help="Number Epochs", default=10)
parser.add_argument(
    "-lr", "--learning_rate", type=float, help="Learning Rate", default=0.01
)
parser.add_argument(
    "-wd", "--weight_decay", type=float, help="Weight Decay", default=0.0005
)

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


class Variational_Auto_Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.enc_fc1 = nn.Linear(input_dim, 200)
        self.enc_mu = nn.Linear(200, 20)
        self.enc_sigma = nn.Linear(200, 20)
        self.dec_fc1 = nn.Linear(20, 200)
        self.dec_fc2 = nn.Linear(200, input_dim)
        self.relu = nn.ReLU()

    def encoder(self, x):
        x = self.relu(self.enc_fc1(x))
        mu, sigma = self.enc_mu(x), self.enc_sigma(x)
        return mu, sigma

    def decoder(self, z):
        z = self.relu(self.dec_fc1(z))
        z = torch.sigmoid(self.dec_fc2(z))
        return z

    def reparam_trick(self, mu, sigma):
        epsilon = torch.randn_like(sigma)
        z_reparam = mu + sigma * epsilon
        return z_reparam

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z_reparam = self.reparam_trick(mu, sigma)
        recon_img = self.decoder(z_reparam)
        return recon_img, mu, sigma

    def fit(self, train_loader, test_loader):
        optim = torch.optim.Adam(
            params=self.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        criterion = nn.MSELoss()
        self.train()

        for epoch in range(args.epochs):
            train_loss = []
            print(f"Epoch - {epoch+1}/{args.epochs}")
            for batch in tqdm(train_loader):
                x = batch[0].to(DEVICE)
                x = x.reshape(x.shape[0], -1)
                recons, mu, sigma = self(x)
                recons_loss = criterion(recons, x)
                kl_div = -torch.sum(
                    1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)
                )
                loss = recons_loss + kl_div
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss.append(loss.detach().cpu().numpy())
            loss = sum(train_loss) / len(train_loss)
            print(f"\tTrain\tLoss - {round(loss, 3)}")

        self.eval()
        with torch.no_grad():
            test_loss = []
            for batch in tqdm(test_loader):
                x = batch[0].to(DEVICE)
                x = x.reshape(x.shape[0], -1)
                recons, mu, sigma = self(x)
                recons_loss = criterion(recons, x)
                kl_div = -torch.sum(
                    1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)
                )
                loss = recons_loss + kl_div
                test_loss.append(loss.detach().cpu().numpy())
            loss = sum(test_loss) / len(test_loss)
            print(f"\tTest\tLoss - {round(loss, 3)}")


def create_loaders():
    train_data = datasets.CIFAR10(
        root=args.dataset_path,
        transform=transforms.Compose(
            [
                transforms.Resize((70, 70)),
                transforms.RandomCrop((64, 64)),
                transforms.ToTensor(),
            ]
        ),
        train=True,
        download=args.download_dataset,
    )
    test_data = datasets.CIFAR10(
        root=args.dataset_path,
        transform=transforms.Compose(
            [
                transforms.Resize((70, 70)),
                transforms.RandomCrop((64, 64)),
                transforms.ToTensor(),
            ]
        ),
        train=False,
        download=args.download_dataset,
    )
    train_loader = DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_data, batch_size=128, shuffle=False)
    return train_loader, test_loader


vae = Variational_Auto_Encoder(3 * 64 * 64).to(DEVICE)
train_loader, test_loader = create_loaders()
vae.fit(train_loader=train_loader, test_loader=test_loader)
