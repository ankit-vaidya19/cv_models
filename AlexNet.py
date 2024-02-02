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


class AlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(AlexNet, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.convnet = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.init_bias()
        self.push_to_device(self.device)

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc_layers(x)
        return x

    def init_bias(self):
        for layer in self.convnet:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.convnet[3].bias, 0)
        nn.init.constant_(self.convnet[8].bias, 0)
        nn.init.constant_(self.convnet[10].bias, 0)

    def push_to_device(self, device):
        self.to(device)

    def accuracy(self, true, pred):
        true = np.array(true)
        pred = np.array(pred)
        acc = np.sum((true == pred).astype(np.float32)) / len(true)
        return acc * 100

    def fit(self, train_loader, test_loader):
        optim = torch.optim.SGD(
            params=self.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        self.train()
        for epoch in range(args.epochs):
            train_loss = []
            train_preds = []
            train_labels = []
            print(f"Epoch - {epoch+1}/{args.epochs}")
            for batch in tqdm(train_loader):
                img = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                scores = self(img)
                loss = criterion(scores, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss.append(loss.detach().cpu().numpy())
                train_preds.append(scores.argmax(dim=-1))
                train_labels.append(batch[1])
            loss = sum(train_loss) / len(train_loss)
            acc = self.accuracy(
                torch.concat(train_labels, dim=0).cpu(),
                torch.concat(train_preds, dim=0).cpu(),
            )
            print(
                f"\tTrain\tLoss - {round(loss, 3)}", "\t", f"Accuracy - {round(acc, 3)}"
            )

        self.eval()
        with torch.no_grad():
            test_loss = []
            test_preds = []
            test_labels = []
            for batch in tqdm(test_loader):
                img = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                scores = self(img)
                loss = criterion(scores, labels)
                train_loss.append(loss.detach().numpy().cpu())
                train_preds.append(scores.argmax(dim=-1))
                train_labels.append(batch[1])
            loss = sum(test_loss) / len(test_loss)
            acc = self.accuracy(
                torch.concat(test_labels, dim=0).cpu(),
                torch.concat(test_preds, dim=0).cpu(),
            )
            print(
                f"\tTest\tLoss - {round(loss, 3)}", "\t", f"Accuracy - {round(acc, 3)}"
            )


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


net = AlexNet()
train_loader, test_loader = create_loaders()
net.fit(train_loader=train_loader, test_loader=test_loader)
