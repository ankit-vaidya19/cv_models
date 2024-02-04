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


residual_architecture = [
    [64, 64, 256],
    [128, 128, 512],
    [256, 256, 1024],
    [512, 512, 2048],
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters[0],
            kernel_size=(1, 1),
            stride=(2, 2),
            padding=0,
        )
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(
            in_channels=filters[0],
            out_channels=filters[1],
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.conv3 = nn.Conv2d(
            in_channels=filters[1],
            out_channels=filters[2],
            kernel_size=(1, 1),
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.skip_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters[2],
            kernel_size=(1, 1),
            stride=(2, 2),
            padding=0,
        )
        self.skip_bn = nn.BatchNorm2d(filters[2])
        self.relu = nn.ReLU()

    def forward(self, x):
        x_skip = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x_skip = self.skip_conv(x_skip)
        x_skip = self.skip_bn(x_skip)
        x += x_skip
        x = self.relu(x)
        return x


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(IdentityBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters[0],
            kernel_size=(1, 1),
            padding=0,
        )
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(
            in_channels=filters[0],
            out_channels=filters[1],
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.conv3 = nn.Conv2d(
            in_channels=filters[1],
            out_channels=filters[2],
            kernel_size=(1, 1),
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.relu = nn.ReLU()

    def forward(self, x):
        x_skip = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += x_skip
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
        )
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.residual1 = self._make_layer(
            num_identity=2, in_channels=64, architecture=residual_architecture[0]
        )
        self.residual2 = self._make_layer(
            num_identity=3, in_channels=256, architecture=residual_architecture[1]
        )
        self.residual3 = self._make_layer(
            num_identity=5, in_channels=512, architecture=residual_architecture[2]
        )
        self.residual4 = self._make_layer(
            num_identity=2, in_channels=1024, architecture=residual_architecture[3]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1 * 1 * 2048, 10)

    def _make_layer(self, num_identity, in_channels, architecture):
        layers = []
        layers.append(ConvBlock(in_channels=in_channels, filters=architecture))
        in_channels_local = architecture[2]
        for i in range(num_identity):
            layers.append(
                IdentityBlock(in_channels=in_channels_local, filters=architecture)
            )
            in_channels_local = architecture[2]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

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
                img = batch[0].to(DEVICE)
                labels = batch[1].to(DEVICE)
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
                img = batch[0].to(DEVICE)
                labels = batch[1].to(DEVICE)
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


net = ResNet()
net.to(DEVICE)
train_loader, test_loader = create_loaders()
net.fit(train_loader=train_loader, test_loader=test_loader)
