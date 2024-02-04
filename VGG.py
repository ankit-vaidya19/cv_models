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

VGG_variations = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_variations["VGG11"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fcs = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        self.push_to_device(self.device)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

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


net = VGG()
train_loader, test_loader = create_loaders()
net.fit(train_loader=train_loader, test_loader=test_loader)
