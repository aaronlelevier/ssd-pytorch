import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models

from ssdmultibox.datasets import TrainPascalDataset


class LinearCats(nn.Module):
    """
    Takes a conv of [4, 20, 4, 4] and returns a vector of 320
    """
    def __init__(self):
        super().__init__()
        self.conv2d_3 = nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.conv2d_4 = nn.Conv2d(40, 80, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv2d_3(x)) # [4, 40, 4, 4]
        x = self.conv2d_4(x) # [4, 80, 4, 4]
        x = F.relu(F.max_pool2d(x, 2)) # [4, 80, 2, 2]
        return x.view(-1, 320) # [4, 320]


class LinearBbs(nn.Module):
    """
    Takes a conv of [4, 20, 4, 4] and returns a vector of 64
    """
    def __init__(self):
        super().__init__()
        self.conv2d_3 = nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.linear128 = nn.Linear(160, 128)
        self.linear64 = nn.Linear(128, 64)

    def forward(self, x):
        out9 = self.conv2d_3(x) # [4, 40, 4, 4]
        out9_mp = F.relu(F.max_pool2d(out9, kernel_size=2)) # [4, 40, 2, 2]
        flat = out9_mp.view(4, -1) # [4, 160]
        flat128 = F.relu(self.linear128(flat))
        flat64 = F.relu(self.linear64(flat128))
        return flat64


class CustomHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2d_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2d_3 = nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.linear = nn.Linear(512, 784)
        self.linear_cats = LinearCats()
        self.linear_bbs = LinearBbs()

    def forward(self, outputs):
        out3 = self.linear(outputs)
        out4 = torch.reshape(out3, (4, 28, 28))
        out5 = out4.unsqueeze(1) # [4, 1, 28, 28]
        out7 = self.conv2d_1(out5) # [4, 10, 24, 24]
        out7_mp = F.relu(F.max_pool2d(out7, kernel_size=2)) # [4, 10, 12, 12]
        out8 = self.conv2d_2(out7_mp) # [4, 20, 8, 8]
        out8_mp = F.relu(F.max_pool2d(out8, kernel_size=2)) # [4, 20, 4, 4]
        return self.linear_bbs(out8_mp), self.linear_cats(out8_mp)
    

def train():
    # model
    model = models.resnet18(pretrained=True)
    model.fc = CustomHead()

    # data
    dataset = TrainPascalDataset(grid_size=4)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # process
    item = next(iter(dataloader))
    image_ids, ims, bbs, cats = item

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward pass
    pred_bbs, pred_cats = model(ims)

    # category loss
    cats_bce = F.binary_cross_entropy_with_logits(pred_cats, cats)

    return outputs
