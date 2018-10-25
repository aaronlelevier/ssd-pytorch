import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from ssdmultibox.datasets import TrainPascalDataset


class CustomHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2d_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2d_drop = nn.Dropout2d()
        self.linear = nn.Linear(512, 784)
        self.linear_cats = nn.Linear(320, 16*20)
        self.linear_bbs = nn.Linear(320, 16*4)

    def forward(self, outputs):
        """
        Reeturns a list of: [torch.Size([4, 64]), torch.Size([4, 320])]

        TODO: labels will have to be flattened. Currently they're [16, 4] and [16, 20]
        """
        out3 = self.linear(outputs)
        out4 = torch.reshape(out3, (4, 28, 28))
        out7 = self.conv2d_1(out4.unsqueeze(1)) # [4, 10, 24, 24]
        out7_mp = F.relu(F.max_pool2d(out7, kernel_size=2)) # [4, 10, 12, 12]
        out8_mp = F.relu(F.max_pool2d(self.conv2d_2(out7_mp), kernel_size=2)) # [4, 20, 4, 4]
        flat = out8_mp.view(4, 320) # [4, 320]

        cats_pred = self.linear_cats(flat)
        bbs_pred = self.linear_bbs(flat)
        return [bbs_pred, cats_pred]
    

def train():
    # model
    model = models.resnet18(pretrained=True)
    model.fc = CustomHead()

    # data
    dataset = TrainPascalDataset(grid_size=4)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # process
    item = next(iter(dataloader))
    image_ids, ims, bbs, cats = item
    outputs = model(ims)

    return outputs
