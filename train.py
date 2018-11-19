import enum
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ssdmultibox.criterion import SSDLoss
from ssdmultibox.datasets import PascalDataset, TrainPascalDataset, ValPascalDataset, device
from ssdmultibox.models import SSDModel

EPOCHS = 1
BATCH = 8
NUM_WORKERS = 0 # MAC: sysctl -n hw.ncpu
LR = 0.01
CLIP = 0.5


class Phase(enum.Enum):
    TRAIN = 'TRAIN'
    VAL = 'VAL'


if 'losses' not in locals(): losses = {Phase.TRAIN: [], Phase.VAL: []}


def get_dataloaders():
    train_dataset = TrainPascalDataset()
    val_dataset = ValPascalDataset()
    return {
        Phase.TRAIN: DataLoader(
            train_dataset, batch_size=BATCH, num_workers=NUM_WORKERS, shuffle=True),
        Phase.VAL: DataLoader(
            val_dataset, batch_size=BATCH, num_workers=NUM_WORKERS)
    }


def get_model():
    return SSDModel().to(device)


def train(model, dataloaders, epochs):
    criterion = SSDLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    current_time = time.time()

    for epoch in range(epochs):
        scheduler.step()
        for phase in [Phase.TRAIN, Phase.VAL]:
            dataloader = dataloaders[phase]
            if phase == Phase.TRAIN:
                model.train()
            else:
                model.eval()

            for i, (image_ids, ims, gt_bbs, gt_cats) in enumerate(dataloader):
                # put data on device
                ims, gt_bbs, gt_cats = PascalDataset.to_device(ims, gt_bbs, gt_cats)

                # zero out gradients
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == Phase.TRAIN):
                    preds = model(ims)
                    loss = criterion(preds, (gt_bbs, gt_cats))

                    # backward pass
                    if phase == Phase.TRAIN:
                        loss.backward()
                        optimizer.step()

                    # stats
                    if i % 5 == 0:
                        print(i, 'loss:', loss.item(), 'time:', time.time() - current_time)
                        current_time = time.time()
                        losses[phase].append(loss.item())


def main():
    model = get_model()
    dataloaders = get_dataloaders()
    train(model, dataloaders, epochs=1)


if __name__ == '__main__':
    main()
