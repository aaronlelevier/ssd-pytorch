import time

from torch import nn, optim
from torch.utils.data import DataLoader

from ssdmultibox.criterion import SSDLoss
from ssdmultibox.datasets import TrainPascalDataset, device
from ssdmultibox.models import SSDModel

EPOCHS = 1
BATCH = 8
NUM_WORKERS = 0 # MAC: sysctl -n hw.ncpu
LR = 0.01
CLIP = 0.5


def main():
    # data
    dataset = TrainPascalDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH, num_workers=NUM_WORKERS)

    # model
    model = SSDModel()
    model = model.to(device)
    criterion = SSDLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)
    current_time = time.time()

    for epoch in range(EPOCHS):
        for i, (image_ids, ims, gt_bbs, gt_cats) in enumerate(dataloader):
            # put data on device
            ims, gt_bbs, gt_cats = dataset.to_device(ims, gt_bbs, gt_cats)

            # forward pass
            preds = model(ims)
            loss = criterion(preds, (gt_bbs, gt_cats))

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()

            # stats
            print(i, 'loss:', loss, 'time:', time.time() - current_time)
            current_time = time.time()

            break
        break


if __name__ == '__main__':
    main()
