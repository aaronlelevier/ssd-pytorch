import time

from torch import optim
from torch.utils.data import DataLoader

from ssdmultibox.criterion import SSDLoss
from ssdmultibox.datasets import TrainPascalDataset, device
from ssdmultibox.models import SSDModel

EPOCHS = 1
BATCH = 4
NUM_WORKERS = 0


def main():
    # data
    dataset = TrainPascalDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH, num_workers=NUM_WORKERS)

    # model
    model = SSDModel()
    model = model.to(device)
    criterion = SSDLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    current_time = time.time()

    for epoch in range(EPOCHS):
        for i, (image_ids, ims, gt_bbs, gt_cats) in enumerate(dataloader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            preds = model(ims)
            loss = criterion(preds, (gt_bbs, gt_cats))

            # backward pass
            loss.backward()
            optimizer.step()

            # stats
            print(i, 'loss:', loss, 'time:', time.time() - current_time)
            current_time = time.time()

            # test run for 5 steps
            if i == 4:
                break
        break


if __name__ == '__main__':
    main()
