from torch import optim
from torch.utils.data import DataLoader

from ssdmultibox.criterion import SSDLoss
from ssdmultibox.datasets import TrainPascalDataset
from ssdmultibox.models import SSDModel, vgg16_bn

EPOCHS = 1
BATCH = 4
NUM_WORKERS = 0


def main():
    # data
    dataset = TrainPascalDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH, num_workers=NUM_WORKERS)

    # model
    vgg_base = vgg16_bn(pretrained=True)

    # freeze base network
    for layer in vgg_base.parameters():
        layer.requires_grad = False

    model = SSDModel(vgg_base)

    criterion = SSDLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(EPOCHS):
        for i, (image_ids, ims, gt_bbs, gt_cats) in enumerate(dataloader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            preds = model(ims)
            loss = criterion(gt_bbs, gt_cats, preds)

            # backward pass
            loss.backward()
            optimizer.step()

            # stats
            print(i, 'loss:', loss)

            # test run for 5 steps  
            if i == 4:
                break
        break


if __name__ == '__main__':
    main()
