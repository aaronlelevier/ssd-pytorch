import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torchvision.models import VGG
from torchvision.models.vgg import model_urls

from ssdmultibox.bboxer import TensorBboxer
from ssdmultibox.config import cfg


class SSDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_base = self._create_vgg_base()
        self.blocks_layer = BlocksLayer()
        self.loc_head = LocHead()
        self.conf_head = ConfHead()

    def forward(self, x):
        x = self.vgg_base(x)
        x = self.blocks_layer(x)
        return self.loc_head(x), self.conf_head(x)

    def _create_vgg_base(self):
        vgg_base = vgg16_bn(pretrained=True)
        for layer in vgg_base.parameters():
            layer.requires_grad = False

        # remove inherited classifier - this is required to load the VGG
        # init_weights, but we don't want it in our model
        delattr(vgg_base, 'classifier')

        return vgg_base

    def unfreeze(self):
        "sets all model layers to trainable"
        for layer in self.parameters():
            layer.requires_grad = True


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGGBase(make_layers(), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


class VGGBase(VGG):
    def __init__(self, features, num_classes=1000, init_weights=True):
        # features are really layers
        super().__init__(features, num_classes, init_weights)

    def forward(self, x):
        feat_layers = ['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
        blocks = {b:None for b in feat_layers}

        for i, feature in enumerate(self.features):
            x = feature(x)
            if i == 32:
                blocks['block4'] = x
            # don't call the final max_pool layer b/c we want an output shape of (512, 19, 19)
            if i == 42:
                break
        return x, blocks


def make_layers():
    layer_config = [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    layers = []
    in_channels = 3
    for v in layer_config:
        if v == 'M':
            # ceil_mode=False in the normal function
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class BlocksLayer(nn.Module):
    "Returns the blocks for the LocHead and ConfHead"
    def __init__(self):
        super().__init__()
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv8 = nn.Conv2d(1024, 256, kernel_size=1, padding=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2)
        self.conv9 = nn.Conv2d(512, 128, kernel_size=1, padding=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv10 = nn.Conv2d(256, 128, kernel_size=1, padding=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv11 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)

    def forward(self, x_and_blocks):
        x, blocks = x_and_blocks
        # block6
        out6 = F.dropout2d(F.relu(self.conv6(x)))
        # block7
        out7 = F.dropout2d(F.relu(self.conv7(out6)))
        blocks['block7'] = out7
        # block8
        out8 = F.relu(self.conv8(out7))
        out8_2 = F.relu(self.conv8_2(out8))
        blocks['block8'] = out8_2
        # block9
        out9 = F.relu(self.conv9(out8_2))
        out9_2 = F.relu(self.conv9_2(out9))
        blocks['block9'] = out9_2
        # block10
        out10 = F.relu(self.conv10(out9_2))
        out10_2 = F.relu(self.conv10_2(out10))
        blocks['block10'] = out10_2
        # block11
        out11 = F.relu(self.conv11(out10_2))
        out11_2 = F.relu(self.conv11_2(out11))
        blocks['block11'] = out11_2
        return blocks


class BaseHead(nn.Module):
    def __init__(self, n):
        """
        Returns final output predictions for either bbs or cats

        Args:
            n (int): should be 4 or cfg.NUM_CLASSES. This is the 2nd dim output size
        """
        super().__init__()
        self.n = n
        self.block_names = ['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
        block_sizes = [512, 1024, 512, 256, 256, 256]
        for name, size in zip(self.block_names, block_sizes):
            setattr(self, name, nn.Conv2d(size, self.n*cfg.ASPECT_RATIOS, kernel_size=3, padding=1))

    def forward(self, preds):
        all_loc = []
        for name in self.block_names:
            all_loc.append(
                getattr(self, name)(preds[name]).permute(0, 2, 3, 1).contiguous())

        return torch.cat([x.view(x.size(0), -1, self.n) for x in all_loc], dim=1)


class ConfHead(BaseHead):
    def __init__(self, n=cfg.NUM_CLASSES):
        super().__init__(n)


class LocHead(BaseHead):
    def __init__(self, n=4):
        super().__init__(n)

        self.anchor_boxes = TensorBboxer.get_stacked_anchor_boxes()
        self.fm_max_offsets = TensorBboxer.get_feature_map_max_offsets()

    def forward(self, preds):
        bbs = super().forward(preds)

        return torch.clamp(
            self.anchor_boxes + (torch.tanh(bbs).to(cfg.DEVICE) * self.fm_max_offsets),
        min=0, max=1)
