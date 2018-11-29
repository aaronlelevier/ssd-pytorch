import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torchvision.models import VGG
from torchvision.models.vgg import cfg, model_urls

from ssdmultibox.datasets import NUM_CLASSES


class OutConv(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.bbs_size = 4
        self.oconv1 = nn.Conv2d(nin, self.bbs_size, 3, padding=1)
        self.oconv2 = nn.Conv2d(nin, NUM_CLASSES, 3, padding=1)

    def forward(self, x):
        return [self.flatten_conv(self.oconv1(x), self.bbs_size),
                self.flatten_conv(self.oconv2(x), NUM_CLASSES)]

    def flatten_conv(self, x, size):
        bs,nf,gx,gy = x.size()
        x = x.permute(0,2,3,1).contiguous()
        return x.view(bs,-1, size)


class OutCustomHead(nn.Module):
    """
    Returns the final stacked predictions as a tuple (bbs, cats)
    """
    def __init__(self):
        super().__init__()
        self.aspect_ratio_count = 6
        self._setup_outconvs()

    def _setup_outconvs(self):
        block_names = ['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
        block_sizes = [512, 1024, 512, 256, 256, 256]
        for i, name in enumerate(block_names):
            for j, ar in enumerate(range(self.aspect_ratio_count)):
                setattr(self, f'{block_names[i]}_{j}', OutConv(block_sizes[i]))

    def forward(self, blocks):
        all_bbs = None
        all_cats = None
        first_loop = True
        for k,v in blocks.items():
            for i in range(self.aspect_ratio_count):
                # NOTE: maybe shouldn't be accessing the `_modules` private OrderedDict here ...
                fm_ar_outconv = self._modules[f'{k}_{i}']
                bbs, cats = fm_ar_outconv(v)
                # just need to check bbs or cats here
                if first_loop:
                    all_bbs = bbs
                    all_cats = cats
                    first_loop = False
                else:
                    all_bbs = torch.cat((all_bbs, bbs), dim=1)
                    all_cats = torch.cat((all_cats, cats), dim=1)
        return all_bbs, all_cats


class BlocksCustomHead(nn.Module):
    "Returns the blocks for the OutCustomHead"
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


class SSDModel(nn.Module):
    def __init__(self):
        # features are really layers
        super().__init__()
        self.vgg_base = self._create_vgg_base()
        self.blocks_model = BlocksCustomHead()
        self.out_conv_head = OutCustomHead()

    def forward(self, x):
        x = self.vgg_base(x)
        x = self.blocks_model(x)
        return self.out_conv_head(x)

    def _create_vgg_base(self):
        vgg_base = vgg16_bn(pretrained=True)
        for layer in vgg_base.parameters():
            layer.requires_grad = False
        return vgg_base


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            # ceil_mode=False in the normal function
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGBase(VGG):
    def __init__(self, features, num_classes=1000, init_weights=True):
        # features are really layers
        super().__init__(features, num_classes, init_weights)

    def forward(self, x):
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
        blocks = {b:None for b in feat_layers}

        for i, feature in enumerate(self.features):
            x = feature(x)
            if i == 32:
                blocks['block4'] = x
            # don't call the final max_pool layer b/c we want an output shape of (512, 19, 19)
            if i == 42:
                break
        return x, blocks


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGGBase(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model
