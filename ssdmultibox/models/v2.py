from torch import nn

# these 2 functions return lists of layers. Combined, they return
# all 6 layers needed as "source" layers for the custom CNN head

def get_base_layers():
    """
    Returns a list of 41 base layers for VGG

    The 32nd layer will be stored and passed into the custom head,
    along with the 5 layers from `get_extras`
    """
    layer_config = [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    layers = []
    in_channels = 3
    for i, v in enumerate(layer_config):
        if v == 'M':
            # don't want max pool on final layer
            if i < 17:
                # ceil_mode=False in the normal function
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return layers



def get_extra_layers():
    """
    Returns a list of 10 extra layers, of which we'll end up storing
    every odd layer, so 5 and pass these into the custom head
    """
    return [
        nn.Conv2d(512, 1024, kernel_size=3, padding=1),
        nn.Conv2d(1024, 1024, kernel_size=1),
        nn.Conv2d(1024, 256, kernel_size=1, padding=1),
        nn.Conv2d(256, 512, kernel_size=3, stride=2),
        nn.Conv2d(512, 128, kernel_size=1, padding=1),
        nn.Conv2d(128, 256, kernel_size=3, stride=2),
        nn.Conv2d(256, 128, kernel_size=1, padding=1),
        nn.Conv2d(128, 256, kernel_size=3, stride=2),
        nn.Conv2d(256, 128, kernel_size=1),
        nn.Conv2d(128, 256, kernel_size=3)
    ]
