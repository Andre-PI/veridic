import torch
import torch.nn as nn
from torchvision import models


def make_layers(cfg, in_channels=3, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    frontend_cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
    backend_cfg  = [512, 512, 512, 256, 128, 64]

    def __init__(self):
        super().__init__()
        self.frontend = make_layers(self.frontend_cfg)
        self.backend  = make_layers(self.backend_cfg, in_channels=512, dilation=True)
        self.output   = nn.Conv2d(64, 1, kernel_size=1)
        self._init_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output(x)

    def _init_weights(self):
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        vgg_layers = [l for l in vgg.features[:23] if isinstance(l, nn.Conv2d)]
        own_layers = [l for l in self.frontend.modules() if isinstance(l, nn.Conv2d)]
        for own, vgg_l in zip(own_layers, vgg_layers):
            own.weight.data = vgg_l.weight.data
            own.bias.data   = vgg_l.bias.data
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.constant_(self.output.bias, 0)
