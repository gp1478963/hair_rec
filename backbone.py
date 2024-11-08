from PIL.ImageOps import scale
from torch import nn

class VGGConvNet(nn.Module):
    def __init__(self,in_channels, out_channels, conv_count, max_pool=True):
        super(VGGConvNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_count = conv_count
        self.max_pool = nn.MaxPool2d(kernel_size=2) if max_pool else nn.Identity()
        self.conv_group = nn.ModuleList()
        for i in range(conv_count):
            self.conv_group.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1))
            self.conv_group.append(nn.ReLU(inplace=True))
            in_channels = out_channels

    def forward(self, x):
        x = self.conv_group(x)
        x = self.max_pool(x)
        return x

class VGGNet(nn.Module):
    def __init__(self,input_size = 224, layer=None, hide_channels=None, n_class = 1000):
        super(VGGNet, self).__init__()
        self.module_layer = nn.ModuleList()
        self.input_size = input_size
        self.layer = [2,2,3,3,3] if layer is None else layer
        self.hide_channels = [3, 64, 128, 256, 512, 512] if hide_channels is None else hide_channels
        self.scale = 1
        for i in range(len(layer)):
            self.module_layer.append(VGGConvNet(in_channels=hide_channels[i],out_channels=hide_channels[i+1], conv_count=layer[i]))
            self.scale  = self.scale * 2

        self.fc1_input_size = self.input_size//self.scale
        self.fc1 = nn.Linear(in_features=hide_channels[-1]*self.fc1_input_size, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=n_class)

    def forward(self, x):
        x = self.module_layer(x)
        x = x.Flatten()
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class VGG16(VGGNet):
    def __init__(self, n_class = 1000):
        self.n_class = n_class
        self.layer = [2,2,3,3,3]
        self.hide_channels = [3, 64, 128, 256, 512, 512]
        super(VGG16, self).__init__(layer=[2,2,3,3,3], hide_channels=self.hide_channels, n_class=n_class)

    def forward(self, x):
        return super(VGG16, self).forward(x)

class VGG19(VGGNet):
    def __init__(self, n_class = 1000):
        self.n_class = n_class
        self.layer = [2,2,4,4,4]
        self.hide_channels = [3, 64, 128, 256, 512, 512]
        super(VGG19, self).__init__(layer=[2, 2, 3, 3, 3], hide_channels=self.hide_channels, n_class=n_class)

    def forward(self, x):
        return super(VGG19, self).forward(x)