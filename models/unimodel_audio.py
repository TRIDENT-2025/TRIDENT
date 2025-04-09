import torch
import torch.nn as nn
import torch.nn.functional as F
from .aux_models import GlobalPooling2D
from torchvision import models as tmodels

class Audio_Net(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=8, n_classes=2):
        super(Audio_Net, self).__init__()

        # Convert Conv2d to Conv1d and adjust kernel sizes and paddings
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(int(hidden_channels))
        self.gp1 = GlobalPooling2D()

        self.conv2 = nn.Conv2d(hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(2 * hidden_channels))
        self.gp2 = GlobalPooling2D()

        self.conv3 = nn.Conv2d(2 * hidden_channels, 3 * hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(3 * hidden_channels))
        self.gp3 = GlobalPooling2D()

        self.conv4 = nn.Conv2d(3 * hidden_channels, 4 * hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(int(4 * hidden_channels))
        self.gp4 = GlobalPooling2D()

        self.classifier = nn.Sequential(
            nn.Linear(int(4 * hidden_channels), n_classes-1),
            nn.Sigmoid()
        )

        # Initialization of weights
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out1, 2)  # Use max_pool1d for 1D data
        gp1 = self.gp1(out)

        out2 = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out2, 2)
        gp2 = self.gp2(out2)

        out3 = F.relu(self.bn3(self.conv3(out)))
        out = F.max_pool2d(out3, 2)
        gp3 = self.gp3(out3)

        out4 = F.relu(self.bn4(self.conv4(out)))
        out = F.max_pool2d(out4, 2)
        gp4 = self.gp4(out4)

        out = self.classifier(gp4)

        return gp1, gp2, gp3, gp4, out.squeeze()


class GP_VGG(nn.Module):
    def __init__(self, n_classes=2):
        super(GP_VGG, self).__init__()

        vgg = list(tmodels.vgg19(pretrained=False).features)
        vgg[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg = nn.ModuleList(vgg)
        self.gp1 = GlobalPooling2D()
        self.gp2 = GlobalPooling2D()
        self.gp3 = GlobalPooling2D()
        self.gp4 = GlobalPooling2D()

        self.bn4 = nn.BatchNorm1d(512)  # only used for classifier

        self.classifier = nn.Sequential(
            nn.Linear(512, n_classes-1),
            nn.Sigmoid()
        )

    def forward(self, x):

        for i_l, layer in enumerate(self.vgg):

            x = layer(x)

            if i_l == 20:
                out_1 = x

            if i_l == 26:
                out_2 = x

            if i_l == 33:
                out_3 = x

            if i_l == 36:
                out_4 = x
                tmp_4 = self.gp4(x)
                bn_4 = self.bn4(tmp_4)

        out = self.classifier(bn_4)

        return out_1, out_2, out_3, out_4, out.squeeze()


if __name__ == '__main__':

    model = Audio_Net()
    model = GP_VGG()
    model.eval()
    print(model(torch.rand(1, 1, 40, 40))[-1])