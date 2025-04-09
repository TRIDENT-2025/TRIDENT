import torch
import torch.nn as nn
import torch.nn.functional as F
from .inflated_resnet import inflated_resnet

class Visual_ResNet(nn.Module):
    def __init__(self, name, m=3, n_classes=2): #For RF m=3, and for Visual m=30
        super(Visual_ResNet, self).__init__()
        self.cnn = inflated_resnet(name)
        self.avgpool_Tx7x7 = nn.AvgPool3d((m, 4, 4))
        if m==30: 
            self.D = 2048
        else: 
            self.D = 2048
        self.classifier = nn.Sequential(
            nn.Linear(self.D, n_classes-1),
            nn.Sigmoid()
        )

    def temporal_pooling(self, x):
        B, D, T, W, H = x.size()
        if self.D == D:
            final_representation = self.avgpool_Tx7x7(x)
            final_representation = final_representation.view(B, self.D)
            return final_representation
        else:
            print("Temporal pooling is not possible due to invalid channels dimensions:", self.D, D)

    def forward(self, x):
        # Changing temporal and channel dim to fit the inflated resnet input requirements
        B, T, C, W, H = x.size()
        x = x.view(B, 1, T, W, H, C)
        x = x.transpose(1, -1)
        x = x.view(B, C, T, W, H)
        x = x.contiguous()

        # Inflated ResNet
        out_1, out_2, out_3, out_4 = self.cnn.get_feature_maps(x)

        # Temporal pooling
        out_5 = self.temporal_pooling(out_4)

        out = self.classifier(out_5)

        return out_1, out_2, out_3, out_4, out_5, out.squeeze()

class GlobalPooling2D(nn.Module):
    def __init__(self):
        super(GlobalPooling2D, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        x = x.view(x.size(0), -1)
        return x

class ConvBNReLU3D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Visual_MobileNet(nn.Module):
    def __init__(self, n_classes=2):
        super(Visual_MobileNet, self).__init__()
        self.conv1 = ConvBNReLU3D(3, 32, stride=2)
        self.conv_blocks = nn.Sequential(
            ConvBNReLU3D(32, 64, stride=2),
            ConvBNReLU3D(64, 128, stride=2),
            ConvBNReLU3D(128, 128),
            ConvBNReLU3D(128, 256, stride=2),
            ConvBNReLU3D(256, 256),
            ConvBNReLU3D(256, 512, stride=2),
            *[ConvBNReLU3D(512, 512) for _ in range(5)],  # Repeated blocks
            ConvBNReLU3D(512, 1024, stride=2),
            ConvBNReLU3D(1024, 1024)
        )
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, n_classes-1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4) 
        x = self.conv1(x)
        out_1 = self.conv_blocks[0](x)
        out_2 = self.conv_blocks[1](out_1)
        out_3 = self.conv_blocks[2](out_2)
        out_4 = self.conv_blocks[3](out_3)
        out_5 = self.conv_blocks[4](out_4)
        x = self.avg_pool(out_5)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out_1, out_2, out_3, out_4, out_5, out.squeeze()



if __name__ == '__main__':

    model = Visual_ResNet()
    model = Visual_MobileNet()
    print(model(torch.rand(6, 30, 3, 112, 112))[-1])
