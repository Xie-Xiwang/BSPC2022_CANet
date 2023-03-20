import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.functional import softmax
from functools import partial



nonlinearity = partial(F.relu, inplace=True)

# # L模块
class MCU(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=True, stride=1):  # 默认use_1x1conv=True
        super(MCU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=9, padding=4, stride=stride)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        # print('X',X.shape) # torch.Size([1, 3, 512, 512])
        # 未使用BN和Relu
        # Y = self.conv2(self.conv1(self.conv(X)))

        Y = F.relu(self.bn(self.conv(X)))
        Y = F.relu(self.bn1(self.conv1(Y)))
        Y = self.bn2(self.conv2(Y))
        # print('Y2',Y.shape)  # torch.Size([1, 32, 512, 512])
        if self.conv3:
            X = self.conv3(X)
            # print('X1*1',X.shape)

        return F.relu(Y+X)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):

        return self.conv(input)

class ThreeConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ThreeConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):

        return self.conv(input)


class RDB(nn.Module):
    def __init__(self, in_ch, out_ch,use_1x1conv=True,stride=1):
        super(RDB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(out_ch,),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm2d(out_ch),
        )
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, input):
        Y=self.conv(input)
        if self.conv3:
            X = self.conv3(input)
        return F.relu(Y+X)



class LUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(LUNet, self).__init__()

        self.conv1 = RDB(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.M1 = MCU(3, 32)
        self.conv2 = RDB(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.M2 = MCU(3, 64)
        self.conv3 = RDB(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.M3 = MCU(3, 128)
        self.conv4 = RDB(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.M4 = MCU(3, 256)
        self.conv5 = RDB(256, 512)

        # center
        self.dblock = CRPblock(512)

        # decoder
        self.conv6 = ThreeConv(512, 256)
        self.conv7 = ThreeConv(256, 128)
        self.conv8 = ThreeConv(128, 64)
        self.conv9 = ThreeConv(64, 32)

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)


        self.conv10 = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        L1 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        p1=self.M1(L1)+p1
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        L2 = F.interpolate(L1, scale_factor=0.5, mode='bilinear')
        p2 = self.M2(L2)+p2
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        L3 = F.interpolate(L2, scale_factor=0.5, mode='bilinear')
        p3 = self.M3(L3)+p3
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        L4 = F.interpolate(L3, scale_factor=0.5, mode='bilinear')
        p4 = self.M4(L4)+p4
        c5 = self.conv5(p4)

        # center
        c5 = self.dblock(c5)



        # decoding + concat path
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        # print('---',up_9.shape)
        merge9 = torch.cat([up_9, c1], dim=1)
        # print('---merge9', merge9.shape)
        c9 = self.conv9(merge9)
        # print('---c9', c9.shape)

        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)

        return out

class RDB_2(nn.Module):
    def __init__(self, in_ch, out_ch,use_1x1conv=True,stride=1):
        super(RDB_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(out_ch,),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(out_ch),
        )
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, input):
        Y=self.conv(input)
        if self.conv3:
            X = self.conv3(input)
        return F.relu(Y+X)



class CRPblock(nn.Module):
    def __init__(self, feats, block_count=3):
        super().__init__()

        self.block_count = block_count
        self.relu = nn.ReLU(inplace=False)
        for i in range(0, block_count):
            self.add_module(
                "block{}".format(i),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                    nn.Conv2d(
                        feats,
                        feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(0, self.block_count):
            path = self.__getattr__("block{}".format(i))(path)
            x = x + path

        return x







