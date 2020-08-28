# 该网络为最终定稿的网络
# 编码部分采用的是resnet-50，并使用了imageNet上的预训练模型作为初始参数
import torch.nn as nn
import torch
from torchsummary import summary
from tensorboardX import SummaryWriter
import numpy as np


# resnet-50的残差结构
# 可以参考pytorch官方源码
class Bottleneck(nn.Module):  # 对于resnet-50以上的使用Bottleneck
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


# resnet34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # 使用BN后不需要使用偏置
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 输出加上捷径后才加上激活函数
        out += identity
        out = self.relu(out)

        return out


# 定义整个重建网络
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 四种残差块的结构
        self.layer1 = self._make_layer(BasicBlock, 64, 3)  # resnet-34
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
        # self.layer1 = self._make_layer(Bottleneck, 64, 3)    # resnet-50
        # self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        # self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        # self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # 使用kaiming_normal进行权重初始化=================================
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1, ):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:  #
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = [block(self.in_channel, channel, downsample=downsample, stride=stride)]
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class BasicTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicTranspose3d, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, **kwargs)
        self.Batch_norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)  # inplace=True可以减少计算量

    def forward(self, x):
        x = self.deconv(x)
        x = self.Batch_norm(x)
        x = self.relu(x)
        return x


# 三维反卷积 + BachNormalize
class NoReluTrans3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(NoReluTrans3d, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, **kwargs)
        self.Batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.deconv(x)
        x = self.Batch_norm(x)
        return x


class NewInception(nn.Module):
    def __init__(self, in_channels, channelreduce, ch3out, ch5out):
        super(NewInception, self).__init__()
        # 点卷积降低计算
        self.reduce = BasicTranspose3d(in_channels, channelreduce, kernel_size=1)

        self.branch1 = nn.Sequential(
            BasicTranspose3d(channelreduce, ch3out, kernel_size=3, padding=1)  # 3*3卷积
        )
        self.branch2 = nn.Sequential(
            BasicTranspose3d(channelreduce, ch5out, kernel_size=5, padding=2)  # 5*5卷积
        )

    def forward(self, x):
        x = self.reduce(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        outputs = [branch1, branch2]

        return torch.cat(outputs, 1)


class XctNet(nn.Module):
    def __init__(self):
        super(XctNet, self).__init__()
        self.resnet = ResNet()

        self.trans_layer = nn.ConvTranspose3d(128, 256, kernel_size=1, stride=1, padding=0, output_padding=0,
                                              bias=False)
        self.relu = nn.ReLU(inplace=True)
        # 256*4*4*4
        self.decov1 = BasicTranspose3d(256, 256, kernel_size=4, stride=2, padding=1)
        self.Incep1 = NewInception(256, 80, 156, 100)
        self.decov2 = BasicTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.Incep2 = NewInception(128, 50, 68, 60)
        self.decov3 = BasicTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.Incep3 = NewInception(64, 20, 34, 30)
        self.decov4 = BasicTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.Incep4 = NewInception(32, 10, 18, 14)
        self.decov5 = BasicTranspose3d(32, 16, kernel_size=4, stride=2, padding=1)

        self.conv3d = nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0)
        self.cov2d = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 128, 4, 4, 4)
        x = self.trans_layer(x)
        x = self.relu(x)
        x = self.decov1(x)
        x = self.Incep1(x)
        x = self.decov2(x)
        x = self.Incep2(x)
        x = self.decov3(x)
        x = self.Incep3(x)
        x = self.decov4(x)
        x = self.Incep4(x)
        x = self.decov5(x)

        x = self.conv3d(x)
        x = self.relu(x)
        x = torch.squeeze(x, 1)
        x = self.cov2d(x)

        return x

# 训练时需要注释这部分
# 用summary在终端打印网络参数
model = XctNet()
model.cuda()
summary(model, (3, 128, 128))
# import os
# from torchviz import make_dot
# x = torch.rand(1, 3, 128, 128).requires_grad_(True)
# model = XctNet()
# y = model(x)
# g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
# g.render('XCTNET', view=False)
#
# dummy_input = torch.rand(1, 3, 128, 128) #假设输入13张1*28*28的图片
# with SummaryWriter(comment='Nenet') as w:
#     w.add_graph(model, (dummy_input, ))