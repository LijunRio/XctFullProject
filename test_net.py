import torch.nn as nn
import torch
from torchsummary import summary


# 2D Conv
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=1, stride=stride, padding=0,
                     bias=False)


def conv2x2(in_planes, out_planes, stride=2):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=2, stride=stride, padding=0,
                     bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3, stride=stride, padding=1,
                     bias=False)


def conv4x4(in_planes, out_planes, stride=2):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=4, stride=stride, padding=1,
                     bias=False)


# 3D Conv
def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=1, stride=stride, padding=0,
                     bias=False)


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=3, stride=stride, padding=1,
                     bias=False)


def conv4x4x4(in_planes, out_planes, stride=2):
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=4, stride=stride, padding=1,
                     bias=False)


# 2D Deconv
def deconv1x1(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=1, stride=stride, padding=0, output_padding=0,
                              bias=False)


def deconv2x2(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=2, stride=stride, padding=0, output_padding=0,
                              bias=False)


def deconv3x3(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=3, stride=stride, padding=1, output_padding=0,
                              bias=False)


def deconv4x4(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=4, stride=stride, padding=1, output_padding=0,
                              bias=False)


# 3D Deconv
def deconv1x1x1(in_planes, out_planes, stride):
    return nn.ConvTranspose3d(in_planes, out_planes,
                              kernel_size=1, stride=stride, padding=0, output_padding=0,
                              bias=False)


def deconv3x3x3(in_planes, out_planes, stride):
    return nn.ConvTranspose3d(in_planes, out_planes,
                              kernel_size=3, stride=stride, padding=1, output_padding=0,
                              bias=False)


def deconv4x4x4(in_planes, out_planes, stride):
    return nn.ConvTranspose3d(in_planes, out_planes,
                              kernel_size=4, stride=stride, padding=1, output_padding=0,
                              bias=False)


def _make_layers(in_channels, output_channels, type, batch_norm=False, activation=None):
    layers = []

    # conv
    if type == 'conv1_s1':
        layers.append(conv1x1(in_channels, output_channels, stride=1))
    elif type == 'conv2_s2':
        layers.append(conv2x2(in_channels, output_channels, stride=2))
    elif type == 'conv3_s1':
        layers.append(conv3x3(in_channels, output_channels, stride=1))
    elif type == 'conv4_s2':
        # def conv4x4(in_planes, out_planes, stride=2):
        # 	return nn.Conv2d(in_planes, out_planes,
        # 					 kernel_size=4, stride=stride, padding=1,
        # 					 bias=False)
        layers.append(conv4x4(in_channels, output_channels, stride=2))
    elif type == 'deconv1_s1':
        layers.append(deconv1x1(in_channels, output_channels, stride=1))
    elif type == 'deconv2_s2':
        layers.append(deconv2x2(in_channels, output_channels, stride=2))
    elif type == 'deconv3_s1':
        layers.append(deconv3x3(in_channels, output_channels, stride=1))
    elif type == 'deconv4_s2':
        layers.append(deconv4x4(in_channels, output_channels, stride=2))
    elif type == 'conv1x1_s1':
        layers.append(conv1x1x1(in_channels, output_channels, stride=1))
    elif type == 'deconv1x1_s1':
        layers.append(deconv1x1x1(in_channels, output_channels, stride=1))
    elif type == 'deconv3x3_s1':
        layers.append(deconv3x3x3(in_channels, output_channels, stride=1))
    elif type == 'deconv4x4_s2':
        layers.append(deconv4x4x4(in_channels, output_channels, stride=2))
    else:
        raise NotImplementedError('layer type [{}] is not implemented'.format(type))

    # batch_normalize
    if batch_norm == '2d':
        layers.append(nn.BatchNorm2d(output_channels))
    elif batch_norm == '3d':
        layers.append(nn.BatchNorm3d(output_channels))

    # activation
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'sigm':
        layers.append(nn.Sigmoid())
    elif activation == 'leakyrelu':
        layers.append(nn.LeakyReLU(0.2, True))
    else:
        if activation is not None:
            raise NotImplementedError('activation function [{}] is not implemented'.format(activation))

    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1  # 残差结构卷积核的个数是否变化

    # 初始化：输入特征矩阵深度，输出特征矩阵深度， stride默认去1， 下采样默认为none

    # 下采样对应虚线的残差结构，每一层残差结构的第一层有降维的作用
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 定义第一层卷积层，第一层卷积层有两种情况
        # 步长为1 的时候对应的是实线残差结构，输出特征的高和宽和输入特征的高和宽相同
        # 步长为2 的时候输出特征矩阵的高和宽等于输入特征矩阵的高和宽除2得到的结果向下取整
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # 使用BN后不需要使用偏置
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        # 定义第二层卷积层， 无论实现还是虚线残差结构第二层残差结构都是等于1 的
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # x 为shortcut的值
        if self.downsample is not None:  # 如果没有输入下采样函数的话，对应的就是实现的残差结构
            identity = self.downsample(x)  # 如果传入的下采样函数不等于none的话

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 输出加上捷径后才加上激活函数
        out += identity
        out = self.relu(out)

        return out


# 定义50层以上的resnet
# 50层以上每个残差快中卷积核变化了，变化刚好是4倍
class Bottleneck(nn.Module):
    expansion = 4  # 刚好四倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        # 第二层卷积层有两种定义，实线和虚线不一样
        # 实线的的步距是等于1 的，虚线是等于2的，是根据传入参数进行调整
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        # 第三层卷积层卷积核个数变了
        # 卷积核的个数等于第一层的4倍，所以乘上4
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


class ResNet(nn.Module):

    # 初始化：block所对应的残差结构，会根据层结构传入不同的block
    # blocknumber对应的是每个块的数目
    # inlude_top为了方便以后在resnet的基础上搭建更复杂的网络
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):

        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # 这里的深度对应的是通过第二层maxpooling之后所得到的特征矩阵的深度，所有网络都是一样的深度
        # 第一层卷积层
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # 最大池化下采样操作
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 四种残差块的结构
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        # resnet50改写[3, 4, 6, 3]
        if blocks_num == [3, 4, 6, 3]:
            print("restnet50")
            self.conv5 = nn.Conv2d(2048, 4096, kernel_size=1, stride=1, bias=False)
            self.relu5 = nn.ReLU(inplace=True)

            # self.trans_layer1 = nn.Conv2d(in_channels=4096, out_channels=4096, stride=1, padding=0, bias=False)
            # self.trans1_relu = nn.ReLU(inplace=True)
            # self.trans_layer2 = nn.ConvTranspose2d(2048, 2048, 1, 1)
            ######### transform module
            self.trans_layer1 = _make_layers(4096, 4096, 'conv1_s1', False, 'relu')
            self.trans_layer2 = _make_layers(2048, 2048, 'deconv1x1_s1', False, 'relu')

            ######### generation network - deconvolution layers
            self.deconv_layer10 = _make_layers(2048, 1024, 'deconv4x4_s2', '3d', 'relu')
            self.deconv_layer8 = _make_layers(1024, 512, 'deconv4x4_s2', '3d', 'relu')
            self.deconv_layer7 = _make_layers(512, 512, 'deconv3x3_s1', '3d', 'relu')
            self.deconv_layer6 = _make_layers(512, 256, 'deconv4x4_s2', '3d', 'relu')
            self.deconv_layer5 = _make_layers(256, 256, 'deconv3x3_s1', '3d', 'relu')
            self.deconv_layer4 = _make_layers(256, 128, 'deconv4x4_s2', '3d', 'relu')
            self.deconv_layer3 = _make_layers(128, 128, 'deconv3x3_s1', '3d', 'relu')
            self.deconv_layer2 = _make_layers(128, 64, 'deconv4x4_s2', '3d', 'relu')
            self.deconv_layer1 = _make_layers(64, 64, 'deconv3x3_s1', '3d', 'relu')
            self.deconv_layer0 = _make_layers(64, 1, 'conv1x1_s1', False, 'relu')
            self.output_layer = _make_layers(64, 128, 'conv1_s1', False)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # 残差块结构定义
    # block就是所选用的block
    # channel为残差结构中卷积层所使用的卷积核个数，18和34的残差结构卷积核的数目一样的，但是50层以上是不一样的
    # block_num 多少个残差结构，对于34层而言，conv2_x的残差结构一共对应三个
    def _make_layer(self, block, channel, block_num, stride=1, flag=4):
        downsample = None
        block.expansion = flag
        # 对于50层以上的残差结构
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
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

        # 新增的

        x = self.conv5(x)
        x = self.relu5(x)
        ### transform module
        features = self.trans_layer1(x)
        # 相当于numpy中的resize
        # -1表示我们不想自己计算，电脑帮我们计算出对应的数字
        trans_features = features.view(-1, 2048, 2, 4, 4)
        trans_features = self.trans_layer2(trans_features)

        ### generation network
        deconv10 = self.deconv_layer10(trans_features)
        deconv8 = self.deconv_layer8(deconv10)
        deconv7 = self.deconv_layer7(deconv8)
        deconv6 = self.deconv_layer6(deconv7)
        deconv5 = self.deconv_layer5(deconv6)
        deconv4 = self.deconv_layer4(deconv5)
        deconv3 = self.deconv_layer3(deconv4)
        deconv2 = self.deconv_layer2(deconv3)
        deconv1 = self.deconv_layer1(deconv2)

        ### output
        out = self.deconv_layer0(deconv1)
        out = torch.squeeze(out, 1)
        out = self.output_layer(out)
        x = out
        # x = self.layer5(x)

        if self.include_top:
            x = self.layer5(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet18(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


#
#
def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


model = resnet50(include_top=False)
model.cuda()
summary(model, (3, 128, 128))
# model_weight_path = "./resnet50-19c8e357.pth"
