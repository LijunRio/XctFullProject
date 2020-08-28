import torch.nn as nn
import torch
from torchsummary import summary
from tensorboardX import SummaryWriter


class Bottleneck(nn.Module):
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


class InceptionGen(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionGen, self).__init__()

        self.branch1 = BasicConvTranspose3d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConvTranspose3d(in_channels, ch3x3red, kernel_size=1),
            BasicConvTranspose3d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConvTranspose3d(in_channels, ch5x5red, kernel_size=1),
            BasicConvTranspose3d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            BasicConvTranspose3d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionGen2(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionGen2, self).__init__()

        self.branch1 = NoReluConvTranspose3d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConvTranspose3d(in_channels, ch3x3red, kernel_size=1),
            NoReluConvTranspose3d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConvTranspose3d(in_channels, ch5x5red, kernel_size=1),
            NoReluConvTranspose3d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            NoReluConvTranspose3d(in_channels, pool_proj, kernel_size=1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identiy = x
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        outputs = torch.cat(outputs, 1)
        outputs = outputs + identiy
        outputs = self.relu(outputs)

        return outputs


class BasicConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConvTranspose3d, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, **kwargs)
        self.Batch_norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.Batch_norm(x)
        x = self.relu(x)
        return x


class Generation_net(nn.Module):
    def __init__(self):
        super(Generation_net, self).__init__()

        # ===========transform===============
        self.trans_layer1 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.trans_layer2 = nn.ConvTranspose3d(2048, 2048, kernel_size=1, stride=1, padding=0, output_padding=0,
                                               bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        # =====net============
        # 1024
        self.deconv1 = BasicConvTranspose3d(2048, 1024, kernel_size=4, stride=2, padding=1)
        self.inception1 = InceptionGen2(1024, 384, 192, 384, 48, 128, 128)
        # # 512
        self.deconv2 = BasicConvTranspose3d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.inception2 = InceptionGen2(512, 128, 128, 256, 24, 64, 64)
        # # 256
        self.deconv3 = BasicConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1)
        self.inception3 = InceptionGen2(256, 64, 96, 128, 16, 32, 32)
        # # 256-128-64-64
        self.deconv4 = BasicConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv5 = BasicConvTranspose3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv6 = BasicConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv7 = BasicConvTranspose3d(64, 64, kernel_size=3, stride=1, padding=1)
        # # 64-1-128
        self.conv8 = nn.Conv3d(64, 1, kernel_size=1, stride=1, padding=0)
        self.relu9 = nn.ReLU(inplace=True)
        self.cov10 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # transform
        x = self.trans_layer1(x)
        x = self.relu1(x)
        x = x.view(-1, 2048, 2, 4, 4)
        x = self.trans_layer2(x)
        x = self.relu2(x)

        # net
        x = self.deconv1(x)
        x = self.inception1(x)
        x = self.deconv2(x)
        x = self.inception2(x)
        x = self.deconv3(x)
        x = self.inception3(x)
        #
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        #
        x = self.conv8(x)
        x = self.relu9(x)

        # 需要压缩成三维
        x = torch.squeeze(x, 1)
        x = self.cov10(x)

        return x


class newReconNet(nn.Module):
    def __init__(self):
        super(newReconNet, self).__init__()
        self.in_channel = 64
        # 第一层卷积层
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # 最大池化下采样操作
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 四种残差块的结构
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.conv5 = nn.Conv2d(2048, 4096, kernel_size=1, stride=1, bias=False)
        self.relu5 = nn.ReLU(inplace=True)
        self.Generion_layer = Generation_net()
        # self.Generion_layer = NewGeneration()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

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

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.Generion_layer(x)
        return x


class NoReluConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(NoReluConvTranspose3d, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, **kwargs)
        self.Batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.deconv(x)
        x = self.Batch_norm(x)
        return x


class IncepResModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IncepResModule, self).__init__()

        self.branch2 = nn.Sequential(
            BasicConvTranspose3d(in_channels, out_channels, kernel_size=1),
            # 保证输出大小等于输入大小
            BasicConvTranspose3d(in_channels, out_channels, kernel_size=3, padding=1),
            NoReluConvTranspose3d(in_channels, out_channels, kernel_size=1)
        )

        self.branch3 = nn.Sequential(
            BasicConvTranspose3d(in_channels, out_channels, kernel_size=1),
            # 保证输出大小等于输入大小
            BasicConvTranspose3d(in_channels, out_channels, kernel_size=5, padding=2),
            NoReluConvTranspose3d(in_channels, out_channels, kernel_size=1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1 = x  # 本身
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        out = branch1 + branch2 + branch3
        out = self.relu(out)
        return out


class NewGeneration(nn.Module):
    def __init__(self):
        super(NewGeneration, self).__init__()

        # ===========transform===============
        self.trans_layer1 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.trans_layer2 = nn.ConvTranspose3d(2048, 2048, kernel_size=1, stride=1, padding=0, output_padding=0,
                                               bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        # =====net============
        # 1024
        self.deconv1 = BasicConvTranspose3d(2048, 1024, kernel_size=4, stride=2, padding=1)
        self.inception1 = IncepResModule(1024, 1024)
        # # 512
        self.deconv2 = BasicConvTranspose3d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.inception2 = IncepResModule(512, 512)
        # # 256
        self.deconv3 = BasicConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1)
        self.inception3 = IncepResModule(256, 256)
        # # 256-128-64-64
        self.deconv4 = BasicConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv5 = BasicConvTranspose3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv6 = BasicConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv7 = BasicConvTranspose3d(64, 64, kernel_size=3, stride=1, padding=1)
        # # 64-1-128
        self.conv8 = nn.Conv3d(64, 1, kernel_size=1, stride=1, padding=0)
        self.relu9 = nn.ReLU(inplace=True)
        self.cov10 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # transform
        x = self.trans_layer1(x)
        x = self.relu1(x)
        x = x.view(-1, 2048, 2, 4, 4)
        x = self.trans_layer2(x)
        x = self.relu2(x)

        # net
        x = self.deconv1(x)
        x = self.inception1(x)
        x = self.deconv2(x)
        x = self.inception2(x)
        x = self.deconv3(x)
        x = self.inception3(x)
        #
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        #
        x = self.conv8(x)
        x = self.relu9(x)

        # 需要压缩成三维
        x = torch.squeeze(x, 1)
        x = self.cov10(x)

        return x


def new_net():
    return newReconNet()


def Newgenrate():
    return NewGeneration()

model = new_net()
model.cuda()
# dummy_input = torch.rand(1, 3, 128, 128) #假设输入13张1*28*28的图片
# with SummaryWriter(comment='Nenet') as w:
#     w.add_graph(model, (dummy_input, ))
summary(model, (3, 128, 128))
