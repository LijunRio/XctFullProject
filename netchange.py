# 该网络为最终定稿的网络
# 编码部分采用的是resnet-50，并使用了imageNet上的预训练模型作为初始参数
import torch.nn as nn
import torch
from torchsummary import summary
from tensorboardX import SummaryWriter


# resnet-50的残差结构
# 可以参考pytorch官方源码
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


# 生成网络部分的双特征并联结构
# 此部分最终输入输出特征图大小与数目均相同
class NewInception(nn.Module):
    def __init__(self, in_channels, out_channels, middle_channels):
        super(NewInception, self).__init__()

        self.branch1 = nn.Sequential(
            BasicConvTranspose3d(in_channels, middle_channels, kernel_size=1),  # 点卷积降低计算
            NoReluConvTranspose3d(middle_channels, out_channels, kernel_size=3, padding=1)  # 3*3卷积
        )
        self.branch2 = nn.Sequential(
            BasicConvTranspose3d(in_channels, middle_channels, kernel_size=1),  # 点卷积降低计算
            NoReluConvTranspose3d(middle_channels, out_channels, kernel_size=5, padding=2)  # 5*5卷积
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        outputs = branch1 + branch2
        # 将branch1+branch2相加之后再输入激活函数
        outputs = self.relu(outputs)
        return outputs


# 三维反卷积 + BachNormalize
class NoReluConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(NoReluConvTranspose3d, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, **kwargs)
        self.Batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.deconv(x)
        x = self.Batch_norm(x)
        return x


# 三维反卷积 + BachNormalize + ReLu激活函数
class BasicConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConvTranspose3d, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, **kwargs)
        self.Batch_norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)  # inplace=True可以减少计算量

    def forward(self, x):
        x = self.deconv(x)
        x = self.Batch_norm(x)
        x = self.relu(x)
        return x


# 生成网络部分
class Generation_net(nn.Module):
    def __init__(self):
        super(Generation_net, self).__init__()

        # ===========transform===============
        # 将编码器部分得到的特征图转为三维的
        self.trans_layer1 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.trans_layer2 = nn.ConvTranspose3d(2048, 2048, kernel_size=1, stride=1, padding=0, output_padding=0,
                                               bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        # =====net============
        # 1024
        self.deconv1 = BasicConvTranspose3d(2048, 1024, kernel_size=4, stride=2, padding=1)
        self.inception1 = NewInception(1024, 1024, 300)
        # # 512
        self.deconv2 = BasicConvTranspose3d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.inception2 = NewInception(512, 512, 100)
        # # 256
        self.deconv3 = BasicConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1)
        self.inception3 = NewInception(256, 256, 50)
        # 256-128-64-64
        self.deconv4 = BasicConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        # self.deconv5 = BasicConvTranspose3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.inception4 = NewInception(128, 128, 50)
        self.deconv6 = BasicConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        # self.deconv7 = BasicConvTranspose3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.inception6 = NewInception(64, 64, 30)

        # # 64-1-128
        self.conv8 = nn.Conv3d(64, 1, kernel_size=1, stride=1, padding=0)
        self.relu9 = nn.ReLU(inplace=True)
        self.cov10 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # transform
        x = self.trans_layer1(x)
        x = self.relu1(x)
        # 将4096*4*4 转化为 1024*2*4*4
        # 这样就可以进行三维的反卷积
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
        x = self.inception4(x)
        # x = self.deconv5(x)
        x = self.deconv6(x)
        # x = self.deconv7(x)
        #
        x = self.conv8(x)
        x = self.relu9(x)

        # 需要压缩成三维
        x = torch.squeeze(x, 1)
        x = self.cov10(x)

        return x


# 定义整个重建网络
class newReconNet(nn.Module):
    def __init__(self):
        super(newReconNet, self).__init__()
        # resnet-50网络部分================================================
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
        self.layer3 = self._make_layer(Bottleneck, 256, 2, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 1, stride=2)
        # self.conv5 = nn.Conv2d(2048, 4096, kernel_size=1, stride=1, bias=False)
        # self.relu5 = nn.ReLU(inplace=True)
        # # 生成网络部分====================================================
        # self.Generion_layer = Generation_net()

        # 使用kaiming_normal进行权重初始化=================================
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    #
    def _make_layer(self, block, channel, block_num, stride=1, flag=4):
        downsample = None
        block.expansion = flag
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
        #
        # x = self.conv5(x)
        # x = self.relu5(x)
        #
        # x = self.Generion_layer(x)
        return x


# 返回新建的网络
def new_net():
    return newReconNet()


# 训练时需要注释这部分
# 用summary在终端打印网络参数
model = new_net()
model.cuda()
summary(model, (3, 128, 128))
# # 用tensorboard可视化网络结构
# # dummy_input = torch.rand(1, 3, 128, 128) #假设输入13张1*28*28的图片
# # with SummaryWriter(comment='Nenet') as w:
# #     w.add_graph(model, (dummy_input, ))
