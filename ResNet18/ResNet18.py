import torch
import torch.nn as nn
import numpy

#定义残差网络基本模块
class Basicblock(nn.Module):
    #在Res18和Res34中残差块卷积层的卷积核个数相同;
    #如果相同的话expansion = 1，表示倍数关系;
    expansion = 1
    #参数说明：in_channel  输入通道数(例如RGB图像有3通道)
    #参数说明：out_channel 输出通道数
    #参数说明：stride 步长
    #参数说明：downsample 残差通道是否是虚线下采样
    def __init__(self, in_channel, out_channel, stride = 1, downsample = None):
        super(Basicblock, self).__init__()  
        #Res18第一个卷积层
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                                out_channels=out_channel,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                bias=False
        )
        #第一个卷积层经过bn层
        self.bn1 = nn.BatchNorm2d(out_channel)
        #激活函数
        self.relu = nn.ReLU()
        #Res18第二个卷积层
        self.conv2 = nn.Conv2d(in_channels=out_channel, 
                                out_channels=out_channel,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False        
        )
        #第二个卷积层经过bn层
        self.bn2 = nn.BatchNorm2d(out_channel)
        #残差通道的直连/下采样情况
        self.downsample = downsample
    
    #正向传播定义
    #参数说明: x特征矩阵
    def forward(self, x):
        #先将输入赋值给残差通道，实现数据的直连
        identity = x
        #判断残差直连种类，如果是不是None，意味着残差通道存在下采样
        if(self.downsample is not None):  
            identity = self.downsample(x)
        #进行网络结构的搭建
        #第一层卷积相关操作
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #第二层卷积的相关操作
        out = self.conv2(out)
        out = self.bn2(out)
        #残差通道和卷积通道相加
        out += identity
        #相加完毕后再进入激活函数
        out = self.relu(out)

        #残差块计算完毕,返回残差块值
        return out 


#定义Res18基本模块
class ResNet18(nn.Module):
    #参数说明：block为残差基本模块
    #参数说明：block_num表示使用残差块的数目，格式为列表形式
    #参数说明：num_classes表示分类数据集种类的个数
    #参数说明：include_top方便网络后续改进，比如成为某一网络的一部分
    def __init__(self, block, blocks_num, num_classes = 1000, include_top=True):
        super(ResNet18,self).__init__()
        self.include_top = include_top
        #输入通道数
        self.in_channel = 64
        #网络输入的第一层为独立卷积层，不属于残差结构，需要单独定义
        #输入深度为RGB图像，所以为3
        #输出通道为64
        #卷积核尺寸7*7
        #卷积核步长为2，图像输出尺寸缩减为输入的一半
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        #池化过程定义，与论文一致
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #定义残差块拼接层
        self.layer1 = self._make_layer(block, 64,  blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if(self.include_top):
            #平均池化下采样层
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512*block.expansion, num_classes)
        
        for m in self.modules():
            if(isinstance(m, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    #函数说明：构造网络层叠结构
    #参数说明：block表示基本残差模块结构
    #参数说明：channel表示通残差块卷积层第一层的道数量
    #参数说明：block_num表示当前模块需要重复几次
    #参数说明：stride表示步长
    def _make_layer(self, block, channel, block_num, stride=1):
        #初始化残差级联通道下采样不开启
        downsample = None
        #如果步长不为1，或者残差块输入通道数与输出通道数不存在倍数关系
        if((stride != 1) or (self.in_channel != channel * block.expansion)):
            #定义下采样行为
            downsample = nn.Sequential(
                #尺寸匹配为channel*block.expansion
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        #记录层叠结构，并存入layers
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        #输入通道数为通道*倍数
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))
        #注意列表的*转换
        return nn.Sequential(*layers)

    #定义正向传播过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if(self.include_top):
            x = self.avgpool(x)
            #展平处理
            x = torch.flatten(x, 1)
            x = self.fc(x)
        
        return x

def resnet18(num_classes = 1000, include_top = True):
    return ResNet18(Basicblock, [2,2,2,2], num_classes=num_classes, include_top=True)