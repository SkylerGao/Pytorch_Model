import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import ResNet18

#设备选取GPU/CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#数据转换信息
data_transform = {

#训练集操作：
#1. 随机剪裁区域，大小(224,224)
#2. 随机水平翻转，默认概率0.5
#3. 转换为tensor
#4. 标准化
"train": transforms.Compose([transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

#训练集操作：
#1. 图片大小变为(256,256)
#2. 以中心剪裁(224,224)
#3. 转换为tensor
#4. 标准化
"val": transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

image_path = "E:\\A_博士学位\\E_数据集\\Flower\\"

#读取指定路径下的所有图片，并形成数据集格式，每个文件夹会被标记成一个label
#每个文件下下的图片具有其文件夹的label
train_dataset = datasets.ImageFolder(image_path + "train", transform=data_transform["train"])
validata_dataset = datasets.ImageFolder(image_path + "val", transform=data_transform["val"])

#获取所有训练/验证数据集的数量
train_num = len(train_dataset)
val_num   = len(validata_dataset)

#获取数据集的索引与label
flower_list = train_dataset.class_to_idx

#将训练集的索引与label存入dict
cla_dict = dict((val, key) for key, val in flower_list.items())

#设定训练过程中的batch_size
batch_size = 16

#将数据集转化为可以迭代索引的格式
train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0
                    )
validate_loader = torch.utils.data.DataLoader(
                    validata_dataset,
                    batch_size=batch_size, 
                    shuffle=False,
                    num_workers=0
                    )

#例化网络
net = ResNet18.resnet18()

#因为输出数据一共有5类，所以需要将最后一层fc设置为5个输出
inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel, 5)
#将网络搬到GPU上
net.to(device)

#交叉熵损失函数
loss_function = nn.CrossEntropyLoss()
#优化器
optimizer = optim.Adam(net.parameters(), lr=0.0001)

best_acc = 0.0
#权重保存路径
save_path = './resNet18.pth'

#一共训练3个周期（所有数据集走3遍）
for epoch in range(50):
    #告诉网络准备执行train操作，会有一些batch的优化操作
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        #取得图像和标签
        images, labels = data
        #每次计算新的grad时，要把原来的grad清零
        optimizer.zero_grad()
        #正向计算
        logits = net(images.to(device))
        #正向输出和标签进行误差计算
        loss = loss_function(logits, labels.to(device))
        #反向传播
        loss.backward()
        #更新梯度信息
        optimizer.step()

        #计算损失误差
        running_loss += loss.item()
        #训练情况显示
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
    print()

    #验证训练
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    #不追踪梯度变化，grad不被track
    with torch.no_grad():
        #测试集验证
        for val_data in validate_loader:
            #获取测试图像和测试label
            val_images, val_labels = val_data
            #获取输出结果
            outputs = net(val_images.to(device)) 
            # loss = loss_function(outputs, test_labels)
            #预测值等于outputs中每行最大值的索引
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')
