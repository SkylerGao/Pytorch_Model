import torch
import ResNet18
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

#检测GPU设备信息，如果GPU不可用就采用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Compse串联多个图片的变换操作，类似Sequence
data_transform = transforms.Compose(
    [
        transforms.Resize(256),     #图片resize
        transforms.CenterCrop(224), #图片截取以图片中心为原点，尺寸为224*224的区域
        transforms.ToTensor(),      #转换为Tensor，将0-255的图像数据转换为0-1
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) #将0到1转化为0到-1
    ]
)

img = Image.open("E:\\A_博士学位\\E_数据集\\Flower\\val\\daisy\\15207766_fc2f1d692c_n.jpg")
plt.imshow(img)

#将图像进行Compose中指定的变换
img = data_transform(img)

#img维度扩充
img = torch.unsqueeze(img, dim=0)
#模型例化
model = ResNet18.resnet18(num_classes=5)
#模型权重读取
model_weight_path = "./resNet.pth"
#载入模型权重，并将模型置于GPU上
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()

with torch.no_grad():
    output = torch.squeeze(model(img))
    predict = torch.softmax(output,dim = 0)
    predict_cla = torch.argmax(predict).numpy()

print(class_indict[str(predict_cla)], predict[predict_cla].numpy())
plt.show()


