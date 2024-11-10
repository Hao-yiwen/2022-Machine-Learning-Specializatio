# predict_digit.py

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps


# 定义 SimpleNet 模型结构
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 加载模型
model = SimpleNet()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # 将图像转为灰度
    image = ImageOps.invert(image)  # 反转颜色
    image = transform(image).unsqueeze(0)  # 预处理并添加 batch 维度
    return image

def predict_image(image):

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()



# 测试预测
image_path = 'test.jpg'  # 替换为你的手写数字图像路径
predicted_digit = predict_image(preprocess_image(image_path))
print(f'预测结果: {predicted_digit}')