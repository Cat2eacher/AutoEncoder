import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from load_data import *
from model_AE import *

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# 记录测试总次数
total_test_step = 0

'''
/**************************task1**************************/
搭建网络模型
创建对应实例
网络模型来自模块model_AE
from model_AE import *
/**************************task1**************************/
'''
# 搭建网络模型
# from model import *

# 创建模型实例并部署到设备
model = AutoEncoder()
model = model.to(device)

# 加载模型参数
model = torch.load('./params/model_v0_25.pth', map_location='cpu')
# 打印模型结构
# print("-------------------------模型结构-------------------------")
# print(model)

'''
/**************************task**************************/
Task
test 可视化
/**************************task**************************/
'''

# view_data = train_set.data[0].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
# encoded_data, _ = model(view_data)  # 提取压缩的特征值
# print(f"encoded_data的size大小为{encoded_data.shape}")

plt.figure(1, figsize=(10, 3))
plt.ion()
# 原数据和经过编码解码后的数据的比较
for i in range(20):
    # test_data = test_set.data[i].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
    test_data = test_set.data[i].view(1, 1, 28, 28).type(torch.FloatTensor)
    _, decoded_data = model(test_data)
    # print('输入数据维度：', test_set.data[i].size())
    print('输入数据维度：', test_data.size())
    # print('输出数据维度：', decoded_data.size())  # torch.Size([1, 1, 28, 28])
    decoded_data = decoded_data.view(28, 28)
    print('输出数据维度：', decoded_data.size())  # torch.Size([28, 28])
    plt.subplot(121), plt.title('test_data')
    plt.imshow(test_set.data[i].numpy(), cmap='Greys')
    plt.subplot(122), plt.title('decoded_data')
    plt.imshow(decoded_data.detach().numpy(), cmap='Greys')
    plt.pause(0.1)

plt.ioff()
plt.show()
