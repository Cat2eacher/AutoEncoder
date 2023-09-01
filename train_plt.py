import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

from load_data import *
from model_AE import *

# 超参数
# BATCH_SIZE 在load_data module 中设置
NUM_EPOCHS = 50  # 训练轮数
'''
/**************************task1**************************/
Task 1
准备并加载数据集
/**************************task1**************************/
'''
# from load_data import *
# load_data

'''
/**************************task2**************************/
设备device
/**************************task2**************************/
'''
# ----------------------------------------------------#
# 定义训练的设备device
# ----------------------------------------------------#

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
'''
/**************************task3**************************/
Task 3
搭建网络模型
创建对应实例
网络模型来自模块model_AE
from model_AE import *
/**************************task3**************************/
'''
# 搭建网络模型
# from model import *

# 创建模型实例并部署到设备
model = AutoEncoder()
model = model.to(device)
print("-------------------------模型创建成功------------------------")

'''
/**************************task4**************************/
Task 4
训练过程
/**************************task4**************************/
'''
# 保存路径
if not os.path.exists("params"):
    os.mkdir("params")

# 损失函数loss_fn
criterion = torch.nn.MSELoss()  # mean square error
criterion = criterion.to(device)

# 优化器
learning_rate = 1e-1
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.6)  # 设置学习率下降策略


# 可视化部分原图
num_imgs = 5  # 可视化的图像数目
# fig是一块画布；axes是一个大小为[nrows,ncols]的数组，数组中的每个元素都是一个图对象
fig, axes = plt.subplots(nrows=2, ncols=num_imgs, figsize=(5, 2))
plt.ion()  # Turn the interactive mode on, continuously plot
# matplotlib 的显示模式转换为交互模式，遇到 plt.show()代码会继续执行

# original data (first row) for viewing
view_data = train_set.data[:num_imgs].type(torch.FloatTensor) / 255.
print(view_data.shape)
# view_data 是一个形状为 (num_imags, 28, 28) 的张量
for i in range(num_imgs):
    axes[0][i].imshow(view_data.data.numpy()[i], cmap='gray')
    axes[0][i].set_xticks(())
    axes[0][i].set_yticks(())

# train and visualization
for epoch in range(NUM_EPOCHS):
    print("-------第 {} 轮训练开始-------".format(epoch + 1))
    # 训练开始
    model.train()
    for i, (imgs, targets) in enumerate(train_dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        # 1. forward
        encoded, decoded = model(imgs)
        loss = criterion(decoded, imgs)
        # 2. reset gradient
        optimizer.zero_grad()
        # 3. backward
        loss.backward()
        # 4. update parameters of net
        optimizer.step()
        # 完成一次训练
        # 绘图
        if i % 100 == 0:
            print(
                "epochs:[{}],iteration:[{}]/[{}],loss:{:.3f}".format(epoch + 1, i, len(train_dataloader), loss.float()))
            # plotting decoded image (second row)
            _, decoded_data = model(view_data)
            for j in range(num_imgs):
                axes[1][j].clear()
                axes[1][j].imshow(np.reshape(decoded_data.cpu().data.numpy()[j], (28, 28)), cmap='gray')
                axes[1][j].set_xticks(())
                axes[1][j].set_yticks(())
            plt.draw()
            plt.pause(0.05)

    # 5. update LR
    print("第 %d 轮epoch的学习率：%f" % (epoch + 1, optimizer.param_groups[0]['lr']))
    scheduler_lr.step()

    # model_save
    # torch.save(model, "./params/model_v0_{}.pth".format(epoch + 1))

plt.ioff()  # Turn the interactive mode off
plt.show()

print('________________________________________')
print('finish training')
