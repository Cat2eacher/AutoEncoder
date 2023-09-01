import os
import time
import torch
import torchvision

from load_data import *
from model_AE import *

start_time = time.time()

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

# train
for epoch in range(NUM_EPOCHS):
    print("-------第 {} 轮训练开始-------".format(epoch + 1))
    # 训练开始
    model.train()
    for index, (imgs, targets) in enumerate(train_dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        # 1. forward
        _, decoded = model(imgs)
        loss = criterion(decoded, imgs)
        # 2. reset gradient
        optimizer.zero_grad()
        # 3. backward
        loss.backward()
        # 4. update parameters of net
        optimizer.step()
        # 完成一次训练
        if index % 100 == 0:
            print(
                "epochs:[{}],iteration:[{}]/[{}],loss:{:.3f}".format(epoch + 1, index, len(train_dataloader),
                                                                     loss.float()))
    # 5. update LR
    # print("第%d轮epoch的学习率：%f" % (epoch+1, optimizer.param_groups[0]['lr']))
    scheduler_lr.step()

    # model_save
    torch.save(model, "./params/model_v0_{}.pth".format(epoch + 1))

print('________________________________________')
print('finish training')

end_time = time.time()
print('训练耗时：', (end_time - start_time))
