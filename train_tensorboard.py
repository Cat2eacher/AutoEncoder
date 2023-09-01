import os
import torch
from torch.utils.tensorboard import SummaryWriter

from load_data import *
from model_AE import *

# 超参数
# BATCH_SIZE 在load_data module 中设置
NUM_EPOCHS = 5  # 训练轮数
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
/**************************task3**************************/
Task 3
训练过程
/**************************task3**************************/
'''
# 保存路径
if not os.path.exists("params"):
    os.mkdir("params")
if not os.path.exists("log"):
    os.mkdir("log")

# 损失函数loss_fn
criterion = torch.nn.MSELoss()  # mean square error
criterion = criterion.to(device)

# 优化器
learning_rate = 1e-1
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 设置学习率下降策略

# 记录训练总次数
total_train_step = 0

# 添加tensorboard，通常使用writer作为类名
writer = SummaryWriter("./log")

# train and visualization
for epoch in range(NUM_EPOCHS):
    print("-------第 {} 轮训练开始-------".format(epoch + 1))
    # 训练步骤开始
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
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print(f'训练次数：{total_train_step}, Loss: {loss.item()}')
            # 显示每轮迭代效果，并可视化损失函数迭代过程
            writer.add_scalar(tag="train_loss", scalar_value=loss.item(), global_step=total_train_step)
            # 维度顺序的转换
            # print(decoded.shape) # torch.Size([64, 1, 28, 28])
            # [batch_size, Channel, Weight, Height] -> [Channel, Weight, Height]
            writer.add_image(tag="decoded_imgs", img_tensor=decoded.squeeze(0), global_step=total_train_step,
                             dataformats='NCHW')

    # 5. update LR
    print("第 %d 轮epoch的学习率：%f" % (epoch + 1, optimizer.param_groups[0]['lr']))
    scheduler_lr.step()

    # model_save
    # torch.save(model, "./params/model_tb_{}.pth".format(epoch + 1))
writer.close()

# fake_image = out_img.cpu().data
# real_image = img.cpu().data
# save_image(fake_image, "./img/epohs-{}-fake_img.jpg".format(epochs), nrow=10)
# save_image(real_image, "./img/epohs-{}-real_img.jpg".format(epochs), nrow=10)
# torch.save(self.net.state_dict(), "./params/net.pth")