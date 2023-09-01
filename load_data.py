import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 超参数
BATCH_SIZE = 64

'''
/**************************task1**************************/
Task 1
准备并加载数据集
/**************************task1**************************/
'''

# 官方网站：http://yann.lecun.com/exdb/mnist/
# 每个图像都是28x28大小的单通道灰度图像，像素值0~255
# 训练集 60,000个样本
# 测试集 10,000个样本

# 准备数据集
dataset_transform = transforms.Compose([
    transforms.ToTensor()
])
train_set = torchvision.datasets.MNIST(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.MNIST(root="./dataset", train=False, transform=dataset_transform, download=True)

# length 长度
train_set_size = len(train_set)
test_set_size = len(test_set)

# 如果train_data_size=10, 训练数据集的长度为：10
print("-------------------------数据集导入-------------------------")
print("训练数据集的长度为：{}".format(train_set_size))
print("测试数据集的长度为：{}".format(test_set_size))

# DataLoader 来加载数据集
train_dataloader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
test_dataloader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

print("dataloader加载成功，batch size大小：{}".format(BATCH_SIZE))
print("--------------------------导入成功--------------------------")
'''
/**************************main**************************/
main
/**************************main**************************/
'''
if __name__ == '__main__':
    # ---------------------------------------------------
    # 展示部分数据集1
    img, tar = train_set[0]
    print(f'输入图像大小为{img.shape}')
    print(f'类别及对应标签为{train_set.classes}')
    # ---------------------------------------------------
    # matplot展示部分数据集2
    fig = plt.figure()
    for i in range(15):
        plt.subplot(3, 5, i+1)
        plt.imshow(train_set.data[i], cmap='gray', interpolation='none')
        plt.title("Labels: {}".format(train_set.targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # for data in test_dataloader:
    #     image, target = data
    #     print(image.shape)

    # ---------------------------------------------------
    # Tensorboard展示部分数据集3
    # writer = SummaryWriter('./log')
    # # 显示test_set中第一张图像
    # writer.add_image("img1", img, 1)
    #
    # # 显示test_set中第1~10张图像
    # for i in  range(10):
    #     img, tar = test_set[i]
    #     writer.add_image("img1~10",img,i)
    #
    # # 显示所有test_set图像，每次显示一个batch_size的图像
    # step = 0
    # for data in test_loader:
    #     image,target = data
    #     writer.add_images("img_batch", image, step)
    #     step = step+1
    # # 把整个dataset显示，每次显示一个batch_size的图像
    # for epoch in range(2):  # 进行两轮， shuffle  打乱顺序
    #     step = 0
    #     for data in test_loader:
    #         imgs, targets = data
    #         # print(imgs.shape)   # torch.Size([4, 3, 32, 32])
    #         #  print(targets)  # tensor([2, 8, 0, 2])  随机从 Data 中，抓取 4 个数据
    #         writer.add_images("Epoch: {}".format(epoch), imgs, step)
    #         step = step + 1
    #
    # writer.close()
