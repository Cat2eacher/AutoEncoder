# AutoEncoder
MIIST-based AutoEncoder
基于MNIST数据集的自编码器

上传了初始版本的AutoEncoder
- 1、load_data.py 文件导入MNIST数据集，可以进行数据集可视化
- 2、model_AE.py 是搭建的神经网络模型。这里是线性层搭建的神经网络
- 3、train.py 最基本的训练过程
- 4、train_plt.py （optional）训练过程对训练效果的可视化，通过plt实现
- 4、train_tensorboard.py （optional）训练过程对训练效果的可视化，通过tensorboard实现

==注：目前存在问题是，运行train_plt.py文件，在CPU上运行没有问题。在GPU上会报数据出现在两种设备的错误。
还不知道怎么修正。==
