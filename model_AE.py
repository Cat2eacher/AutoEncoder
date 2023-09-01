import torch
from torch import nn

'''
/**************************task**************************/
Task 
搭建网络模型
/**************************task**************************/
'''


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        # encoder 学习特征
        # [batch_size, 784]--->[batch_size, 20]
        # 28*28=784
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # decoder 生成
        # [batch_size, 20]--->[batch_size, 784]
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
        )

    def forward(self, x):
        """
        :param x: [batch_size, 1, 28, 28]
        :return: [batch_size, 1, 28, 28]
        """
        batch_size = x.size(0)
        x = self.flatten(x)  # flatten，也可写成 x = x.view(batch_size, -1)
        encoded = self.encoder(x)  # encode
        decoded = self.decoder(encoded)  # decode
        decoded = decoded.view(batch_size, 1, 28, 28)  # reshape
        return encoded, decoded


'''
/**************************main**************************/
main
/**************************main**************************/
'''
if __name__ == '__main__':
    model = AutoEncoder()
    # batch_size = 64
    # image: 1@28x28
    input = torch.ones((64, 1, 28, 28))
    encoded, decoded = model(input)
    print(encoded.shape)
    print(decoded.shape)
