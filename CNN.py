import torch.nn as nn
import torch

class Swish(nn.Module):
    """Swish激活函数: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class CNNModel(nn.Module):

    def __init__(self, hidden_features=32,out_dim=1,**kwargs):
        super().__init__()
        # We increase the hidden dimension over layers. Here pre-calculated for simplicity.
        c_hid1 = hidden_features//2 #设置CNN网络结构中隐藏层的不同宽度
        c_hid2 = hidden_features
        c_hid3 = hidden_features*2

        # Series of convolutions and Swish activation functions
        #卷积结构
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1,c_hid1,kernel_size=5,stride=2,padding=4),
            Swish(),
            nn.Conv2d(c_hid1, c_hid2,kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid2,c_hid3 ,kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid3,c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Flatten(),#展平网络输出
            nn.Linear(c_hid3*4,c_hid3),  # 修正：最后一个卷积层输出是1x1，所以是c_hid3*1
            Swish(),
            nn.Linear(c_hid3,out_dim)
        )

    def forward(self,x):
        x = self.cnn_layers(x).squeeze(dim=-1)
        return x






    #前向

