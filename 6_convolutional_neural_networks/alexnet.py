import time
import numpy as np
import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import torch_directml
import sys
# sys.path.append('../3_linear_network')
# sys.path.append('../6_convolutional_neural_networks')
# 将load_data_fashion_mnist函数所在路径加入系统环境变量路径中

# from softmax_regression_scratch import load_data_fashion_mnist, Accumulator, Animator, accuracy

# from convolution_LeNet import train_ch6, try_gpu
device = torch_directml.device(0)
print(torch_directml.device_name(0))
# device = 'cpu'

def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 1

def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]   # 将图像从PIL类型转换成张量
    if resize:
        trans.insert(0, transforms.Resize(resize)) # resize是把小图片放大,在0之前插入
    trans = transforms.Compose(trans)   # 组合多个transform
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data/", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data/", train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False))

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  #判断y_hat是一个矩阵
        y_hat = y_hat.argmax(axis=1)  # torch.argmax
    cmp = y_hat.type(y.dtype) == y    # 将y_hat转换成与y相同的数据类型，便于比较
    return float(cmp.type(y.dtype).sum()) # cmp是bool向量，转换为与y相同的数据类型，可以求和

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n  # 生成初始值为0的含n个元素的列表

    def add(self, *args):
        # 对n个变量分别累加
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]   # 类中定义__getitem__函数，可以获取指定序号的数据


class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            net = net.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

alexnet = nn.Sequential(
    nn.Conv2d(1, 48, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(48, 128, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(128, 192, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(192, 192, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(192, 128, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(3200, 2048), nn.ReLU(),
    nn.Dropout(p=0.5),
    # nn.Linear(4096, 4096), nn.ReLU(),
    # nn.Dropout(p=0.5),
    nn.Linear(2048, 10))

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):# 比之前的train_ch3多个device参数
    """用GPU训练模型(在第六章定义)"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    print('training on', device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    # animator = Animator(xlabel='epoch', xlim=[1, num_epochs],legend=['train loss', 'train acc', 'test acc'],figsize=(12,8))
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = Accumulator(2)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad(set_to_none=True)
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[1]
            # train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(f'epoch {epoch} num_batch {i} train loss {train_l}')
            #     animator.add(epoch + (i + 1) / num_batches,
            #                  (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter, device)
        # animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[1] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.01, 10
train_ch6(alexnet, train_iter, test_iter, num_epochs, lr, device)