import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# 定义数据预处理
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, padding=4),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载CIFAR-10数据集
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=4)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,shuffle=False, num_workers=4)

# 定义ResNet模型
net = resnet18()

# 如果有GPU可用，将模型移至GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

if __name__ == '__main__':
    # torch.multiprocessing.freeze_support()  # 多进程支持

    # 训练网络
    for epoch in range(100):  # 遍历数据集多次

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播，反向传播，优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印统计信息
            running_loss += loss.item()
            if i % 180 == 179:  # 每200个小批量打印一次
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # 在测试数据上测试网络性能
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on the test dataset: {100 * correct / total:.2f}%')
