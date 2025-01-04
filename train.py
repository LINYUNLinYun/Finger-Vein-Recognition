import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime  # 用于获取当前时间
import pandas as pd


# 定义数据集类，用于加载掌静脉图像数据
class VeinDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        初始化函数
        :param root: 数据集根目录路径，包含各个类别文件夹的上级目录
        :param transform: 可选的图像变换操作，用于对图像进行预处理
        """
        self.root = root
        self.transform = transform
        self.image_paths = []  # 存储图像文件的路径列表
        self.labels = []  # 存储对应图像的标签列表
        self.classes = []  # 存储类别名称列表
        self.class_to_idx = {}  # 类别名称到索引的映射字典
        self.idx_to_class = {}  # 索引到类别名称的映射字典
        self._find_classes()  # 查找数据集中的类别信息
        self._find_images()  # 查找每个类别下的图像文件路径及对应的标签

    def _find_classes(self):
        """
        查找数据集中的类别信息，即各个类别文件夹的名称
        将类别名称排序后存储到self.classes中，并建立类别名称与索引的双向映射字典
        """
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort()
        self.classes = classes
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        self.idx_to_class = {idx: class_name for idx, class_name in enumerate(classes)}

    def _find_images(self):
        """
        遍历每个类别文件夹，获取其中图像文件的路径，并根据类别名称确定对应的标签
        将图像路径添加到self.image_paths列表，对应的标签添加到self.labels列表
        """
        for class_name in self.classes:
            class_path = os.path.join(self.root, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

    def __getitem__(self, index):
        """
        根据给定的索引获取数据集中的一个样本（图像和对应的标签）
        :param index: 样本的索引
        :return: 处理后的图像数据（可能经过变换）和对应的标签
        """
        img_path = self.image_paths[index]
        label = self.labels[index]
        img = Image.open(img_path).convert('RGB')  # 打开图像并转换为RGB格式
        if self.transform is not None:
            img = self.transform(img)  # 对图像应用变换操作（如果有定义）
        return img, label

    def __len__(self):
        """
        返回数据集的样本数量，即图像的总数
        """
        return len(self.image_paths)


# 定义MobileNetV3网络结构类，用于掌静脉识别任务，修改此处以方便提取特征
class MobileNetV3(nn.Module):
    def __init__(self, num_classes=10, model_type='small'):
        """
        初始化函数
        :param num_classes: 分类的类别数量，默认是10
        :param model_type: MobileNetV3的类型，可选'small'或'large'，这里默认'small'
        """
        super(MobileNetV3, self).__init__()
        if model_type == 'small':
            self.model = torchvision.models.mobilenet_v3_small(pretrained=True)
        else:
            self.model = torchvision.models.mobilenet_v3_large(pretrained=True)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        # 定义特征提取部分，这里选择去掉最后的分类器层作为特征提取层，可根据需求调整
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        """
        定义前向传播过程，将输入数据通过网络模型得到输出，同时返回提取到的特征
        :param x: 输入数据
        :return: 网络的最终输出结果以及提取到的特征
        """
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # 展平特征，方便后续使用（可根据实际情况调整）
        output = self.model.classifier(features)
        return output, features


# 初始化模型的函数，根据给定参数创建并返回MobileNetV3模型实例，同时移动到指定设备上
def init_model(num_classes, model_type, device):
    """
    初始化模型
    :param num_classes: 分类的类别数量
    :param model_type: MobileNetV3的类型，可选'small'或'large'
    :param device: 训练使用的设备（如GPU或CPU）
    :return: 初始化好的模型实例并移动到指定设备上
    """
    model = MobileNetV3(num_classes=num_classes, model_type=model_type).to(device)
    print(model)
    return model


# 定义训练函数，用于训练模型
def train(model, train_loader, criterion, optimizer, device):
    """
    训练模型的函数，进行一个批次数据的训练过程
    :param model: 要训练的模型实例
    :param train_loader: 训练数据的加载器，提供批次数据
    :param criterion: 损失函数，用于计算预测结果与真实标签之间的损失
    :param optimizer: 优化器，用于更新模型的参数
    :param device: 训练使用的设备（如GPU或CPU）
    :return: 平均训练损失
    """
    model.train()  # 将模型设置为训练模式
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到指定设备上
        optimizer.zero_grad()  # 梯度清零，防止梯度累积
        outputs, features = model(inputs)  # 通过模型得到输出和特征
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 根据梯度更新模型参数
        running_loss += loss.item()  # 累计批次损失
    return running_loss / len(train_loader)  # 返回平均训练损失


# 定义测试函数，用于评估模型在测试集上的性能
def test(model, test_loader, criterion, device):
    """
    测试模型的函数，评估模型在测试集上的性能
    :param model: 要测试的模型实例
    :param test_loader: 测试数据的加载器，提供批次数据
    :param criterion: 损失函数，用于计算预测结果与真实标签之间的损失
    :param device: 测试使用的设备（如GPU或CPU）
    :return: 平均测试损失，测试准确率以及特征（可根据后续需求决定是否使用）
    """
    model.eval()  # 将模型设置为评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    all_features = []  # 用于存储所有批次的特征
    with torch.no_grad():  # 在测试过程中不需要计算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到指定设备上
            outputs, features = model(inputs)  # 通过模型得到输出和特征
            all_features.append(features.cpu().numpy())  # 将特征转移到CPU并转换为numpy数组后存储
            loss = criterion(outputs, labels)  # 计算损失
            running_loss += loss.item()  # 累计批次损失
            _, predicted = torch.max(outputs.data, 1)  # 获取预测的类别标签
            total += labels.size(0)  # 统计总样本数
            correct += (predicted == labels).sum().item()  # 统计正确预测的样本数
    return running_loss / len(test_loader), correct / total, np.concatenate(all_features)  # 返回平均测试损失、准确率和所有特征

# 定义函数用于加载模型并获取所有样本的特征保存到CSV文件
def load_model_and_extract_features(model_path, data_dir, batch_size=16, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    加载已保存的模型，遍历数据集中所有样本获取特征并保存到CSV文件
    :param model_path: 已保存的模型文件路径
    :param data_dir: 数据集所在的目录路径
    :param batch_size: 数据加载的批次大小，默认16
    :param device: 运行设备，默认优先使用GPU，若不可用则使用CPU
    """
    # 加载数据集，不进行数据增强变换（这里可根据实际情况决定是否需要变换，若之前训练时有特定变换，此处尽量保持一致）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = VeinDataset(data_dir, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型结构（需要与保存的模型结构一致）
    model = init_model(num_classes=10, model_type='small', device=device)
    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_features = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            _, features = model(inputs)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_features = np.concatenate(all_features)
    # 将特征和标签组合成DataFrame，方便保存为CSV
    df = pd.DataFrame(all_features)
    df['label'] = all_labels
    # 保存CSV文件，文件名可根据实际情况调整
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # df.to_csv('extracted_features.csv', index=False)
    df.to_csv(f'./feature/extracted_features_{now_time}.csv', index=False)
    print("特征已成功保存到extracted_features.csv文件中")


# 定义主函数，整合整个流程，包括数据加载、模型训练、保存、加载和测试等操作
def main(data_dir='train_finger', num_classes=10, num_epochs=50, learning_rate=0.001, batch_size=10, model_type='small'):
    # 设置随机种子，确保实验的可重复性
    torch.manual_seed(0)
    np.random.seed(0)

    # 设置设备，优先使用GPU，如果GPU不可用则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置超参数
    # batch_size = 10  # 每个批次的数据量大小
    num_classes = 10  # 分类的类别数量
    # num_epochs =   # 训练的轮数
    # learning_rate = 0.001  # 学习率，用于优化器更新参数的步长

    # # 设置数据集路径，这里假设训练数据在名为'train'的文件夹下
    # data_dir = 'train_finger'

    # 设置数据集的变换操作，包含了一些增强操作
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=10),
        # transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集，创建VeinDataset实例，并应用定义好的变换操作
    dataset = VeinDataset(data_dir, transform)

    # 划分训练集和测试集，按照8:2的比例划分
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建数据加载器，用于按批次加载训练集和测试集数据
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型并移动到指定设备上
    model = init_model(num_classes, 'small', device)

    # 定义损失函数，这里使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器，使用Adam优化器来更新模型参数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 记录训练开始时间，用于创建保存模型的文件夹以及文件名相关信息
    start_train_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # 用于记录训练过程中的损失和准确率
    train_losses = []
    test_losses = []
    test_accuracies = []
    best_test_acc = 0  # 记录最高的测试集准确率
    best_epoch = 0  # 记录达到最高测试准确率的轮次

    # 训练模型
    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_features = test(model, test_loader, criterion, device)

        print('test features shape:', test_features.shape)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch + 1, num_epochs, train_loss, test_loss, test_acc))

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            # 获取当前时间并格式化为字符串，用于命名模型文件，文件名包含更多模型相关信息
            current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            model_folder = os.path.join("model", f'{data_dir}_'+start_train_time)
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            model_path = os.path.join(model_folder, f"mobilenetv3_small_epoch_{epoch + 1}_acc_{test_acc:.4f}_{current_time}.pth")
            torch.save(model.state_dict(), model_path)

    end_time = time.time()
    print('Time: {:.2f}s'.format(end_time - start_time))
    print(f'Best test accuracy: {best_test_acc} was achieved at epoch {best_epoch}')

    # 绘制训练和测试的损失、准确率曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Testing Accuracy')
    plt.legend()

    plt.savefig(os.path.join(model_folder, f"loss_and_accuracy_curves.png"))  # 保存图像到当前目录
    plt.show()


if __name__ == '__main__':
    # main(data_dir='train_hand',batch_size=4,num_epochs=100)
    load_model_and_extract_features('model/train_hand_2025-01-04-11-15-12/mobilenetv3_small_epoch_69_acc_0.9333_2025-01-04-11-16-40.pth', 'train_hand')