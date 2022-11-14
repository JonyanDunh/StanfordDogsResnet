import torch
from pathlib import Path
from os import listdir
from os.path import isfile, join
import re
from torchvision.io import image as Image
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
import matplotlib.pyplot as plt
import numpy as np


def listData():
    Images_dir = "/data/Images/"
    pattern = re.compile('^\/data\/Images\/(?P<dogs_id>[a-zA-Z0-9 ]*)-(?P<dogs_class>[a-zA-Z0-9 ]*)')
    dog_id_count = 0
    dogs_class_dir = [
        (str(dir), pattern.search(str(dir)).group('dogs_id'), pattern.search(str(dir)).group('dogs_class')) for dir in
        [e for e in Path(Images_dir).iterdir() if e.is_dir()]]
    class_map_id_number = {}
    class_map_name = {}
    for _, dog_id_str, dog_class in dogs_class_dir:
        class_map_id_number[dog_id_str] = dog_id_count
        class_map_name[dog_id_count] = dog_class
        dog_id_count += 1
    return dogs_class_dir, class_map_name, class_map_id_number


def Image2Tensor(save=False):
    p = transforms.Compose([transforms.Resize((256, 256))])
    files = []
    for mypath, dogs_id, dogs_class in dogs_class_dir:
        for f in listdir(mypath):
            if isfile(join(mypath, f)):
                img = p(Image.read_image(mypath + "/" + f)).cuda()
                if img.shape[0] == 3:
                    files.append((img, torch.zeros(1, 120).scatter_(1, torch.tensor(
                        [class_map_id_number[dogs_id]]).unsqueeze(1), 1.0).cuda()))
    imgs = torch.stack([img_t for img_t, _ in files], dim=3)
    labels = torch.stack([label for _, label in files], dim=2)
    if (save):
        torch.save(imgs, "/data/saved_tensor/dogs_classification/images_256.pt")
        torch.save(labels, "/data/saved_tensor/dogs_classification/labels.pt")
        print("save success!")
    return imgs, labels


def loadData():
    dogs_images_data = torch.load("/data/saved_tensor/dogs_classification/images.pt").permute(3, 0, 1, 2).to(
        dtype=torch.float32)
    dogs_labels_data = torch.load("/data/saved_tensor/dogs_classification/labels.pt").squeeze(0).permute(1, 0)
    n_val = int(0.1 * dogs_images_data.shape[0])
    shuffled_indices = torch.randperm(dogs_images_data.shape[0])
    train_dogs_images_data = dogs_images_data[shuffled_indices[:-n_val]]
    train_dogs_labels_data = dogs_labels_data[shuffled_indices[:-n_val]]
    val_dogs_images_data = dogs_images_data[shuffled_indices[-n_val:]]
    val_dogs_labels_data = dogs_labels_data[shuffled_indices[-n_val:]]
    train_data = [(train_dogs_images_data[i, :, :, :], train_dogs_labels_data[i, :]) for i in
                  range(train_dogs_images_data.shape[0])]
    val_data = [(val_dogs_images_data[i, :, :, :], val_dogs_labels_data[i, :]) for i in
                range(val_dogs_images_data.shape[0])]
    return train_data, val_data


def show_image(model):
    plt.figure(figsize=(15, 15))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(val_data[i][0].to(dtype=torch.uint8).cpu().permute(1, 2, 0), cmap=plt.cm.binary)
        with torch.no_grad():
            print(train_data[i][1].argmax(0))
            print(train_data[i][0].unsqueeze(0))
            print(train_data[i][0])
            print(model(train_data[i][0].unsqueeze(0)).argmax(1))
            plt.xlabel("origin:" + class_map_name[int(train_data[i][1].argmax(1).cpu())] + "\n"
                       + "predicted:" + class_map_name[int(model(train_data[i][0].unsqueeze(0)).argmax(1).cpu())])

    plt.show()

def show_training_trend():
    plt.title('The trend of training')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy or Loss')
    x=range(len(Accuracy_train_list))
    plt.plot(x,Accuracy_train_list, 'b',label="Accuracy of Training Data", alpha = 0.7)
    plt.plot(x,Accuracy_val_list, 'r',label="Accuracy of Validation Data", alpha = 0.7)
    plt.plot(x, Loss_list, 'g', label="Loss", alpha = 0.3)
    plt.legend()
    plt.show()

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, one_hot_labels in train_loader:
            outputs = model(imgs)
            loss = loss_fn(outputs, one_hot_labels.squeeze_(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        Loss_list.append(loss_train / len(train_loader))
        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch,
            loss_train / len(train_loader)))
        validate(model)
    torch.save(model.state_dict(), '/data/saved_model/dogs_classification/dogs_classification_model_2.pt')


def validate(model):
    train_loader2 = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=False)
    val_loader2 = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)
    for name, loader in [("train", train_loader2), ("val", val_loader2)]:
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, one_hot_labels in loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                _, labels = torch.max(one_hot_labels, dim=1)
                total += one_hot_labels.shape[0]
                correct += int((predicted == labels).sum())
        if name == 'train':
            Accuracy_train_list.append(correct / total)
        if name == 'val':
            Accuracy_val_list.append(correct / total)
        print("Accuracy {}: {:.4f}".format(name, correct / total))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # 如果步长不为1，或者输入与输出通道不一致，则需要进行Projection Shortcut操作
        if stride != 1 or in_planes != self.expansion * planes:
            # Projection Shortcut
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # 依次通过两个卷积层，和shortcut连接层，再累加起来。
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()
        self._config = config
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(config['block_type'], 64, config['num_blocks'][0], stride=1)
        self.layer2 = self._make_layer(config['block_type'], 128, config['num_blocks'][1], stride=2)
        self.layer3 = self._make_layer(config['block_type'], 256, config['num_blocks'][2], stride=2)
        self.layer4 = self._make_layer(config['block_type'], 512, config['num_blocks'][3], stride=2)
        self.linear = nn.Linear(512 * config['block_type'].expansion, config['num_classes'])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, planes, stride))
            self.in_channels = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.max_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


dogs_class_dir, class_map_name, class_map_id_number = listData()
train_data, val_data = loadData()
resnet_config = \
    {
        'block_type': BasicBlock,
        # 'num_blocks': [3,4,6,3], #ResNet34
        'num_blocks': [2, 2, 2, 2],  # ResNet18
        'num_classes': 120
    }
model = ResNet(resnet_config).to(device=torch.device('cuda'))
# model.load_state_dict(torch.load('/data/saved_model/dogs_classification/dogs_classification_model_1.pt'))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
Accuracy_train_list = []
Accuracy_val_list = []
Loss_list = []
Loss_list.reverse()
print("Number of parameter: %.2fM" % (sum([param.nelement() for param in model.parameters()]) / 1e6))
print(model._modules)

training_loop(n_epochs=100, optimizer=optim.AdamW(model.parameters(), lr=0.001), model=model,loss_fn=nn.CrossEntropyLoss().cuda(), train_loader=train_loader, )
print(Accuracy_train_list)
print(Accuracy_val_list)
print(Loss_list)
show_training_trend()
# show_image(model)
