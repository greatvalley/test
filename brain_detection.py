"""step1:自定义数据集"""
import glob
import os
import random
import time
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from torchvision import models

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

# 用于设置随机数生成器的种子，若设置种子数为 0 ，表明每次运行代码时都会生成相同的随机数序列
torch.manual_seed(42)
# 用于控制是否使用确定性的卷积算法，若设置为 False ，使得每次运行代码时都会生成不同的随机数序列，从而使用不同的卷积结果
torch.backends.cudnn.deterministic = False
# 用于控制是否使用CUDNN的自动调优功能，若设置为 True ，禁用CUDNN的自动调优功能，从而节省计算资源和时间
torch.backends.cudnn.benchmark = True
# def seed_torch(seed=42):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True


import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

"""glob.glob()函数，用于查找文件目录和文件，并将搜索到的结果返回到一个列表中"""
train_path = glob.glob('./脑PET图像分析和疾病预测挑战赛公开数据/Train/*/*')
test_path = glob.glob('./脑PET图像分析和疾病预测挑战赛公开数据/Test/*')

"""np.random.shuffle()函数，用于将一个数组或列表中的元素随机打乱"""
np.random.shuffle(train_path)
np.random.shuffle(test_path)

# 定义一个 DATA_CACHE 的空的字典，用于缓存放置图片地址
DATA_CACHE = {}

# 定义 MyDataset 类，用于处理图像数据集
class MyDataset(Dataset):
    # __init__ 方法：初始化函数，接收两个参数。img_path：图像的路径列表；transform：可选图像转换操作（默认为 None）
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    #__getitem__方法：用于获取指定索引的图像和标签
    def __getitem__(self, index):
        if self.img_path[index] in DATA_CACHE:
            img = DATA_CACHE[self.img_path[index]]
        else:
            img = nib.load(self.img_path[index])
            img = img.dataobj[:, :, :, 0]
            DATA_CACHE[self.img_path[index]] = img

        idx = np.random.choice(range(img.shape[-1]), 50)
        img = img[:, :, idx]
        img = img.astype(np.float32)

        if self.transform is not None:
            img = self.transform(image = img)['image']

        img = img.transpose([2, 0, 1])
        return img, torch.from_numpy(np.array(int('NC' in self.img_path[index])))

    def __len__(self):
        return len(self.img_path)

"""step2:自定义CNN模型"""
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        model = models.resnet18(True)
        model.conv1 = torch.nn.Conv2d(50, 64, kernel_size=(7, 7), stride=(2, 2),
                                      padding=(3, 3), bias=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 2)
        self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        end_time = time.time()
        return out


from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import albumentations as A
skf = KFold(n_splits=10, random_state=233, shuffle=True)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_path)):
    print(train_idx.shape)
    model = MyNet()
    model = model.to('cuda')

    #混合精度
    scaler = GradScaler()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), 0.0001)

    train_loader = torch.utils.data.DataLoader(
        MyDataset(np.array(train_path)[train_idx],
                  A.Compose([
                    A.RandomRotate90(),
                    A.RandomCrop(120, 120),
                    A.AdvancedBlur(blur_limit=(3, 7), p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomContrast(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                  ])
        ), batch_size=2, shuffle=True, num_workers=0, pin_memory=False
    )

    val_loader = torch.utils.data.DataLoader(
        MyDataset(np.array(train_path)[val_idx],
                  A.Compose([
                    A.RandomCrop(120, 120),
                  ])
        ), batch_size=2, shuffle=True, num_workers=0, pin_memory=False
    )


    """step3:模型训练与验证"""


    def train(train_loader, model, criterion, optimizer):
        model.train()
        train_loss = 0.0
        for i, (input, target) in enumerate(train_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            #使用混合精度进行前向和反向传播
            with autocast():
                output = model(input)
                loss = criterion(output, target.long())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % 20 == 0:
                print(loss.item())

            train_loss += loss.item()

        return train_loss / len(train_loader)


    def validate(val_loader, model, criterion):
        model.eval()
        val_acc = 0.0

        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                input = input.cuda()
                target = target.cuda()

                output = model(input)
                loss = criterion(output, target.long())

                val_acc += (output.argmax(1) == target).sum().item()

        return val_acc / len(val_loader.dataset)


    training_start_time = time.time()
    result_folder = 'results'
    os.makedirs(result_folder, exist_ok=True)
    tensorboard_logs_folder = os.path.join(result_folder, 'tensorboard_logs')
    os.makedirs(tensorboard_logs_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_logs_folder)


    for _ in range(10):
        epoch_start_time = time.time()
        print(f"-----第{_+1}轮-----")
        train_loss = train(train_loader, model, criterion, optimizer)
        val_acc = validate(val_loader, model, criterion)
        train_acc = validate(train_loader, model, criterion)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f'训练损失：{train_loss}, 训练准确率:{train_acc}, 验证准确率:{val_acc}, 耗时{epoch_time}秒')

        #记录训练日志
        with open(os.path.join(result_folder, 'train_log.txt'), 'a') as f:
            f.write(f'第{_ + 1}轮：训练损失：{train_loss}, 训练准确率:{train_acc}, 验证准确率:{val_acc}, 耗时{epoch_time}秒')


        #可视化
        writer.add_scalar("Train_loss", train_loss / len(train_loader), _+1)
        writer.add_scalar("Train_acc", train_acc, _+1)
        writer.add_scalar("Val_acc", val_acc, _+1)

        #保存模型
        torch.save(model.state_dict(), os.path.join(result_folder, f'resnet18_fold{fold_idx}.pth'))
        training_end_time = time.time()
        total_time = training_end_time - training_start_time
        print(f'总训练时间：{total_time}秒')

    # #保存训练好的模型
    # torch.save(model.state_dict(), f'model_fold_{fold_idx}.pth')
    #
    # #加载之前保存的模型作为下一个折叠的初始模型
    # model.load_state_dict(torch.load(f'model_fold_{fold_idx}.pth'))
    #
    # #在验证集上评估模型性能
    # val_acc = validate(val_loader, model, criterion)
    # print(f'Fold{fold_idx + 1} - Validation Accuracy: {val_acc}')

writer.close()

test_loader = torch.utils.data.DataLoader(
    MyDataset(test_path,
              A.Compose([
                A.RandomCrop(128, 128),
                A.HorizontalFlip(p=0.5),
                A.RandomContrast(p=0.5),
              ])
    ), batch_size=2, shuffle=False, num_workers=0, pin_memory=False
)

"""step4:模型预测与提交"""
def predict(test_loader, model, criterion):
    model.eval()
    val_acc = 0.0

    test_pred = []
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            test_pred.append(output.data.cpu().numpy())

    return np.vstack(test_pred)


pred = None
for _ in range(10):
    if pred is None:
        pred = predict(test_loader, model, criterion)
    else:
        pred += predict(test_loader, model, criterion)

submit = pd.DataFrame(
    {
        'uuid': [int(x.split('\\')[-1][:-4]) for x in test_path],
        'label': pred.argmax(1)
    }
)

submit['label'] = submit['label'].map({1: 'NC', 0: 'MCI'})
submit = submit.sort_values(by='uuid')
submit.to_csv('submit2.csv', index=None)

print("a")
print("b")
print("v")