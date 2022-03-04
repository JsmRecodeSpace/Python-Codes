


# 목차
* Pytorch 훈련 코드 예시
* import codes







---------- Pytorch 훈련 코드 예시 ----------

    # 이미지 train, valid, test - Ex 1
# https://www.dacon.io/competitions/official/235874/codeshare/4584?page=1&dtype=recent
transform = transforms.Compose([
    transforms.ToTensor(), #이미지 데이터를 tensor 데이터 포멧으로 바꾸어줍니다.
    transforms.Resize([224,224]), #이미지의 크기가 다를 수 있으니 크기를 통일해 줍니다.
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) #픽셀 단위 데이터를 정규화 시켜줍니다.
])

# Pytorch의 ImageFolder 메소드를 사용하면 folder의 이름으로 자동으로 라벨링 됨
train_data = datasets.ImageFolder(root='./data/train/', transform=transform)

# 평가를 위해 train데이터에서 validation 데이터를 나누어 줌
train_idx, valid_idx = train_test_split(np.arange(len(train_data)), test_size=0.2,
                                       random_state=42, shuffle=True, stratify=train_data.targets)

batch_size = 4
num_workers = int(cpu_count() / 2)

train_loader = DataLoader(train_data, batch_size=batch_size,
                         sampler=SubsetRandomSampler(train_idx), num_workers=num_workers)
valid_loader = DataLoader(train_data, batch_size=batch_size,
                         sampler=SubsetRandomSampler(valid_idx), num_workers=num_workers)

# 데이터 크기 확인
train_total = len(train_idx)
valid_total = len(valid_idx)

train_batches = len(train_loader)
valid_batches = len(valid_loader)

print(f'total train imgs: {train_total} / total train batches: {train_batches}')
print(f'total valid imgs: {valid_total} / total valid batches: {valid_batches}')

# device 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.is_available()

net = models.efficientnet_b3(pretrained=False)
net.classifier

net.fc = nn.Linear(1000, 10)
net = net.to(device)

criterion = CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=0.001)
epochs = 10


for epoch in range(epochs):
    net.train()

    train_loss = 0
    train_correct = 0
    tqdm_dataset = tqdm(train_loader)
    for x, y in tqdm_dataset:
        x = x.to(device)
        y = y.to(device)
        outputs = net(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(y).sum().item()

        tqdm_dataset.set_postfix({
            'Epoch': epoch + 1,
            'Loss': f'{loss.item():.06f}'
        })

    train_loss = train_loss / train_batches
    train_acc = train_correct / train_total

    net.eval()

    valid_loss = 0
    valid_correct = 0
    tqdm_dataset = tqdm(valid_loader)
    with torch.no_grad():
        for x, y in tqdm_dataset:
            x = x.to(device)
            y = y.to(device)

            outputs = net(x)
            loss = criterion(outputs, y)
            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            valid_correct += predicted.eq(y).sum().item()

            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Loss': f'{loss.item():06f}'
            })
    valid_loss = valid_loss / valid_batches
    valid_acc = valid_correct / valid_total

    print(f'epochs {epoch + 1} train_loss {train_loss} train_acc {train_acc} valid_loss: {valid_loss} valid_acc: {valid_acc}')

# 모델 저장
path = './model.pth'
torch.save(net.state_dict(), path)

# 모델 불러오기
path = './model.pth'
net.load_state_dict(torch.load(path))

# test 데이터 불러옴

from glob import glob
import PIL.Image
import numpy as np

test_images = []

path = './data/'
for filename in sorted(glob(path + 'test/*.jpg')):
    an_img = PIL.Image.open(filename)
    img_array = np.array(an_img)
    test_images.append(img_array)

test_images = np.array(test_images)

class CustomDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.img_list = test_images
        self.img_labels = [0] * 10000

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self.transform(self.img_list[idx]), self.img_labels[idx]

test_set = CustomDataset(transform)
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

import pandas as pd
sample_submission = pd.read_csv('./data/sample_submission.csv')

net.eval()

batch_index = 0

for i, (images, targets) in enumerate(test_loader):
    images = images.to(device)
    outputs = net(images)
    batch_index = i * batch_size
    max_vals, max_indices = torch.max(outputs, 1)
    sample_submission.iloc[batch_index: batch_index + batch_size, 1:] = max_indices.long().cpu().numpy()[:,np.newaxis]

# train 데이터를 불러올 때 ImageFolder 메소드를 이용해 데이터를 불러와서
# 예측된 라벨은 숫자로 되어있음.
# 다시 복원시켜 줌.
labels = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
          5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
sample_submission['target'] = sample_submission['target'].map(labels)
sample_submission.head()
sample_submission.to_csv('submit.csv',index=False)














---------- import codes ----------


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import random
import numpy as np
from tqdm import tqdm


import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision.transforms as transforms


from torch.nn import CrossEntropyLoss
from torchvision.models import efficientnet_b3 as efficientnet_b2
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split









────────────────────────────────────────────────────────────
