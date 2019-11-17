import torch
import numpy as np
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import models
import datetime
from dataset import MyDataset
from torch.autograd import Variable
import tqdm
data_path = 'dataset/'
Batch_size = 4
EPOCH_NUM = 10

train_data = MyDataset(data_path,334)
train_loader = data.DataLoader(train_data,Batch_size,shuffle=True)
model = models.vgg16(pretrained=True)   #model定義
criterion = nn.CrossEntropyLoss()       #loss定義
optimizer = torch.optim.Adam(model.parameters())
model.train()

print("Train")
st = datetime.datetime.now()
for epoch in range(EPOCH_NUM):
    total_loss = 0
    for i, data in enumerate(train_loader):
        x, t = data
        x, t = Variable(x), Variable(t)
        optimizer.zero_grad()
        y = model(x)
        loss = criterion(y, t)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 1 == 0:
        ed = datetime.datetime.now()
        print("epoch:\t{}\ttotal loss:\t{}\ttime:\t{}".format(epoch+1, total_loss, ed-st))
        st = datetime.datetime.now()

