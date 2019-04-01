# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:40:30 2019

@author: sarah.du
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import random
import os

import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, utils
from tensorboardX import SummaryWriter
from torch import randperm

 
#file1 = 'C:\\Users\\sarah.du\\OneDrive - AIMCo\\Desktop\\Test\\GE.csv' # read position
#file2 = 'C:\\Users\\sarah.du\\OneDrive - AIMCo\\Desktop\\Test\\GE_output.csv'
file1 = './GE.csv'
file2 = './GE_output.csv'
 
InputData = pd.read_csv(file1, sep='\t')
TargetData = pd.read_csv(file2, sep='\t')

Combination = pd.merge(InputData, TargetData, how='inner', on='Date')

index = [i for i in range(Combination.shape[0])]
random.shuffle(index)
Combination=Combination.set_index([index]).sort_index()

Combination1=Combination.drop(["Date",],axis=1)

std_Combination = Combination1.std()
mean_Combination = Combination1.mean()

normalized_Combination=(Combination1-Combination1.mean())/Combination1.std()

train_size = int(0.8*len(normalized_Combination))  # 80% training data
test_size = len(normalized_Combination) - train_size


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(9,128)
    self.fc2 = nn.Linear(128, 256)
    self.fc3 = nn.Linear(256, 128)
    self.fc4 = nn.Linear(128, 1)
  
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return x
 
def train(model, optimizer, train_size):
  model.train()
  train= normalized_Combination.iloc[0:train_size,:]

  train_loss = 0

  for i in range(0,len(train)):
    data = torch.tensor(train.iloc[i,0:9].astype(np.float32))
    target = torch.tensor(train.iloc[i,9].astype(np.float32))
    optimizer.zero_grad()
    output = model(data)
    loss = F.mse_loss(output, target)  ##mse_loss = ((input-target)**2).mean()
    train_loss += loss.item()
    loss.backward()
    optimizer.step()
  print('Average train loss:', train_loss / len(train))
savedata=[]


def test(model, test_size):
  model.eval()
  test_loss = 0
  with torch.no_grad():
    test = normalized_Combination.iloc[train_size:train_size+test_size,:]
    

    for i in range(0,len(test)):
      data = torch.tensor(test.iloc[i,0:9].astype(np.float32))
      target = torch.tensor(test.iloc[i,9].astype(np.float32))
      output = model(data)
      test_loss += F.mse_loss(output, target).item()
  test_loss /= test_size

  savedata.append(float(test_loss))
  print('\nTest set: average loss: {:.4f}\n'.format(test_loss))

  return test_loss


model = Net()

#optimizer = optim.Adagrad(model.parameters())
optimizer = optim.SGD(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)


prev_loss = 100000000
for epoch in range(1000):
  train(model, optimizer, train_size)
  test_loss = test(model, test_size)
  scheduler.step()
  if test_loss < prev_loss:
    torch.save(model.state_dict(), './best_random_model.mod')



model.load_state_dict(torch.load('./best_random_model.mod'))
print('best test loss:', test(model, 1000))
#print('Estimate for 64: {:.4f}'.format(model(torch.FloatTensor([64,32])).item()))







