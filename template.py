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
from statistics import mean 
 
#####file one is the input data, and file 2 is the output data, please make sure there is "Date" on top of the data column. 
#### Of course we can use something else than time series data, but please make sure that input and target have the matching ID column

#file1 = 'C:\\Users\\sarah.du\\OneDrive - AIMCo\\Desktop\\NN\\Code\\template\\input.csv' 
#file2 = 'C:\\Users\\sarah.du\\OneDrive - AIMCo\\Desktop\\NN\\Code\\template\\target.csv'
 

#InputData = pd.read_csv(file1)
#TargetData = pd.read_csv(file2)


#Combination = pd.merge(InputData, TargetData, how='inner', on='Date')

combination = pd.read_csv('input.csv')
print(combination)

index = [i for i in range(combination.shape[0])]
random.seed(2)
random.shuffle(index)
combination=combination.set_index([index]).sort_index()

#Combination1=Combination.drop(["Date",],axis=1)    #### If it's not "Date", change it to the column name of the matching ID
#Combination1=Combination

#std_Combination = Combination1.std(axis=0)
#mean_Combination = Combination1.mean(axis=0)

#std_Combination = 1
#mean_Combination = 0

#normalized_Combination = (Combination1 - mean_Combination) / std_Combination

#print(normalized_Combination)

train_size = int(0.5*len(combination))  # 80% training data
test_size = len(combination) - train_size



class Net(nn.Module):                       ######### This part defines how many layers and how many nodes. Note the first layer is the input, last layer is output. 
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(1,128)             #### the number of nodes should match the next layer
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 1)
  
  def forward(self, x):                     ##### function match architechture of the nn
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
 
def train(model, optimizer, train_size):
  model.train()
  train = combination.iloc[0:train_size,:]

  train_loss = 0

  for i in range(0,len(train)):
    data = torch.tensor(train.iloc[i,0:1].astype(np.float32))
    target = torch.tensor(train.iloc[i,1].astype(np.float32))
    optimizer.zero_grad()
    output = model(data)
    loss = F.mse_loss(output, target)  ##mse_loss = ((input-target)**2).mean()
    train_loss += loss.item()
    loss.backward()
    optimizer.step()
  print('Average train loss:', train_loss / len(train))
  return train_loss / len(train)


def test(model, test_size):
  model.eval()
  test_loss = 0
  near_count = 0
  with torch.no_grad():
    test = combination.iloc[train_size:train_size+test_size,:]

    for i in range(0,len(test)):
      data = torch.tensor(test.iloc[i,0:1].astype(np.float32))
      target = torch.tensor(test.iloc[i,1].astype(np.float32))
      output = model(data)
      test_loss += F.mse_loss(output, target).item()
      pred_diff = (abs(output-target))
      if pred_diff < 0.1:
        near_count += 1
  test_loss /= test_size
  print('Test set: average loss: {:.4f}'.format(test_loss))
  print('Test near rate:', float(near_count) / test_size)

  return test_loss


model = Net()

#optimizer = optim.Adagrad(model.parameters())
#optimizer = optim.SGD(model.parameters(), lr=0.1)
#scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
optimizer = optim.Adam(model.parameters(), amsgrad=True)

prev_loss = 100000000
for epoch in range(200):
  train_loss = train(model, optimizer, train_size)
  test_loss = test(model, test_size)
  optimizer.step()
  if test_loss < prev_loss:
    torch.save(model.state_dict(), './best_random_model.mod')
  print('Epoch {} done'.format(epoch))  


model.load_state_dict(torch.load('./best_random_model.mod'))
print('best test loss:', test(model, test_size))
print('Estimate for 400: {:.4f}'.format(model(torch.FloatTensor([400])).item()))

