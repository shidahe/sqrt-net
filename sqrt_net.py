import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(1,128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 1)
  
  def forward(self, x):
    x = F.sigmoid(self.fc1(x))
    x = F.sigmoid(self.fc2(x))
    x = self.fc3(x)
    return x


def train(model, optimizer, train_size):
  model.train()
  samples = list(range(1, train_size*2+1, 2))
  random.shuffle(samples)
  for i in samples:
    data = torch.FloatTensor([i])
    target = torch.FloatTensor([math.sqrt(i)])
    optimizer.zero_grad()
    output = model(data)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()

def test(model, test_size):
  model.eval()
  test_loss = 0
  with torch.no_grad():
    samples = list(range(0, test_size*2, 2))
    random.shuffle(samples)
    for i in samples:
      data = torch.FloatTensor([i])
      target = torch.FloatTensor([math.sqrt(i)])
      output = model(data)
      test_loss += F.mse_loss(output, target).item()
  test_loss /= test_size
  print('\nTest set: average loss: {:.4f}\n'.format(test_loss))
  return test_loss


model = Net()
#optimizer = optim.SGD(model.parameters(), lr=1e-10, momentum=0.8)
#optimizer = optim.Adam(model.parameters())
#optimizer = optim.Adadelta(model.parameters())
optimizer = optim.Adagrad(model.parameters())
#optimizer = optim.Adamax(model.parameters())

prev_loss = 100000000
for epoch in range(200):
  train(model, optimizer, 500)
  test_loss = test(model, 500)
  if test_loss < prev_loss:
    torch.save(model.state_dict(), './best_sqrt.mod')

model.load_state_dict(torch.load('./best_sqrt.mod'))
print('best test loss:', test(model, 500))
print('sqrt of 400: {:.4f}'.format(model(torch.FloatTensor([400])).item()))

