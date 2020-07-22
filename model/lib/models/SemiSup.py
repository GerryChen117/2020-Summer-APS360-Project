import torch.nn as nn
import torch.nn.functional as F
class semiSupA(nn.Module):
    # Supports autoEncD
    def __init__(self, enc, name='semiSupA'):
        super(semiSupA, self).__init__()
        self.name = name
        self.enc  = enc

        self.conv1 = nn.Conv2d(10, 17, 4, stride=2)
        self.conv2 = nn.Conv2d(17, 20, 4, stride=2)

        self.fc1   = nn.Linear(30*30*20, 20)
        self.fc2   = nn.Linear(20, 1)

    def forward(self, x):
        x = self.enc.encoder(x).detach()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 30*30*20)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return(x)

class semiSupB(nn.Module):
    # Supports autoEncD
    def __init__(self, enc, name='semiSupB'):
        super(semiSupB, self).__init__()
        self.name = name
        self.enc  = enc

        self.conv1 = nn.Conv2d(10, 30, 4, stride=2)
        self.conv2 = nn.Conv2d(30, 45, 4, stride=2)
        self.conv3 = nn.Conv2d(45, 60, 4, stride=2)

        self.fc1   = nn.Linear(30*30*60, 50)
        self.fc2   = nn.Linear(50, 25)
        self.fc3   = nn.Linear(25, 1)

    def forward(self, x):
        x = self.enc.encoder(x).detach()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 30*30*60)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return(x)

class genSemiA(nn.Module):
    def __init__(self, enc, name="genSemiA"):
        super(genSemiA, self).__init__()
        self.name = name
        self.enc  = enc
        self.softmax = nn.LogSoftmax()

        self.conv1 = nn.Conv2d(2 , 10, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(10, 20, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(20, 30, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(30, 40, 4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(40, 50, 4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(50, 60, 4, stride=2, padding=1)

        self.fc1   = nn.Linear(16*16*60, 40)
        self.fc2   = nn.Linear(40, 20)
        self.fc3   = nn.Linear(20, 1)

    def forward(self, x):
        x = self.softmax(self.enc(x)).detach()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(-1, 16*16*60)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return(x)

class fullGenSemiA(nn.Module):
    def __init__(self, name="fullGenSemiA"):
        super(fullGenSemiA, self).__init__()
        self.name = name

        self.conv1 = nn.Conv2d(3 , 10, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(10, 20, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(20, 30, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(30, 40, 4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(40, 50, 4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(50, 60, 4, stride=2, padding=1)

        self.fc1   = nn.Linear(16*16*60, 40)
        self.fc2   = nn.Linear(40, 20)
        self.fc3   = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(-1, 16*16*60)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return(x)
