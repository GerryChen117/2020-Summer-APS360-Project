import torch.nn as nn
class semiSupA(nn.Module):
    # Supports autoEncD
    def __init__(self, name, enc):
        super(semiSupA, self).__init__()
        self.name = name
        self.enc  = enc
        self.conv = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(15, 17, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(17, 20, 4, stride=2),
            nn.ReLU(),
        )
        self.fconn = nn.Sequential(
            nn.Linear(20*30*30, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        x = self.enc.encoder(x)
        x = self.conv(x.detach())
        x = x.view(-1, 20*30*30)
        x = self.fconn(x)
        return x

class semiSupB(nn.Module):
    # Supports autoEncD
    def __init__(self, name, enc):
        super(semiSupB, self).__init__()
        self.name = name
        self.enc  = enc
        self.conv = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(15, 30, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(30, 45, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(45, 60, 4, stride=2),
            nn.ReLU(),
        )
        self.fconn = nn.Sequential(
            nn.Linear(60*14*14, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
        )

    def forward(self, x):
        x = self.enc.encoder(x)
        x = self.conv(x.detach())
        x = x.view(-1, 60*14*14)
        x = self.fconn(x)
        return x