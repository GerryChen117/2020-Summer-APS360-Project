import torch.nn as nn
class semiSupA(nn.Module):
    # Supports autoEncC
    def __init__(self, name):
        super(semiSupA, self).__init__()
        self.name = name
        self.conv = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(15, 20, 3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(20, 25, 4),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(25, 30, 4),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )
        self.fconn = nn.Sequential(
            nn.Linear(30*13*13, 20),
            nn.Sigmoid(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 30*13*13)
        x = self.fconn(x)
        return x