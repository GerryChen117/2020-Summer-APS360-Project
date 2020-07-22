import torch.nn as nn
class autEncA(nn.Module):
    def __init__(self, name="autEncA"):
        super(autEncA, self).__init__()
        self.name = name
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(3, 5, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(5, 10, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10, 5, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(5, 2, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class autEncB(nn.Module):
    def __init__(self, name="autEncB"):
        super(autEncB, self).__init__()
        self.name = name
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(3, 5, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(5, 10, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10, 5, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(5, 2, 4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class autEncC(nn.Module):
    def __init__(self, name="autEncC"):
        super(autEncC, self).__init__()
        self.name = name
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(3, 5, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(5, 7, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(7, 10, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10, 7, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(7, 5, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(5, 2, 4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class autEncD(nn.Module):  # Ratio of Input Size to code size = 12.8
    def __init__(self, name="autEncD"):
        super(autEncD, self).__init__()
        self.name = name
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(3, 6, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(6, 10, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 15, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(15, 10, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 6, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 2, 4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class autEncE(nn.Module):  # Ratio of inputsize to code size 3.15
    def __init__(self, name="autEncE"):
        super(autEncE, self).__init__()
        self.name = name
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(3, 6, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(6, 10, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10, 6, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 2, 4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x