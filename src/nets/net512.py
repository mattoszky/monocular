
import torch.nn as nn

class Net512(nn.Module):
    """
    Neural Net with 512 neurons on first hidden layer and drop to 2 by the power of 2
    5 neurons in input
    2 in output
    """
    
    def __init__(self):
        super(Net512, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(5, 512),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.layer8 = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
        )
        self.output = nn.Linear(4, 2) 

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.output(x)
        return x