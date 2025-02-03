
import torch
import torch.nn as nn

# codice della rete
# 5 input e 2 output
class Rete(nn.Module):
    def __init__(self):
        super(Rete, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            #nn.Tanh(),
        )
        """
        self.layer5 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            #nn.Tanh(),
        )
        """
        self.layer6 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            #nn.Tanh(),
        )
        self.layer7 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            #nn.Tanh(),
        )
        self.layer8 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            #nn.Tanh(),
        )
        self.layer9 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            #nn.Tanh(),
        )
        self.layer10 = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            #nn.Tanh(),
        )
        self.layer11 = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            #nn.Tanh(),
        )
        self.output = nn.Linear(4, 2) 

    def forward(self, x):
        x = self.layer1(x)
        #x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.output(x)  # Nessuna funzione di attivazione sull'output per la regressione
        return x