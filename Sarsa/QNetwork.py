import torch.nn

class QNetwork(torch.nn.Module):
    def __init__(self, in_dim,out_dim,device) -> None:
        super(QNetwork,self).__init__()
        layers = [
            torch.nn.Linear(in_dim,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,out_dim),
            ]
        self.model = torch.nn.Sequential(*layers)
        self.onpolicy_reset()
        self.device = device
    
    def onpolicy_reset(self):
        self.rewards = []
        self.QTable = []

    def forward(self,x):
        return self.model(x)