import torch

class QNetwork(torch.nn.Module):
    def __init__(self, in_dim,out_dim) -> None:
        super(QNetwork,self).__init__()
        layers = [
            torch.nn.Linear(in_dim,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,out_dim),
            ]
        self.model = torch.nn.Sequential(*layers)
        self.train()
    
    def forward(self,x):
        return self.model(x)