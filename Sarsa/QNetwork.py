import torch

class QNetwork(torch.nn.Module):
    def __init__(self, in_dim) -> None:
        super(QNetwork,self).__init__()
        layers = [
            torch.nn.Linear(in_dim,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,1),
            ]
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self,action,state):
        state = torch.tensor([state]).float().view(1, 1)
        action = torch.tensor([action]).float().view(1, 1)
        input = torch.cat((action, state), dim=1)
        return self.model(input)