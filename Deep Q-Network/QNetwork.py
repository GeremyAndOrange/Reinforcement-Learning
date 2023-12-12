import torch.nn

class QNetwork(torch.nn.Module):
    def __init__(self, in_dim,out_dim,device) -> None:
        super(QNetwork,self).__init__()
        Layers = [
            torch.nn.Linear(in_dim,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,out_dim),
            ]
        self.model = torch.nn.Sequential(*Layers)
        self.onpolicy_reset()
        self.device = device
    
    def onpolicy_reset(self):
        self.rewards = []
        self.loss = []

    def modelForward(self,x):
        return self.model(x)