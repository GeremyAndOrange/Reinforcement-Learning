import torch.nn

class ACNetWork(torch.nn.Module):
    def __init__(self, in_dim,out_dim,device) -> None:
        super(ACNetWork,self).__init__()
        ActorLayer = [
            torch.nn.Linear(in_dim,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,out_dim),
            torch.nn.Softmax(1),
        ]
        CriticLayer = [
            torch.nn.Linear(in_dim,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,1),
        ]
        self.ActorModel = torch.nn.Sequential(*ActorLayer)
        self.CriticModel = torch.nn.Sequential(*CriticLayer)
        self.device = device
        self.onpolicy_reset()

    def ActorForward(self,x):
        return self.ActorModel(x)
    
    def CriticFarward(self,x):
        return self.CriticModel(x)
    
    def onpolicy_reset(self):
        self.rewards = []
        self.loss = []
        self.TDloss = []