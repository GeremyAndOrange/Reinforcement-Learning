import torch.nn

class QNetwork(torch.nn.Module):
    def __init__(self, in_dim,out_dim,device) -> None:
        super(QNetwork,self).__init__()
        updateLayers = [
            torch.nn.Linear(in_dim,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,out_dim),
            ]
        delayLayers = [
            torch.nn.Linear(in_dim,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,out_dim),
        ]
        self.model = torch.nn.Sequential(*updateLayers)
        self.delayModel = torch.nn.Sequential(*delayLayers)
        self.delayModel.load_state_dict(self.model.state_dict())
        self.onpolicy_reset()
        self.device = device
    
    def onpolicy_reset(self):
        self.rewards = []
        self.loss = []

    def modelForward(self,x):
        return self.model(x)
    
    def delayModelFoward(self,x):
        return self.delayModel(x)