import torch.nn

class netWork(torch.nn.Module):
    def __init__(self,in_dim,out_dim,device) -> None:
        super(netWork,self).__init__()
        ConvLayer = [
            torch.nn.Conv2d(3,32,8,stride=4),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32,32,4,stride=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32,out_dim,3,stride=1),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
        ]

        self.model = torch.nn.Sequential(*ConvLayer)
        self.onpolicy_reset()
        self.device = device
    
    def forward(self,x):
        x = x.permute(0, 3, 1, 2)
        x = self.model(x)
        return x

    def onpolicy_reset(self):
        self.rewards = []
        self.loss = []