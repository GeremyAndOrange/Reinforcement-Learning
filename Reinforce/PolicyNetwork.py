import torch

class PolicyNetwork(torch.nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(PolicyNetwork,self).__init__()
        layers = [
            torch.nn.Linear(in_dim,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,out_dim),
            ]
        self.model = torch.nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []
    
    def forward(self,x):
        pdparam = self.model(x)
        return pdparam

    def act(self,state):
        state = torch.as_tensor(state, dtype=torch.float32)
        pdparam = self.forward(state)
        pd = torch.distributions.Categorical(logits=pdparam)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()