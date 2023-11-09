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

    def act(self,state,in_dim):
        state = torch.as_tensor(state, dtype=torch.float32)
        if state.dim() == 0 and in_dim is not None:
            state = state.to(torch.int64)
            state = torch.nn.functional.one_hot(state, num_classes=in_dim).float()
        pdparam = self.forward(state)
        pd = torch.distributions.Categorical(logits=pdparam)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()