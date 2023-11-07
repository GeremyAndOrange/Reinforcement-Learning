from torch.distributions import Categorical
import gym
import numpy
import torch

gamma = 0.99

class Pi(torch.nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(Pi,self).__init__()
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
        x = torch.from_numpy(state.astype(numpy.float32))
        pdparam = self.forward(x)
        pd = Categorical(logits=pdparam)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()

def train(pi,optimizer):
    T = len(pi.rewards)
    rets = numpy.empty(T,dtype=numpy.float32)
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs * rets
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def main():
    env = gym.make('CartPole-v0',render_mode='human')
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n
    pi = Pi(in_dim,out_dim)
    optimizer = torch.optim.Adam(pi.parameters(),lr=0.01)
    for epi in range(300):
        state = env.reset()
        unwrapped_state = state[0]
        for i in range(1000):
            action = pi.act(unwrapped_state)
            unwrapped_state,reward,terminated, truncated, info = env.step(action)
            pi.rewards.append(reward)
            env.render()
            if terminated or truncated:
                break
        loss = train(pi,optimizer)
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset()
        print(f'Episode {epi}, loss {loss}, total_reward: {total_reward}, solved: {solved}')

if __name__ == '__main__':
    main()