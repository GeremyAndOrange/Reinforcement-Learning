import gym
import numpy
import torch
import PolicyNetwork

gamma = 0.99

def train(policyNet,optimizer):
    T = len(policyNet.rewards)
    rets = numpy.empty(T,dtype=numpy.float32)
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = policyNet.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(policyNet.log_probs)
    loss = - log_probs * rets
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def main():
    env = gym.make('FrozenLake-v1')
    in_dim = env.observation_space.n
    out_dim = env.action_space.n
    policyNet = PolicyNetwork.PolicyNetwork(in_dim,out_dim)
    optimizer = torch.optim.Adam(policyNet.parameters(),lr=0.01)
    for epi in range(200000):
        state = env.reset()
        unwrapped_state = state[0]
        for i in range(100):
            action = policyNet.act(unwrapped_state,in_dim)
            unwrapped_state,reward,terminated, truncated, info = env.step(action)
            # if (terminated or truncated) and (reward == 0):
            #     reward = -50
            # if not (terminated or truncated):
            #     reward = -1
            # if (terminated or truncated) and (reward == 1):
            #     reward = 10000
            policyNet.rewards.append(reward)
            env.render()
            if terminated or truncated:
                break
        loss = train(policyNet,optimizer)
        total_reward = sum(policyNet.rewards)
        solved = (unwrapped_state == 15)
        policyNet.onpolicy_reset()
        print(f'Episode {epi}, loss {loss}, total_reward: {total_reward}, solved: {solved}')

if __name__ == '__main__':
    main()