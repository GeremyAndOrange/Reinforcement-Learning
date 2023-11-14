import gym
import numpy
import torch
import PolicyNetwork
import matplotlib.pyplot

gamma = 0.999

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

def plot(lossList):
    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.plot(range(len(lossList)), lossList, marker='o')
    matplotlib.pyplot.title('Loss over time')
    matplotlib.pyplot.xlabel('Index')
    matplotlib.pyplot.ylabel('Loss')
    matplotlib.pyplot.show(block=True)


def main():
    env = gym.make('FrozenLake-v1')
    count = 0
    lossList = []
    in_dim = env.observation_space.n    # For a better neural network model, the one-dimensional state space input is passed through one_hot encoding to 16 dimensions
    out_dim = env.action_space.n        # To get the probability of actions,the output of the neural network is the number of actions
    policyNet = PolicyNetwork.PolicyNetwork(in_dim,out_dim)
    optimizer = torch.optim.Adam(policyNet.parameters(),lr=0.01)
    for epi in range(30000):
        state = env.reset()
        unwrapped_state = state[0]
        for i in range(100):
            unwrapped_state = torch.as_tensor(unwrapped_state, dtype=torch.int64)
            unwrapped_state = torch.nn.functional.one_hot(unwrapped_state, num_classes=in_dim).float()
            action = policyNet.act(unwrapped_state)
            unwrapped_state,reward,terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if reward == 0:
                    reward = -100
                elif reward == 1:
                    reward = 500 + (100-i)*5
            else:
                if unwrapped_state in [11,14]:
                    reward = 5
                elif unwrapped_state in [7,10,13]:
                    reward = 4
                elif unwrapped_state in [3,6,9,12]:
                    reward = 3
                elif unwrapped_state in [2,5,8]:
                    reward = 2
                elif unwrapped_state in [1,4]:
                    reward = 1
            policyNet.rewards.append(reward)
            env.render()
            if terminated or truncated:
                break
        loss = train(policyNet,optimizer)
        total_reward = sum(policyNet.rewards)
        solved = (unwrapped_state == 15)
        policyNet.onpolicy_reset()
        if solved:
            count = count + 1
            lossList.append(loss.item())
            print(f'Episode {epi}, loss {loss}, total_reward: {total_reward}, solved: {solved}')
    print(count)
    plot(lossList)

if __name__ == '__main__':
    main()