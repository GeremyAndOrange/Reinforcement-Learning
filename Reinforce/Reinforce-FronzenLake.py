import gym
import numpy
import torch
import PolicyNetwork
import time

def train(policyNet,optimizer):
    T = len(policyNet.rewards)
    rets = numpy.empty(T,dtype=numpy.float32)
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = policyNet.rewards[t] + 0.999 * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets).to(policyNet.device)
    log_probs = torch.stack(policyNet.log_probs)
    loss = - log_probs * rets
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def getReward(position):
    return position//4 + position%4

def frozenLake(device,epoch):
    env = gym.make('FrozenLake-v1',is_slippery=False)
    count = 0
    in_dim = 2                          # The dimension of the state space is 2
    out_dim = env.action_space.n        # To get the probability of actions,the output of the neural network is the number of actions
    
    policyNet = PolicyNetwork.PolicyNetwork(in_dim,out_dim,device)
    policyNet.to(device)
    optimizer = torch.optim.Adam(policyNet.parameters(),lr=0.005)

    for epi in range(epoch):
        state = env.reset()
        unwrapped_state = state[0]
        for i in range(20):
            positionX, positionY = unwrapped_state//4, unwrapped_state%4
            action = policyNet.act([positionX,positionY])
            unwrapped_state,reward,terminated, truncated, _ = env.step(action)
            policyNet.rewards.append(getReward(unwrapped_state) + reward*100 - 3)
            env.render()
            if terminated or truncated:
                break
        loss = train(policyNet,optimizer)
        total_reward = sum(policyNet.rewards)
        solved = (unwrapped_state == 15)
        policyNet.onpolicy_reset()

        if solved:
            count = count + 1
            print(f'Episode {epi}, loss {loss}, total_reward: {total_reward}, solved: {solved}')
    print(count/epoch)

def main():
    device = torch.device("cpu")
    startTime = time.time()
    frozenLake(device,10000)
    endTime = time.time()
    print(endTime-startTime)

if __name__ == '__main__':
    main()