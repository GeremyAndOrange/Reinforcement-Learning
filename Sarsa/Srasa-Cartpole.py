import gym
import torch
import numpy
import QNetwork
import time

def train(QNet, optimizer):
    T, loss = len(QNet.rewards), []
    for i in range(1,T):
        singleLoss = (torch.tensor(QNet.rewards[i-1]) + 0.99 * torch.max(QNet.QTable[i]) - torch.max(QNet.QTable[i-1])) ** 2
        loss.append(singleLoss.clone())
    loss = torch.stack(loss)
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def cartPole(device,epoch,epsilon):
    env = gym.make('CartPole-v1')
    count = 0
    in_dim = env.observation_space.shape[0] + 1             # The dimension of the state space is 4, and plus action dim
    out_dim = 1                                             # For any (state,action) output a QValue

    QNet = QNetwork.QNetwork(in_dim,out_dim,device)
    QNet.to(device)
    optimizer = torch.optim.Adam(QNet.parameters(),lr=0.005)

    for epi in range(epoch):
        state = env.reset()
        unwrappedState = state[0]
        for i in range(200):
            QValue = torch.empty(0).to(device)
            for action in range(env.action_space.n):
                inputTensor = torch.tensor(numpy.append(unwrappedState, action), dtype=torch.float32).to(QNet.device)
                QValue = torch.cat((QValue, QNet.forward(inputTensor)), dim=0).to(device)
            if epsilon > numpy.random.rand():
                action = env.action_space.sample()
            else:
                action = torch.argmax(QValue).item()
            unwrappedState,reward,terminated, truncated, _ = env.step(action)
            QNet.rewards.append(reward)
            QNet.QTable.append(QValue.clone())
            env.render()
            if terminated or truncated:
                break
        loss = train(QNet,optimizer)
        epsilon = max(epsilon * 0.999, 0.01)
        total_reward = sum(QNet.rewards)
        solved = total_reward > 195.0
        QNet.onpolicy_reset()

        if solved:
            count = count + 1
        print(f'Episode {epi}, loss {loss}, total_reward: {total_reward}, solved: {solved}')
    print(count/epoch)

def main():
    device = torch.device("cpu")
    startTime = time.time()
    cartPole(device,10000,0.9)
    endTime = time.time()
    print(endTime-startTime)

if __name__ == '__main__':
    main()