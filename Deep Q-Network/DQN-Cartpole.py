import gym
import torch
import numpy
import QNetwork
import time

def train(reward, nextQValue, QValue, optimizer):
    loss = (reward + 0.9*nextQValue - QValue)**2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def cartPole(device,epoch,epsilon):
    env = gym.make('CartPole-v1')
    count = 0
    in_dim = env.observation_space.shape[0]                 # The dimension of the state space is 4
    out_dim = env.action_space.n                            # For any action output a QValue in this state

    QNet = QNetwork.QNetwork(in_dim,out_dim,device)
    QNet.to(device)
    optimizer = torch.optim.Adam(QNet.parameters(),lr=0.01)

    for epi in range(epoch):
        state = env.reset()
        unwrappedState = state[0]
        for i in range(200):
            inputTensor = torch.tensor(unwrappedState, dtype=torch.float32).to(QNet.device)
            QValue = QNet.forward(inputTensor)
            if epsilon > numpy.random.rand():
                action = env.action_space.sample()
            else:
                action = torch.argmax(QValue).item()

            unwrappedState,reward,terminated, truncated, _ = env.step(action)
            
            inputTensor = torch.tensor(unwrappedState, dtype=torch.float32).to(QNet.device)
            nextQValue = QNet.forward(inputTensor)
            if epsilon > numpy.random.rand():
                nextAction = env.action_space.sample()
            else:
                nextAction = torch.argmax(nextQValue).item()
            loss = train(sum(QNet.rewards),nextQValue[nextAction],QValue[action],optimizer)
            QNet.rewards.append(reward)
            QNet.loss.append(loss.item())
            env.render()
            if terminated or truncated:
                break
        epsilon = max(epsilon * 0.999, 0.01)
        total_reward = sum(QNet.rewards)
        total_loss = sum(QNet.loss)
        solved = total_reward > 195.0
        QNet.onpolicy_reset()

        if solved:
            count = count + 1
            print(f'Episode {epi}, loss {total_loss}, total_reward: {total_reward}, solved: {solved}')
    print(count/epoch)

def main():
    device = torch.device("cpu")
    startTime = time.time()
    cartPole(device,100000,0.9)
    endTime = time.time()
    print(endTime-startTime)

if __name__ == '__main__':
    main()