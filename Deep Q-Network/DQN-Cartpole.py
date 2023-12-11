import gym
import torch
import numpy
import QNetwork
import time
import random

def saveData(thisState,thisAction,reward,nextState,dataPool):
    if len(dataPool) == 10000:
        dataPool.pop(0)
    dataPool.append((thisState,thisAction,reward,nextState))

def loadData(dataPool,QNet):
    if len(dataPool) >= 200:
        trainData = random.sample(dataPool, 200)
    else:
        trainData = random.sample(dataPool, len(dataPool))
    thisState = torch.tensor([data[0] for data in trainData], dtype=torch.float32).reshape(-1,4).to(QNet.device)
    thisAction = torch.tensor([data[1] for data in trainData], dtype=torch.int64).reshape(-1,1).to(QNet.device)
    reward = torch.tensor([data[2] for data in trainData], dtype=torch.float32).reshape(-1,1).to(QNet.device)
    nextState = torch.tensor([data[3] for data in trainData], dtype=torch.float32).reshape(-1,4).to(QNet.device)
    return thisState,thisAction,reward,nextState

def train(dataPool,QNet,optimizer):
    for i in range(200):
        thisState,thisAction,reward,nextState = loadData(dataPool,QNet)
        QValue = QNet.modelForward(thisState)
        QValue = QValue.gather(dim=1, index=thisAction)
        with torch.no_grad():
            nextQValue = QNet.delayModelFoward(nextState)
            nextQValue = nextQValue.max(dim=1)[0]
            nextQValue = nextQValue.reshape(-1,1)
        loss = (reward + 0.98*nextQValue - QValue)**2
        loss = torch.sum(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        QNet.loss.append(loss)
        if i % 10 == 0:
            QNet.delayModel.load_state_dict(QNet.model.state_dict())

def cartPole(device,epoch,epsilon):
    env = gym.make('CartPole-v1')
    count, dataPool = 0, []
    in_dim = env.observation_space.shape[0]                 # The dimension of the state space is 4
    out_dim = env.action_space.n                            # For any action output a QValue in this state

    QNet = QNetwork.QNetwork(in_dim,out_dim,device)
    QNet.to(device)
    optimizer = torch.optim.Adam(QNet.model.parameters(),lr=0.005)

    for epi in range(epoch):
        state = env.reset()
        nextState = state[0]
        for i in range(200):
            inputTensor = torch.tensor(nextState, dtype=torch.float32).to(QNet.device)
            if epsilon > numpy.random.rand():
                thisAction = env.action_space.sample()
            else:
                thisAction = QNet.modelForward(inputTensor).argmax().item()
            thisState = nextState
            nextState,reward,terminated,truncated,_ = env.step(thisAction)
            saveData(thisState,thisAction,reward,nextState,dataPool)
            QNet.rewards.append(reward)
            env.render()
            if terminated or truncated:
                break
        train(dataPool,QNet,optimizer)
        epsilon = max(epsilon * 0.99, 0.01)
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
    cartPole(device,10000,0.9)
    endTime = time.time()
    print(endTime-startTime)

if __name__ == '__main__':
    main()