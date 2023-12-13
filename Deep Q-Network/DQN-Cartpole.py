import gym
import torch
import numpy
import QNetwork
import time
import random

def train(dataPool,QNet,optimizer,lossFunction):
    trainData = random.sample(dataPool, 128)
    thisState = torch.tensor(numpy.array([data[0] for data in trainData]), dtype=torch.float32).to(QNet.device)
    thisAction = torch.tensor(numpy.array([data[1] for data in trainData]), dtype=torch.int64).to(QNet.device)
    reward = torch.tensor(numpy.array([data[2] for data in trainData]), dtype=torch.float32).to(QNet.device)
    nextState = torch.tensor(numpy.array([data[3] for data in trainData]), dtype=torch.float32).to(QNet.device)
    over = torch.tensor(numpy.array([data[4] for data in trainData]), dtype=torch.int64).to(QNet.device)

    QValue = QNet.modelForward(thisState)
    QValue = QValue[range(128), thisAction]
    with torch.no_grad():
        nextQValue = QNet.modelForward(nextState).max(dim=1)[0]
    for i in range(128):
        nextQValue[i] = 0 if over[i] else nextQValue[i]
    loss = lossFunction(reward + 0.9 * nextQValue, QValue)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def cartPole(device,epoch,epsilon):
    env = gym.make('CartPole-v1')
    count, dataPool = 0, []
    in_dim = env.observation_space.shape[0]                 # The dimension of the state space is 4
    out_dim = env.action_space.n                            # For any action output a QValue in this state

    QNet = QNetwork.QNetwork(in_dim,out_dim,device)
    QNet.to(device)
    optimizer = torch.optim.Adam(QNet.parameters(),lr=0.001)
    lossFunction = torch.nn.MSELoss()

    for epi in range(epoch):
        epsilon = max(epsilon * 0.999, 0.01)
        # update dataPool
        dataNum = 0
        while dataNum < 200:
            state = env.reset()
            nextState = state[0]
            over = False
            while not over:
                inputTensor = torch.tensor(nextState, dtype=torch.float32).to(QNet.device)
                if epsilon > numpy.random.rand():
                    thisAction = env.action_space.sample()
                else:
                    thisAction = QNet.modelForward(inputTensor).argmax().item()
                thisState = nextState
                nextState,reward,terminated,truncated,_ = env.step(thisAction)
                over = terminated or truncated
                dataPool.append((thisState,thisAction,reward,nextState,over))
                dataNum += 1
        while len(dataPool) > 10000:
            dataPool.pop(0)

        # off-line training
        for trainNum in range(100):
            loss = train(dataPool,QNet,optimizer,lossFunction)
            QNet.loss.append(loss.item())
        
        # play one game
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
            QNet.rewards.append(reward)
            env.render()
            if terminated or truncated:
                break

        # record results
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
    cartPole(device,200000,0.9)
    endTime = time.time()
    print(endTime-startTime)

if __name__ == '__main__':
    main()