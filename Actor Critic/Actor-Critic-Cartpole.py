import ACNetWork
import random
import gym
import numpy
import torch
import time


def getAction(ACnet,state):
    state = torch.tensor(state, dtype=torch.float32).reshape(1,4).to(ACnet.device)
    prob = ACnet.ActorModel(state)
    action = random.choices(range(2), weights=prob[0].tolist(), k=1)[0]
    return action

def getData(ACnet,env):
    dataPool = []

    state = env.reset()
    thisState = state[0]
    over = False
    while not over:
        thisAction = getAction(ACnet,thisState)
        nextState,reward,terminated,truncated,_ = env.step(thisAction)
        over = terminated or truncated
        dataPool.append((thisState,reward,thisAction,nextState,over))
        thisState = nextState

    thisState = torch.tensor(numpy.array([data[0] for data in dataPool]), dtype=torch.float32).reshape(-1,4).to(ACnet.device)
    thisAction = torch.tensor(numpy.array([data[1] for data in dataPool]), dtype=torch.int64).reshape(-1,1).to(ACnet.device)
    reward = torch.tensor(numpy.array([data[2] for data in dataPool]), dtype=torch.float32).reshape(-1,1).to(ACnet.device)
    nextState = torch.tensor(numpy.array([data[3] for data in dataPool]), dtype=torch.float32).reshape(-1,4).to(ACnet.device)
    over = torch.tensor(numpy.array([data[4] for data in dataPool]), dtype=torch.int64).reshape(-1,1).to(ACnet.device)
    return thisState, thisAction, reward, nextState, over

def cartPole(device,epoch):
    env = gym.make('CartPole-v1')
    count = 0
    in_dim = env.observation_space.shape[0]                 # The dimension of the state space is 4
    out_dim = env.action_space.n                            # For any action output a QValue in this state

    ACnet = ACNetWork.ACNetWork(in_dim,out_dim,device)
    ACnet.to(device)
    ActorOptimizer = torch.optim.Adam(ACnet.ActorModel.parameters(),lr=0.001)
    CriticOptimizer = torch.optim.Adam(ACnet.CriticModel.parameters(),lr=0.001)
    lossFunction = torch.nn.MSELoss()

    for epi in range(epoch):
        thisState, thisAction, reward, nextState, over = getData(ACnet,env)
        values = ACnet.CriticFarward(thisState)
        targets = ACnet.CriticFarward(nextState) * 0.9 * (1 - over) + reward

        delta = (targets - values).detach()
        probs = ACnet.ActorForward(thisState)
        probs = probs.gather(dim=1, index=thisAction)
        loss = (-probs.log() * delta).mean()
        TDloss = lossFunction(values, targets.detach())

        ACnet.loss.append(loss.item())
        ACnet.TDloss.append(TDloss.item())

        ActorOptimizer.zero_grad()
        loss.backward()
        ActorOptimizer.step()

        CriticOptimizer.zero_grad()
        TDloss.backward()
        CriticOptimizer.step()

        # play one game
        state = env.reset()
        nextState = state[0]
        for i in range(200):
            thisAction = getAction(ACnet,nextState)
            nextState,reward,terminated,truncated,_ = env.step(thisAction)
            ACnet.rewards.append(reward)
            env.render()
            if terminated or truncated:
                break

        # record results
        total_reward = sum(ACnet.rewards)
        total_loss = sum(ACnet.loss)
        total_TDloss = sum(ACnet.TDloss)
        solved = total_reward > 195.0
        ACnet.onpolicy_reset()
        if solved:
            count = count + 1
        print(f'Episode {epi}, loss {total_loss}, TDloss {total_TDloss}, total_reward: {total_reward}, solved: {solved}')
    print(count/epoch)

def main():
    device = torch.device("cpu")
    startTime = time.time()
    cartPole(device,100000)
    endTime = time.time()
    print(endTime-startTime)

if __name__ == '__main__':
    main()