import gym
import torch
import numpy
import QNetwork
import matplotlib.pyplot

epsilon = 0.9
gamma = 0.9
alpha = 0.05

def train(predictions, targets, optimizer):
    criterion = torch.nn.MSELoss()
    loss = predictions + criterion(predictions, targets)*alpha
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def getTensor(state,action,state_dim,action_dim):
    stateTensor = torch.as_tensor(state, dtype=torch.int64)
    stateTensor = torch.nn.functional.one_hot(stateTensor, num_classes=state_dim).float()
    actionTensor = torch.as_tensor(action, dtype=torch.int64)
    actionTensor = torch.nn.functional.one_hot(actionTensor, num_classes=action_dim).float()
    inputTensor = torch.cat((stateTensor,actionTensor),dim=0)
    return inputTensor

def getReward(reward):
    calReward = 18 - ((reward // 4) - 3)**2 - ((reward % 4) - 3)**2
    return calReward/10

def plot(lossList):
    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.plot(range(len(lossList)), lossList, marker='o')
    matplotlib.pyplot.title('Loss over time')
    matplotlib.pyplot.xlabel('Index')
    matplotlib.pyplot.ylabel('Loss')
    matplotlib.pyplot.show(block=True)

def main():
    env = gym.make('FrozenLake-v1')
    global epsilon
    count = 0
    finalLossList = []
    in_dim = env.action_space.n + env.observation_space.n   # For a better neural network model, the one-dimensional state space and action space input is passed through one_hot encoding to 16+4 dimensions
    out_dim = 1                                             # For any (state,action) output a QValue
    QNet = QNetwork.QNetwork(in_dim,out_dim)
    optimizer = torch.optim.Adam(QNet.parameters(),lr=0.01)

    for epi in range(10000):
        state = env.reset()
        unwrappedState = state[0]
        action = env.action_space.sample()
        lossList = []
        rewardList = []
        for i in range(100):
            QValue = []
            for action in range(env.action_space.n):
                inputTensor = getTensor(unwrappedState,action,env.observation_space.n,env.action_space.n)
                QValue.append(QNet.forward(inputTensor).item())
            if epsilon > numpy.random.rand():
                action = env.action_space.sample()
            else:
                action = QValue.index(max(QValue))
            unwrappedState,reward,terminated, truncated, info = env.step(action)
            reward = 100 + (100-i) if unwrappedState == 15 else getReward(unwrappedState)
            rewardList.append(reward)
            nextInputTensor = getTensor(unwrappedState,action,env.observation_space.n,env.action_space.n)
            nextQValue = QNet.forward(nextInputTensor)
            targetQValue = reward + gamma * nextQValue
            lossList.append(train(nextQValue,targetQValue,optimizer).item())
            epsilon = max(epsilon * 0.9995, 0.01)
            env.render()
            if terminated or truncated:
                break
        finalLoss = sum(lossList)
        finalReward = sum(rewardList)
        solved = (unwrappedState == 15)
        if solved:
            count = count + 1
            finalLossList.append(finalLoss)
            print(f'Episode {epi}, loss {finalLoss}, total_reward: {finalReward}, solved: {solved}')
    print(count)
    plot(finalLossList)

if __name__ == '__main__':
    main()