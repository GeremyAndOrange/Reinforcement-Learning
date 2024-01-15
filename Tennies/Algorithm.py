import torch
import numpy
import NetWork
import Enviroment

def getAction(epsilon,QNet:NetWork.netWork,env,nextState):
    inputTensor = torch.tensor(nextState, dtype=torch.float32).to(QNet.device)
    inputTensor = inputTensor.unsqueeze(0)
    if epsilon > numpy.random.rand():
        thisAction = env.action_space.sample()
    else:
        thisAction = QNet.forward(inputTensor).argmax().item()
    return thisAction

def train(envrioment:Enviroment.Enviroment,QNet,optimizer,lossFunction):
    thisState, thisAction, reward, nextState, over = envrioment.getData(128,'random')
    
    QValue = QNet.forward(thisState)
    QValue = QValue[range(128), thisAction]
    with torch.no_grad():
        nextQValue = QNet.forward(nextState).max(dim=1)[0]
    for i in range(128):
        nextQValue[i] = 0 if over[i] else nextQValue[i]
    loss = lossFunction(reward + 0.9 * nextQValue, QValue)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def playOneGame(env,epsilon,QNet):
    state = env.reset()
    nextState, over, rewards = state[0], False, []
    while not over:
        thisAction = getAction(epsilon,QNet,env,nextState)
        nextState,reward,terminated,truncated,_ = env.step(thisAction)
        reward += 0.1
        QNet.rewards.append(reward)
        env.render()
        if terminated or truncated:
            break
    return sum(rewards)

def RLgame(device,epoch,gameName):
    epsilon = 0.999
    enviroment = Enviroment.Enviroment(gameName)
    in_dim = 4
    out_dim = enviroment.env.action_space.n
    
    QNet = NetWork.netWork(in_dim,out_dim,device)
    QNet.to(device)
    optimizer = torch.optim.Adam(QNet.parameters(),lr=0.001)
    lossFunction = torch.nn.MSELoss()

    for epi in range(epoch):
        epsilon = max(epsilon * 0.9999, 0.01)
        # update dataPool
        enviroment.genData(getAction,200,epsilon,QNet,enviroment.env)

        # off-line training
        for trainNum in range(100):
            loss = train(enviroment,QNet,optimizer,lossFunction)
            QNet.loss.append(loss.item())
        
        # play one game
        playOneGame(enviroment.env,epsilon,QNet)

        # record results
        total_reward = sum(QNet.rewards)
        total_loss = sum(QNet.loss)
        QNet.onpolicy_reset()
        print(f'Episode {epi}, loss {total_loss}, total_reward: {total_reward}')

def main():
    device = torch.device("cpu")
    RLgame(device,100000,'Tennis-v0')

if __name__ == '__main__':
    main()