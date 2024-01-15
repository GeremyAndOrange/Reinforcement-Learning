import gym
import random
import numpy
import torch

class Enviroment():
    def __init__(self,envrioment) -> None:
        self.env = gym.make(envrioment,render_mode = 'rgb_array')
        self.dataPool = []
    
    def clearData(self):
        self.dataPool = []

    def genData(self,actionPolicy,dataNum,epsilon,QNet,env):
        count = 0
        while count < dataNum:
            state = self.env.reset()
            nextState = state[0]
            over = False
            reward = 0
            while not over:
                thisAction = actionPolicy(epsilon,QNet,env,nextState)
                thisState = nextState
                nextState,reward,terminated,truncated,_ = self.env.step(thisAction)
                reward += 0.1
                over = terminated or truncated
                self.dataPool.append((thisState,thisAction,reward,nextState,over,1.0))
                count += 1
        while len(self.dataPool) > 10000:
            self.dataPool.pop(0)
    
    def getData(self,getNum,getDataPolicy):
        '''
        getDataPolicy:{random,preExperience}
        '''
        if getDataPolicy == 'random':
            random.shuffle(self.dataPool)
            trainData = random.sample(self.dataPool, getNum)
        elif getDataPolicy == 'preExperience':
            self.dataPool = sorted(self.dataPool,key=lambda x: x[5])
            trainData = self.dataPool[:getNum]
        thisState = torch.tensor(numpy.array([data[0] for data in trainData]), dtype=torch.float32)
        thisAction = torch.tensor(numpy.array([data[1] for data in trainData]), dtype=torch.int64)
        reward = torch.tensor(numpy.array([data[2] for data in trainData]), dtype=torch.float32)
        nextState = torch.tensor(numpy.array([data[3] for data in trainData]), dtype=torch.float32)
        over = torch.tensor(numpy.array([data[4] for data in trainData]), dtype=torch.int64)
        return thisState, thisAction, reward, nextState, over
