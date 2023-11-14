import gym
import torch
import QNetwork

epsilon = 0.9
gamma = 0.9



def train():
    return None

def main():
    env = gym.make('FrozenLake-v1')
    # in_dim = env.action_space.n + env.observation_space.n
    in_dim = 2
    QNet = QNetwork.QNetwork(in_dim)
    optimizer = torch.optim.Adam(QNet.parameters(),lr=0.01)

    for epi in range(10000):
        state = env.reset()
        state = state[0]
        done = False
        while not done:
            action = env.action_space.sample()
            QValue = QNet.forward(action,state)
            action = torch.argmax(QValue).item()
            
            nextState,reward,terminated, truncated, info = env.step(action)
            
            nextAction = env.action_space.sample()
            nextQValue = QNet.forward(nextAction,nextState)
            action = torch.argmax(QValue).item()
            
            targetQValue = reward + gamma*nextQValue[0][action]
            loss = (QValue[0][action] - targetQValue).pow(2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = nextState

if __name__ == '__main__':
    main()