import gym
import numpy
import time

def frozenLake(epoch,epsilon,algorithmName):
    env = gym.make('FrozenLake-v1',is_slippery=False)
    count = 0
    Q = [[0.0 for i in range(env.action_space.n)] for i in range(env.observation_space.n)]

    for epi in range(epoch):
        state = env.reset()
        nextState = state[0]
        for i in range(100):
            if epsilon > numpy.random.rand() or Q[nextState].count(Q[nextState][0]) == len(Q[nextState]):
                thisAction = env.action_space.sample()
            else:
                thisAction = Q[nextState].index(max(Q[nextState]))
            thisState = nextState
            nextState,reward,terminated,truncated,_ = env.step(thisAction)
            if epsilon > numpy.random.rand() or Q[nextState].count(Q[nextState][0]) == len(Q[nextState]):
                nextAction = env.action_space.sample()
            else:
                nextAction = Q[nextState].index(max(Q[nextState]))
            if algorithmName == 'Sarsa':
                Q[thisState][thisAction] = Q[thisState][thisAction] + (reward + 0.9*Q[nextState][nextAction] - Q[thisState][thisAction])
            elif algorithmName == 'Q-learning':
                Q[thisState][thisAction] = Q[thisState][thisAction] + (reward + 0.9*max(Q[nextState]) - Q[thisState][thisAction])
            env.render()
            if terminated or truncated:
                break
        solved = (nextState == 15)

        if solved:
            epsilon = max(epsilon * 0.999, 0.01)
            count = count + 1
            print(f'Episode {epi}, total_reward: {reward}, solved: {solved}')
    print(count/epoch)

def main():
    startTime = time.time()
    algorithmName = {0:'Sarsa',1:'Q-learning'}
    frozenLake(1000000,0.9,algorithmName[1])
    endTime = time.time()
    print(endTime-startTime)

if __name__ == '__main__':
    main()