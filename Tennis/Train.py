import Enviroment
import numpy

a = Enviroment.Enviroment()
a.reset()
over = False
while not over: 
    action = a.env.action_space.sample()
    observation, reward, terminated, truncated, info = a.play(action)
    over = terminated or truncated
    print(observation)
    print('\n')
    print(reward)