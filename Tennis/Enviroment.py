import gym

class Enviroment:
    def __init__(self) -> None:
        self.env = gym.make("ALE/Tennis-v5",render_mode="human")    # 创建环境
        return None
    
    def reset(self):
        state = self.env.reset()                                    # 初始化环境
        return state

    def play(self,action):
        self.env.render()                                           # 渲染环境
        return self.env.step(action)                                # 执行动作并获取反馈
    
    def close(self) -> None:
        self.env.close()                                            # 关闭环境
        return None