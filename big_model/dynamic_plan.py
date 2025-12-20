import numpy as np
import sys

# 定义动作空间
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# 定义宝藏区
DONE_LOCATION = 8


# 定义网格世界 环境模型
class GridWorldEnv():
    def __init__(self, shape=[5, 5]):

        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')
        self.shape = shape
        self.nS = np.prod(shape)  # 状态个数
        self.nA = 4  # 动作个数
        MAX_Y = shape[0]
        MAX_X = shape[1]
        P = {}
        grid = np.arange(self.nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            P[s] = {a: [] for a in range(self.nA)}
            # s得到的是一个索引值，P[s][a]记录的是对于每一个格子采用四种走法，即上下左右之后产生的回报、下一个状态、是否达到宝藏区的信息

            is_done = lambda s: s == DONE_LOCATION
            # 定义奖励函数 除了宝藏区 都是-1 宝藏区奖励为0
            reward = 0.0 if is_done(s) else -1.0

            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]
            it.iternext()
        self.isd = np.ones(self.nS) / self.nS
        self.P = P
        # super(GridWorldEnv,self).__init__(self.nS, self.nA, P, isd)


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    策略评估函数，计算给定策略下的状态值函数
    :param policy: 策略函数，输入状态，输出动作概率分布
    :param env: 环境模型，包含状态转移概率和奖励函数
    :param discount_factor: 折扣因子，用于计算未来奖励的折扣
    :param theta: 迭代停止阈值，当状态值函数的变化小于该值时停止迭代
    :return: 状态值函数
    """
    # 初始化状态值函数为0
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # 遍历所有状态
        for s in range(env.nS):
            v = 0
            # 遍历当前状态下的所有动作
            for a, action_prob in enumerate(policy[s]):
                # 遍历当前动作下的所有可能状态转移
                for prob, next_state, reward, done in env.P[s][a]:
                    # 计算状态值函数的更新值
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # 计算状态值函数的最大变化
            delta = max(delta, np.abs(v - V[s]))
            # 更新状态值函数
            V[s] = v
        # 如果状态值函数的变化小于阈值，停止迭代
        if delta < theta:
            break
    return V


def get_max_index(action_values):
    """
    根据传入的四个行为，获取值函数中最大的索引，返回的是一个索引数组何一个行为策略
    :param action_values: 动作值函数，包含每个动作的价值
    :return: 最大价值的索引
    """
    indexes = []
    policy_arr = np.zeros(len(action_values))
    max_action_value = np.max(action_values)

    for i in range(len(action_values)):
        action_value = action_values[i]

        if action_value == max_action_value:
            indexes.append(i)
            policy_arr[i] = 1.0
    return indexes, policy_arr


def change_policy(policys):
    action_tuple = []
    for policy in policys:
        action_tuple.append(tuple(get_max_index(policy)))
    return action_tuple


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    改变策略函数，根据当前的状态值函数，计算每个状态下的动作值函数，然后根据动作值函数选择最优的动作
    :param env: 环境模型，包含状态转移概率和奖励函数
    :param policy_eval_fn: 策略评估函数，用于计算状态值函数
    :param discount_factor: 折扣因子，用于计算未来奖励的折扣
    :return: 新的策略函数
    """
    # 初始化随机策略
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        # 评估当前策略
        V = policy_eval_fn(policy, env, discount_factor)
        # 标志位，用于判断是否改进
        policy_stable = True
        # 遍历所有状态
        for s in range(env.nS):
            # 选择当前状态下的动作
            chosen_a = np.argmax(policy[s])
            # 计算当前状态下的动作值函数
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            # 获取最大价值的索引
            best_a, best_policy = get_max_index(action_values)
            # 如果当前策略不是最优策略，更新策略
            if chosen_a not in best_a:
                policy_stable = False
            policy[s] = best_policy
        # 如果策略不再改进，停止迭代
        if policy_stable:
            break
    return policy


if __name__ == '__main__':
    env = GridWorldEnv()
    policy = policy_improvement(env)
    print(policy)
