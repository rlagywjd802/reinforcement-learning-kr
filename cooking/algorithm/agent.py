import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions, q_table=None):
        # 행동 = [0, 1, 2, 3] 순서대로 상, 하, 좌, 우
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.9
        if q_table is None:
            self.q_table = defaultdict(lambda: [0.0] * len(actions))  ### SxA(WxHxA)
        else:
            self.q_table = self.add_object(q_table, len(actions))

    def add_object(self, qtable, n_actions):
        # q_added = defaultdict(lambda: [0.0] * len(n_actions))
        apple1_action_len = 11
        q_added = dict()
        for state, qval in qtable.items():
            # s5 == s1
            state += ('ingredient', 'apple2', state[0][2])
            qval.append(qval[:apple1_action_len])
            # q_added[state] = apple1


    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    # <s, a, r, s'> 샘플로부터 큐함수 업데이트
    def learn(self, state, action, reward, next_state):
        q_1 = self.q_table[state][action]
        # 벨만 최적 방정식을 사용한 큐함수의 업데이트
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])  ###
        self.q_table[state][action] += self.learning_rate * (q_2 - q_1)

    # 큐함수에 의거하여 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)  ####
        else:
            # 큐함수에 따른 행동 반환
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    def get_optimal_action(self, state):
        # 큐함수에 따른 행동 반환
        state_action = self.q_table[state]
        action = self.arg_max(state_action)
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)