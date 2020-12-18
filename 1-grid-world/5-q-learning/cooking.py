import numpy as np
import random
import copy
from collections import defaultdict

IN_SINK1 = 2

def state_list_to_tuple(state_list):
    state_tuple = tuple()
    for state in state_list:
        st = tuple()
        for element in state:
            if type(element) == list:
                st += (tuple(element),)
            else:
                st += (element,)
        state_tuple += (st,)
    return state_tuple

def state_tuple_to_list(state_tuple):
    state_list = []
    for state in state_tuple:
        sl = []
        for element in state:
            if type(element) == tuple:
                sl.append(list(element))
            else:
                sl.append(element)
        state_list.append(sl)
    return state_list

class KitchenEnv():
    def __init__(self, ingredient_list=[['apple1', [4, False, False]]], fixed_list=[['sink', [False]]],
                 movable_list=[['plate1', [0, None]]], Nloc=10):

        self.Nloc = Nloc
        # self.object_space = {'ingredient': ingredient_list, 'fixed': fixed_list, 'movable': movable_list}

        # map : str --> int
        obj_list = ingredient_list + fixed_list + movable_list
        self.obj_idx = dict()
        for idx in range(len(obj_list)):
            self.obj_idx[obj_list[idx][0]] = idx

        ingredient_action_space = ['move_to_' + str(i) for i in range(Nloc)]
        ingredient_action_space.append('slice')
        fixed_action_space = ['turn_on', 'turn_off']
        movable_action_space = ['move_to_' + str(i) for i in range(Nloc)]
        # movable_action_space = []

        # action : num --> (class , object, action[int])
        self.akey = dict()
        idx = 0
        for ingredient in ingredient_list:
            for action in range(len(ingredient_action_space)):
                self.akey[idx] = ('ingredient', ingredient[0], action)
                idx += 1
        for fixed in fixed_list:
            for action in range(len(fixed_action_space)):
                self.akey[idx] = ('fixed', fixed[0], action)
                idx += 1
        for movable in movable_list:
            for action in range(len(movable_action_space)):
                self.akey[idx] = ('movable', movable[0], action)
                idx += 1
        self.n_actions = len(self.akey)

        self.ing_skey = {'loc': 0, 'washed': 1, 'sliced': 2}
        self.fix_skey = {'activated': 0}
        self.mov_skey = {'loc': 0, 'on': 1}

        def add_class(class_name, object_list):
            class_object_list = []
            for object in object_list:
                class_object_list.append([class_name] + object)
            return class_object_list

        self.init_state = state_list_to_tuple(add_class("ingredient", ingredient_list) + add_class("fixed", fixed_list) + add_class("movable", movable_list))
        self.state = self.init_state
        self.episode_len = 0

    def reset(self):
        self.state = self.init_state
        self.episode_len = 0
        return self.state

    def step(self, action):
        """
        :param action : (class, object, action[int])
        - ingredient_action_key = ['move_to_0', 'move_to_1', ..., 'slice']
        - fixed_action_key = ['turn_on', 'turn_off']
        - movable_action_key = ['move_to_0', 'move_to_1', ... ]
        """
        next_state = state_tuple_to_list(self.state)
        self.episode_len += 1

        # get next state from current state and selected action
        (act_class, act_obj, act_num) = self.akey[action]
        act_objX = self.obj_idx[act_obj]
        if act_class == 'ingredient':
            if act_num < self.Nloc:
                next_state[act_objX][2][self.ing_skey['loc']] = act_num
            elif act_num == self.Nloc - 1:
                next_state[act_objX][2][self.ing_skey['sliced']] = True
        elif act_class == 'movable':
            next_state[act_objX][2][self.mov_skey['loc']] = act_num
        elif act_class == 'fixed':
            # if 'sink' is 'turn on', activate object and wash any ing which was in sink,
            if act_obj[:4] == 'sink':
                if act_num == 0:
                    next_state[act_objX][2][self.fix_skey['activated']] = True
                    if next_state[self.obj_idx['apple1']][2][self.ing_skey['loc']] == IN_SINK1:
                        next_state[self.obj_idx['apple1']][2][self.ing_skey['washed']] = True
                    # for obj in self.state:
                    #     if obj[0] == "ingredient" and obj[2][0] == IN_SINK1:
                    #         next_state[self.obj_idx[obj[1]]][2][self.ing_skey['washed']] = True
                    #         break
                elif act_num == 1:
                    next_state[act_objX][2][self.fix_skey['activated']] = False

        # get reward from next_state(state, action)
        # --> goal = "wash apple1"
        # reward = -1
        reward = 0
        done = False

        # if next_state[self.obj_idx['apple1']][2][self.ing_skey['loc']] != IN_SINK1:
        #     reward = -5
        #     done = True

        if next_state[self.obj_idx['apple1']][2][self.ing_skey['washed']]:
            reward = 10
            done = True

        # if self.episode_len > 10:
        #     done = True

        next_state = state_list_to_tuple(next_state)
        self.state = next_state

        return next_state, reward, done

    def render(self, episode):
        print(episode, self.state)

# noinspection PyInterpreter
class QLearningAgent:
    def __init__(self, actions):
        # 행동 = [0, 1, 2, 3] 순서대로 상, 하, 좌, 우
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.9
        self.q_table = defaultdict(lambda: [0.0] * len(actions))  ### SxA(WxHxA)

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

def print_qtable(qtable):
    print("=" * 100)
    for key, val in qtable.items():
        print(key, val)

def print_important_qtable(qtable, akey):
    s0 = (('ingredient', 'apple1', (4, False, False)), ('fixed', 'sink', (False,)), ('movable', 'plate1', (0, None)))
    s1 = (('ingredient', 'apple1', (2, False, False)), ('fixed', 'sink', (False,)), ('movable', 'plate1', (0, None)))
    s2 = (('ingredient', 'apple1', (2, True, False)), ('fixed', 'sink', (True,)), ('movable', 'plate1', (0, None)))
    a0 = 2
    a1 = 11 #5 #10
    # print("=" * 100)
    for key, val in qtable.items():
        if (key == s0) or (key == s1) or (key == s2):
            print(key, "|", str(akey[a0])+":", val[a0], str(akey[a1])+":", val[a1])
            # print(key, "|", val)

if __name__ == "__main__":
    env = KitchenEnv()
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    # state = env.reset()                   # ((class, object, (object_states)), ...)
    # action = ('ingredient', 'apple1', 2)  # (class, object, action[int])
    # action = agent.get_action(state_list_to_tuple(state))
    # next_state, reward, done = env.step(action)
    ##
    repeat_num = 10
    train_episode = 50
    test_episode = 10
    epsilon = 0.9
    epsilon_decrease_rate = 0.99
    for num in range(repeat_num):
        # print("*"*100, num, epsilon)
        # agent.set_epsilon(epsilon)
        # epsilon = epsilon * epsilon_decrease_rate
        ########################################
        # train
        ########################################
        for episode in range(train_episode):
            state = env.reset()

            while True:
                # 현재 상태에 대한 행동 선택
                action = agent.get_action(state)
                # 행동을 취한 후 다음 상태, 보상 에피소드의 종료여부를 받아옴
                next_state, reward, done = env.step(action)

                # <s,a,r,s'>로 큐함수를 업데이트
                agent.learn(state, action, reward, next_state)

                # 모든 큐함수를 화면에 표시
                # print(state[:2], "|",  action, env.akey[action], "|", next_state[:2], "|",  reward, "|", done) *********************debug for step
                # print_qtable(agent.q_table)********************debug for learn
                state = next_state
                if done:
                    # if episode % 100 == 0:
                        # print(str(episode) + "/" + str(train_episode) + "="*40)

                    # if episode == train_episode - 1:
                    #     print(next_state, reward, done)
                    break
        # print_important_qtable(agent.q_table, env.akey)

        ########################################
        # test
        ########################################
        # print_important_qtable(agent.q_table, env.akey)
        total_reward = 0
        total_step = 0
        for episode in range(test_episode):
            state = env.reset()
            while True:
                action = agent.get_optimal_action(state_list_to_tuple(state))
                next_state, reward, done = env.step(action)
                state = next_state
                total_reward += reward
                total_step += 1
                if done:
                    break
        print("average reward is ", total_reward/test_episode)
        print("average step is ", total_step/test_episode)

    print_qtable(agent.q_table)