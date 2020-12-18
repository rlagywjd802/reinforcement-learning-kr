import pickle
from environment import KitchenEnv
from agent import QLearningAgent

def print_qtable(qtable):
    print("=" * 100)
    for key, val in qtable.items():
        print(key, val)

def print_selected_qtable(qtable, akey, selected_state, action_list):
    # s0 = (('ingredient', 'apple1', (4, False, False)), ('fixed', 'sink', (False,)), ('movable', 'plate1', (0, None)))
    # s1 = (('ingredient', 'apple1', (2, False, False)), ('fixed', 'sink', (False,)), ('movable', 'plate1', (0, None)))
    # s2 = (('ingredient', 'apple1', (2, True, False)), ('fixed', 'sink', (True,)), ('movable', 'plate1', (0, None)))
    # a0 = 2
    # a1 = 11 #5 #10
    # print("=" * 100)
    # for key, val in qtable.items():
    #     if key in selected_state:
    #         print(key, "|", str(akey[a0])+":", val[a0], str(akey[a1])+":", val[a1]) # **********
    pass

def save_qtable(file_path, qtable):
    with open(file_path, 'wb') as handle:
        pickle.dump(dict(qtable), handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved : len=", len(qtable))

def load_qtable(file_path):
    with open(file_path, 'rb') as handle:
        qtable = pickle.load(handle)
    print("loaded : len=", len(qtable), ', type=', type(qtable))
    return qtable

def main():
    # env
    goal = ['ingredient', 'apple1', 'washed', True]
    ingredient = [['ingredient', 'apple1', [4, False, False]]]
    fixed = [['fixed', 'sink', [False]]]
    movable = [['movable', 'plate1', [0, None]], ['movable', 'bowl1', [1, None]]]
    Nloc = 10

    # selected_state =

    env = KitchenEnv(goal, ingredient, fixed, movable, Nloc)
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    repeat_num = 10
    train_episode = 50

    for num in range(repeat_num):
        ########################################
        # train
        ########################################
        for episode in range(train_episode):
            state = env.reset()

            while True:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state)
                # debug for step
                # debug for learn
                state = next_state

                if done:
                    break

        # print_selected_qtable(agent.q_table)

        ########################################
        # test
        ########################################
        test_reward = 0
        test_step = 0
        test_states = []
        test_actions = []

        state = env.reset()
        while True:
            action = agent.get_optimal_action(state)
            next_state, reward, done = env.step(action)

            test_states.append(state)
            test_actions.append(env.akey[action])
            test_reward += reward
            test_step += 1

            state = next_state
            if done:
                break

        print("[{}/{}]".format(num, repeat_num) + "="*50)
        for t in range(len(test_states)):
            print("state={}, action={}".format(test_states[t], test_actions[t]))
        print("test reward: {}, test step: {}".format(test_reward, test_step))

    save_qtable("data/wash_apple1_q.pickle", agent.q_table)
    load_qtable("data/wash_apple1_q.pickle")

if __name__ == "__main__":
    main()
    # save_qtable()