from environment import KitchenEnv
from agent import QLearningAgent

def main():
    # env
    goal1 = ['ingredient', 'apple1', 'washed', True]
    goal2 = ['ingredient', 'apple1', 'sliced', True]
    goal3 = ['movable', 'plate1', 'on', 'apple1']
    goal_list = [goal1, goal2, goal3]
    # goal_list = [goal3]
    env_list = []
    agent_list = []

    ingredient = [['ingredient', 'apple1', [4, False, False]]]
    fixed = [['fixed', 'sink', [False]]]
    movable = [['movable', 'plate1', [0, None]], ['movable', 'bowl1', [1, None]]]
    Nloc = 10

    repeat_num = 10
    train_episode = 50
    for g in range(len(goal_list)):
        print(g)
        env = KitchenEnv(goal_list[g], ingredient, fixed, movable, Nloc)
        agent = QLearningAgent(actions=list(range(env.n_actions)))

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
                    # print(state, env.akey[action])
                    # debug for learn
                    state = next_state

                    if done:
                        break

            # print_selected_qtable(agent.q_table)
            # save_qtable(agent.q_table)

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

            print("[{}/{}]".format(num, repeat_num) + "=" * 50)
            for t in range(len(test_states)):
                print("state={}, action={}".format(test_states[t], test_actions[t]))
            print("test reward: {}, test step: {}".format(test_reward, test_step))

            env_list.append(env)
            agent_list.append(agent)

    total_reward = 0
    total_step = 0
    total_states = []
    total_actions = []

    state = env_list[0].reset()

    for g in range(len(goal_list)):
        env_list[g].state = state

        while True:
            action = agent_list[g].get_optimal_action(state)
            next_state, reward, done = env_list[g].step(action)

            total_states.append(state)
            total_actions.append(env_list[g].akey[action])
            total_reward += reward
            total_step += 1

            state = next_state
            if done:
                break

    print("\n" + "*" * 100)
    for t in range(len(total_states)):
        print("state={}, action={}".format(total_states[t], total_actions[t]))
    print("total reward: {}, total step: {}".format(total_reward, total_step))

if __name__ == "__main__":
    main()