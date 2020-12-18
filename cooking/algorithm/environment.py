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
    def __init__(self, goal_list, ingredient_list, fixed_list, movable_list, Nloc=10):

        self.goal = goal_list
        self.Nloc = Nloc

        # self.obj_idx : str --> int
        obj_list = ingredient_list + fixed_list + movable_list
        self.obj_idx = dict()
        for idx in range(len(obj_list)):
            self.obj_idx[obj_list[idx][1]] = idx

        # self.skey : str --> int
        self.ing_skey = {'loc': 0, 'washed': 1, 'sliced': 2}
        self.fix_skey = {'activated': 0}
        self.mov_skey = {'loc': 0, 'on': 1}

        # self.akey : num --> (class , object, action(int))
        self.akey = list()
        ingredient_action_space = ['move_to_' + str(i) for i in range(Nloc)]
        ingredient_action_space.append('slice')
        fixed_action_space = ['turn_on', 'turn_off']
        movable_action_space = ['move_to_' + str(i) for i in range(Nloc)]
        for ingredient in ingredient_list:
            for action in range(len(ingredient_action_space)):
                self.akey.append((ingredient[0], ingredient[1], action))
        for fixed in fixed_list:
            for action in range(len(fixed_action_space)):
                self.akey.append((fixed[0], fixed[1], action))
        for movable in movable_list:
            for action in range(len(movable_action_space)):
                self.akey.append((movable[0], movable[1], action))
        self.n_actions = len(self.akey)

        # initialize
        self.init_state = state_list_to_tuple(obj_list)
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
        # 옮길 수 있는 class는 ingredient와 movable
        # 만약 옮기려는 위치에 ingredient가 이미 있으면 그 위치로 옮길 수 없음
        # 만약 movable이 존재하면, 그 위치로 옮기고 밑에 있는 obj의 on에 옮길 물체의 idx/str를 넣는
        # 자기 자리에 그대로 움직여도 -
        # 현재 같은 자리에 있는 물체가 있고 이를 욺직이게 된다면 on에서 제거
        (act_class, act_obj, act_num) = self.akey[action]
        act_objX = self.obj_idx[act_obj]
        wrong_action = False
        obj_moved = False

        if act_class == 'ingredient':
            if act_num < self.Nloc:
                for obj in self.state:
                    if obj[0] == "ingredient":
                        if obj[2][self.ing_skey['loc']] == act_num:
                            wrong_action = True
                            break
                    elif obj[0] == "movable":
                        if (obj[2][self.mov_skey['loc']] == act_num) and (obj[2][self.mov_skey['on']] is None):
                            next_state[act_objX][2][self.ing_skey['loc']] = act_num
                            next_state[self.obj_idx[obj[1]]][2][self.mov_skey['on']] = act_obj
                            obj_moved = True
                            break
                if (not wrong_action) and (not obj_moved):
                    next_state[act_objX][2][self.ing_skey['loc']] = act_num
            else:
                next_state[act_objX][2][self.ing_skey['sliced']] = True

        elif act_class == 'movable':
            below_act_objX = None
            for obj in self.state:
                if obj[0] == "ingredient":
                    if obj[2][self.ing_skey['loc']] == act_num:
                        wrong_action = True
                        break
                elif obj[0] == "movable":
                    if (obj[2][self.mov_skey['loc']] == act_num) and (obj[2][self.mov_skey['on']] is None):
                        if obj[1] == act_obj:
                            wrong_action = True
                            break
                        else:
                            next_state[act_objX][2][self.mov_skey['loc']] = act_num
                            next_state[self.obj_idx[obj[1]]][2][self.mov_skey['on']] = act_obj
                            obj_moved = True
                            # break
                    if obj[2][self.mov_skey['on']] == act_obj:
                        below_act_objX = self.obj_idx[obj[1]]

            if (not wrong_action) and (not obj_moved):
                next_state[act_objX][2][self.mov_skey['loc']] = act_num
                obj_moved = True

            if (below_act_objX is not None) and obj_moved:
                next_state[below_act_objX][2][self.mov_skey['on']] = None

        elif act_class == 'fixed':
            # if 'sink' is 'turn on', activate object and wash any ing which was in sink,
            if act_obj[:4] == 'sink':
                if act_num == 0:
                    next_state[act_objX][2][self.fix_skey['activated']] = True
                    for obj in self.state:
                        if obj[0] == "ingredient" and obj[2][self.ing_skey['loc']] == IN_SINK1:
                            next_state[self.obj_idx[obj[1]]][2][self.ing_skey['washed']] = True
                            break
                elif act_num == 1:
                    next_state[act_objX][2][self.fix_skey['activated']] = False

        # get reward based on goal
        goal = self.goal
        reward = 0
        done = False
        skey = None
        if goal[0] == 'ingredient':
            skey = self.ing_skey
        elif goal[0] == 'fixed':
            skey = self.fix_skey
        elif goal[0] == 'movable':
            skey = self.mov_skey

        if wrong_action:
            reward -= 2

        if next_state[self.obj_idx[goal[1]]][2][skey[goal[2]]] == goal[3]:
            reward += 10
            done = True

        # update state
        next_state = state_list_to_tuple(next_state)
        self.state = next_state

        return next_state, reward, done

    def render(self):
        print(self.state)