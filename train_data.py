import random
import agent

radius_max = 1
radius_min = 0.1
angle_max = 1
angle_min = 0.1
level_max = 1
level_min = 0


class TrainData:
    def __init__(self, generated_data_number, PF_add_threshold, train_mode):

        self.generated_data_number = generated_data_number
        self.train_mode = train_mode
        self.PF_add_threshold = PF_add_threshold
        self.data_set = []
        self.false_data_set = []
        self.level_pool = []
        self.angle_pool = []
        self.radius_pool = []
        self.load_count = 0

    def train_configuration(self, generated_data_number, PF_add_threshold,  train_mode):

        # the number of data we want to generate for estimating
        self.generated_data_number = generated_data_number
        self.train_mode = train_mode
        self.PF_add_threshold = PF_add_threshold



    ####################################################################################################################
    # =================Generating  D = (p,f(p)) , f(p) = P(a|H_t_1,teta,p)==============================================

    def generate_data_for_update_parameter(self,previous_state, previous_agent,  action, selected_type):

        # D= (p,f(p)) , f(p) = P(a|H_t_1,teta,p)

        for i in range(0, self.generated_data_number):

            # Generating random values for parameters

            tmp_radius = radius_min + (1.0 * (radius_max - radius_min) / self.generated_data_number) * i
            tmp_angle = angle_min + (1.0 * (angle_max - angle_min) / self.generated_data_number) * i
            tmp_level = level_min + (1.0 * (level_max - level_min) / self.generated_data_number) * i

            previous_agent.agent_type = selected_type
            previous_agent.set_parameters(previous_state, tmp_level, tmp_radius, tmp_angle)

            previous_agent = previous_state.move_a_agent(previous_agent, True)  # f(p)
            p_action = previous_agent.get_action_probability(action)

            if p_action is not None:
                self.data_set.append([tmp_level, tmp_radius, tmp_angle, p_action])
    # ##################################################################################################################

    @staticmethod
    def compare_actions(ds_actions, actions_to_reach_target):

        same_actions = []
        false_actions = []
        j = 0
        for i in range(len(actions_to_reach_target)):
            if len(ds_actions) < j + 1:
                return False
            if ds_actions[j] == actions_to_reach_target[i]:
                same_actions.append(actions_to_reach_target[i])
                j += 1
            else:
                false_actions.append(actions_to_reach_target[i])
        if (2*len(actions_to_reach_target))/3 > len(false_actions):
            return True
        else :
            return False

    # ##################################################################################################################
    def update_data_set(self, choose_target_state, actions_to_reach_target):
        cts_agent = choose_target_state.agents[0]
        remove_pf = []
        # print 'actions_to_reach_target',actions_to_reach_target
        self.load_count += 1
        seq = [x['succeeded_steps'] for x in self.data_set]

        if seq != []:
            max_succeeded_steps = max(seq)
        else:
            max_succeeded_steps = 0

        # print 'max succeeded steps:', max_succeeded_steps
        # print '**** beforeee ********'
        for ds in self.data_set:
            # print ds
            tmp_agent = agent.Agent(cts_agent.position[0], cts_agent.position[1], cts_agent.direction,
                                    cts_agent.agent_type, -1)

            [tmp_level, tmp_radius, tmp_angle] = ds['parameter']

            tmp_agent.set_parameters(choose_target_state, tmp_level, tmp_radius, tmp_angle)
            if self.compare_actions(ds['route'], actions_to_reach_target) and \
                    ds['succeeded_steps'] == max_succeeded_steps:

                self.level_pool.append(tmp_level)
                self.angle_pool.append(tmp_angle)
                self.radius_pool.append(tmp_radius)
                tmp_agent = choose_target_state.move_a_agent(tmp_agent, True)  # f(p)
                target = tmp_agent.get_memory()

                if tmp_agent.route_actions is not None:

                    ds['route'] = tmp_agent.route_actions
                    ds['target'] = target
                    # ds['reward'] = 0
                    ds['succeeded_steps'] += 1

                else:
                    remove_pf.append(ds)
            else:
                remove_pf.append(ds)

        for d in remove_pf:
            self.data_set.remove(d)

        print '**** updating after loading item ********'
        for ds in self.data_set:
            print ds

    # ##################################################################################################################
    def generate_data(self,  choose_target_state, action_history, actions_to_reach_target, selected_type):
        last_action = action_history[-1]

        data_set = self.data_set
        remove_pf = []
        print 'actions_to_reach_target: ', actions_to_reach_target ,' for type :', selected_type

        # if self.data_set!=[] :
        #     for ds in self.data_set:
        #         # print ds
        #         if ds['route'] is not None:
        #             if ds['route'][0:len(actions_to_reach_target)] == actions_to_reach_target:
        #                 # ds['reward'] += 1
        #                 # print ds
        #             # self.data_set.remove(ds)
        #         # else:
        #         #     self.false_data_set.append(ds)
        #         #     remove_pf.append(ds)
        #
        # for d in remove_pf:
        #     data_set.remove(d)

        cts_agent = choose_target_state.agents[0]

        for i in range(self.generated_data_number - len(self.data_set)):
            particle_filter = {}

            if self.angle_pool == []:
                # tmp_radius = random.uniform(radius_min, radius_max)  # 'radius'
                # tmp_angle = random.uniform(angle_min, angle_max)  # 'angle'
                # tmp_level = random.uniform(level_min, level_max)  # 'level'
                tmp_radius = radius_min + (1.0 * (radius_max - radius_min) / self.generated_data_number) * i
                tmp_angle = angle_min + (1.0 * (angle_max - angle_min) / self.generated_data_number) * i
                tmp_level = level_min + (1.0 * (level_max - level_min) / self.generated_data_number) * i
            else:
                tmp_radius = random.choice(self.radius_pool)
                tmp_angle = random.choice(self.angle_pool)
                tmp_level = random.choice(self.level_pool)

            if [tmp_level, tmp_radius, tmp_angle] not in self.false_data_set:

                tmp_agent = agent.Agent(cts_agent.position[0], cts_agent.position[1],
                                        cts_agent.direction, selected_type, -1)
                tmp_agent.set_parameters(choose_target_state, tmp_level, tmp_radius, tmp_angle)

                tmp_agent = choose_target_state.move_a_agent(tmp_agent)  # f(p)
                target = tmp_agent.get_memory()
                p_action = tmp_agent.get_action_probability(last_action)
                route_actions = tmp_agent.route_actions

                if route_actions is not None:
                    if p_action > self.PF_add_threshold and\
                            route_actions[0:len(actions_to_reach_target)] == actions_to_reach_target:

                        particle_filter['target'] = target
                        particle_filter['parameter'] = [tmp_level, tmp_radius, tmp_angle]
                        # particle_filter['action_probability'] = p_action
                        particle_filter['route'] = tmp_agent.route_actions
                        # particle_filter['reward'] = 1
                        particle_filter['succeeded_steps'] = 0
                        self.data_set.append(particle_filter)

                        # print(particle_filter)
                        # print [tmp_level, tmp_radius, tmp_angle]
                        # print(p_action)


        print 'Selected type is :::::::::::::::',selected_type
        print 'Load numberes are :::: ',  self.load_count
        for ds in self.data_set:
            print ds
        return self.data_set

    def extract_train_set(self):
        x_train = []
        y_train = []
        if self.train_mode == 'history_based':
            # for ds in self.data_set:
                # x_train.append(ds['parameter'])
                # y_train.append(ds['action_probability'])
            for i in range(len(self.angle_pool)):
                x_train.append([self.level_pool[i],self.radius_pool[i],self.angle_pool[i]])
                # print 'x_train',[self.level_pool[i],self.radius_pool[i],self.angle_pool[i]]
                y_train.append(0.96)
        else:
           if len(self.data_set) == 0:
                return

            # Extract x, y train from generated data
           for i in range(0, self.generated_data_number):
                x_train.append(self.data_set[i][0:3])
                y_train.append(self.data_set[i][3])

        return x_train, y_train
