import random
import agent
from copy import deepcopy

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
    def update_internal_state(self, radius,angle,level, selected_type, unknown_agent , po =False):

        if po:
            u_agent = unknown_agent.choose_target_state.main_agent.agent_memory[unknown_agent.index]
        else:
            u_agent = unknown_agent.choose_target_state.main_agent.visible_agents[unknown_agent.index]


        tmp_sim = unknown_agent.choose_target_state
        (x, y) = u_agent.get_position()

        tmp_agent = agent.Agent(x, y, u_agent.direction, selected_type, -1)

        tmp_agent.set_parameters(unknown_agent.choose_target_state, level, radius, angle)

        # find the target with
        tmp_agent.visible_agents_items(tmp_sim.items, tmp_sim.agents)
        target = tmp_agent.choose_target(tmp_sim.items, tmp_sim.agents)

        return target

    #################################################################################################################

    # =================Generating  D = (p,f(p)) , f(p) = P(a|H_t_1,teta,p)==============================================

    def generate_data_for_update_parameter(self, previous_state, unknown_agent, selected_type,po = False):

        previous_agent = unknown_agent.previous_agent_status
        action = unknown_agent.next_action
        # D= (p,f(p)) , f(p) = P(a|H_t_1,teta,p)
        #print('------------------------------------------------------------')
        for i in range(0, self.generated_data_number):

            # Generating random values for parameters

            tmp_radius = radius_min + (1.0 * (radius_max - radius_min) / self.generated_data_number) * i
            tmp_angle = angle_min + (1.0 * (angle_max - angle_min) / self.generated_data_number) * i
            tmp_level = level_min + (1.0 * (level_max - level_min) / self.generated_data_number) * i
            x, y = previous_agent.get_position()
            tmpAgent = agent.Agent(x, y, previous_agent.direction, selected_type)
            tmpAgent.agent_type = selected_type
            tmpAgent.memory = self.update_internal_state(tmp_radius,tmp_angle,tmp_level, selected_type, unknown_agent,po)

            tmpAgent.set_parameters(previous_state, tmp_level, tmp_radius, tmp_angle)

            tmpAgent = previous_state.move_a_agent(tmpAgent)  # f(p)
            p_action = tmpAgent.get_action_probability(action)
            # print p_action,action,tmp_radius,tmp_angle,tmp_level
            if p_action is not None:
                self.data_set.append([tmp_level, tmp_radius, tmp_angle, p_action])
    
    ###################################################################################################################
    def extract_train_set(self):
        x_train, y_train = [], []

        if self.train_mode == 'history_based':
            for i in range(len(self.angle_pool)):
                x_train.append([self.level_pool[i],self.radius_pool[i],self.angle_pool[i]])
                y_train.append(0.96)
        else:
           if len(self.data_set) == 0:
                return
           for i in range(0, self.generated_data_number):
                x_train.append(self.data_set[i][0:3])
                y_train.append(self.data_set[i][3])

        return x_train, y_train

    ###################################################################################################################
    def generate_data(self, unknown_agent, action_history,
                      actions_to_reach_target, selected_type):
        # 1. Getting the information
        vis_agents = unknown_agent.choose_target_state.main_agent.visible_agents
        last_action = action_history[-1]

        # 2. Getting the "cts" agent
        cts_agent = None
        for v_a in vis_agents:
            if v_a.index == unknown_agent.index: 
                cts_agent = v_a
        
        if cts_agent == None:
            return 0 

        # 3. Generating data (particles)
        new_data_count = self.generated_data_number - len(self.data_set)
        #for i in range(0,new_data_count):
        while len(self.data_set) < self.generated_data_number:
            print len(self.data_set)
            particle_filter = {}

            # i. Initializing the temporary parameters
            if self.angle_pool == []:
                tmp_radius = random.uniform(radius_min, radius_max)  # 'radius'
                tmp_angle = random.uniform(angle_min, angle_max)  # 'angle'
                tmp_level = random.uniform(level_min, level_max)  # 'level'
            else:
                tmp_radius = random.choice(self.radius_pool)
                tmp_angle = random.choice(self.angle_pool)
                tmp_level = random.choice(self.level_pool)

            # ii. Simulating the particle
            if [tmp_level, tmp_radius, tmp_angle] not in self.false_data_set:
                x,y,direction = cts_agent.position[0], cts_agent.position[1], cts_agent.direction
                tmp_agent = agent.Agent(x,y,direction, selected_type, -1)
                tmp_agent.set_parameters(unknown_agent.choose_target_state, tmp_level, tmp_radius, tmp_angle)

                tmp_agent = unknown_agent.choose_target_state.move_a_agent(tmp_agent)  # f(p)
                target = tmp_agent.get_memory()
                p_action = tmp_agent.get_action_probability(last_action)
                route_actions = tmp_agent.route_actions

                if route_actions is not None:
                    if p_action > self.PF_add_threshold and\
                    route_actions[0:len(actions_to_reach_target)] == actions_to_reach_target:

                        particle_filter['target'] = target
                        particle_filter['parameter'] = [tmp_level, tmp_radius, tmp_angle]
                        particle_filter['route'] = tmp_agent.route_actions
                        particle_filter['succeeded_steps'] = 0
                        self.data_set.append(particle_filter)

        # 4. Getting the max_succeeded_steps with the new particle
        # in the data set
        seq = [data['succeeded_steps'] for data in self.data_set]
        if seq != []:
            max_succeeded_steps = max(seq)
        else:
            max_succeeded_steps = 0

        # 5. Calculating the Type Probability
        if self.load_count == 0:
            type_prob = 1
        else:
            type_prob = float(max_succeeded_steps) / float(self.load_count)
        return type_prob

    ###################################################################################################################
    @staticmethod
    def compare_actions(ds_actions, actions_to_reach_target):
        # Running the comparison between the real and 
        # the support vector of the PF
        j, false_actions = 0, []
        for i in range(0,len(actions_to_reach_target)):
            if j == len(ds_actions):
                return False

            if actions_to_reach_target[i] is None or\
            actions_to_reach_target[i] == ds_actions[j]:
                j += 1
            else:
                false_actions.append(actions_to_reach_target[i])

        if ((2/3)*len(actions_to_reach_target)) > len(false_actions):
            return True
        else:
            return False

    ###################################################################################################################
    def update_data_set(self, unknown_agent, actions_to_reach_target, selected_type, po=False):
        # 1. Getting the agent to update
        if po:
            cts_agent = unknown_agent.choose_target_state.main_agent.agent_memory[unknown_agent.index]
        else:
            cts_agent = unknown_agent.choose_target_state.main_agent.visible_agents[unknown_agent.index]

        # 2. Counting the succeeded steps and increasing
        # the load count
        self.load_count += 1
        seq = [data['succeeded_steps'] for data in self.data_set]
        print seq
        if seq != []:
            max_succeeded_steps = max(seq)
        else:
            max_succeeded_steps = 0

        # 3. Running and updating the particle filter method
        remove_pf = []
        if unknown_agent.choose_target_state.items_left() != 0:
            for ds in self.data_set:
                # i. Creating a tmp agent to update
                x, y = cts_agent.position[0], cts_agent.position[1]
                direction = cts_agent.direction
                tmp_agent = agent.Agent(x,y,direction,selected_type, -1)

                # ii. Getting and setting the parameters data
                [tmp_level, tmp_radius, tmp_angle] = ds['parameter']
                tmp_agent.set_parameters(unknown_agent.choose_target_state, tmp_level, tmp_radius, tmp_angle)

                print "1)",self.compare_actions(ds['route'], actions_to_reach_target)
                print ds['route'],actions_to_reach_target
                print "2)",ds['succeeded_steps'] > (2/3) * self.load_count
                if self.compare_actions(ds['route'], actions_to_reach_target)\
                 or ds['succeeded_steps'] > (2/3) * self.load_count:
                    self.level_pool.append(tmp_level)
                    self.angle_pool.append(tmp_angle)
                    self.radius_pool.append(tmp_radius)

                    tmp_agent = unknown_agent.choose_target_state.move_a_agent(tmp_agent)
                    target = tmp_agent.get_memory()

                    if tmp_agent.route_actions is not None:
                        ds['route'] = tmp_agent.route_actions
                        ds['target'] = target
                        ds['succeeded_steps'] += 1
                    else:
                        remove_pf.append(ds)
                else:
                    self.false_data_set.append(ds)
                    remove_pf.append(ds)

        # 4. Removing the weird data
        for d in remove_pf:
            self.data_set.remove(d)

        # 5. Updating the succeeded steps
        seq = [data['succeeded_steps'] for data in self.data_set]
        if seq != []:
            max_succeeded_steps = max(seq)
        else:
            max_succeeded_steps = 0

        return float(max_succeeded_steps)/float(self.load_count)
