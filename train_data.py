import random
import agent
from copy import copy, deepcopy

radius_max = 1
radius_min = 0.1
angle_max = 1
angle_min = 0.1
level_max = 1
level_min = 0


class TrainData:
    def __init__(self, generated_data_number, PF_add_threshold, train_mode, train_data_type, uknown_agent, sim):

        self.generated_data_number = generated_data_number
        self.PF_add_threshold = PF_add_threshold
        self.train_mode = train_mode
        self.type = train_data_type

        self.data_set = []
        if train_mode == 'history_based':
            self.initialise_particle_data_set(uknown_agent, sim)
        #else:
        #    self.initialise_data_set(uknown_agent, sim)

        self.false_data_set = []
        self.level_pool = []
        self.angle_pool = []
        self.radius_pool = []
        self.load_count = 0

    ####################################################################################################################
    def initialise_particle_data_set(self, unknown_agent, sim):
        # 1. Generating initial data (particles)
        none_count, none_thereshold = 0, 500
        while len(self.data_set) < self.generated_data_number:
            if none_count == none_thereshold:
                break
            else:
                particle = {}

            # 2. Random uniform parameter sampling
            tmp_radius = random.uniform(radius_min, radius_max)  # 'radius'
            tmp_angle = random.uniform(angle_min, angle_max)  # 'angle'
            tmp_level = random.uniform(level_min, level_max)  # 'level'

            # 3. Creating the temporary agent
            x,y,direction = unknown_agent.position[0], unknown_agent.position[1], unknown_agent.direction
            tmp_agent = agent.Agent(x,y,direction, self.type, -1)
            tmp_agent.set_parameters(sim, tmp_level, tmp_radius, tmp_angle)

            # 4. Defining route
            tmp_sim = copy(sim)
            tmp_agent = tmp_sim.move_a_agent(tmp_agent)
            target = tmp_agent.get_memory()
            route_actions = tmp_agent.route_actions

            # 5. Adding to the data set
            if route_actions is not None:
                particle['target'] = target
                particle['parameter'] = [tmp_level, tmp_radius, tmp_angle]
                particle['route'] = tmp_agent.route_actions
                particle['succeeded_steps'] = 1
                particle['total_steps'] = 0
                self.data_set.append(particle)
            else:
                none_count += 1

    ####################################################################################################################
    def initialise_data_set(self, unknown_agent, sim,po = False):
        # 1. Generating initial data (particles)
        none_count, none_thereshold = 0, 500
        while len(self.data_set) < self.generated_data_number:
            if none_count == none_thereshold:
                break

            # 2. Random uniform parameter sampling
            tmp_radius = random.uniform(radius_min, radius_max)  # 'radius'
            tmp_angle = random.uniform(angle_min, angle_max)  # 'angle'
            tmp_level = random.uniform(level_min, level_max)  # 'level'

            # 3. Creating the temporary agent
            x,y,direction = unknown_agent.position[0], unknown_agent.position[1], unknown_agent.direction
            tmp_agent = agent.Agent(x,y,direction, self.type, -1)
            tmp_agent.set_parameters(sim, tmp_level, tmp_radius, tmp_angle)
            
            # 4. Defining route
            tmp_sim = copy(sim)
            tmp_agent = tmp_sim.move_a_agent(tmp_agent)
            target = tmp_agent.get_memory()
            route_actions = tmp_agent.route_actions

            if route_actions is not None and route_actions != []:
                p_action = tmp_agent.get_action_probability(route_actions[0])
                if p_action is not None:
                    self.data_set.append([tmp_level, tmp_radius, tmp_angle, p_action])
                else:
                    none_count += 1
            else:
                none_count += 1

    ####################################################################################################################
    def train_configuration(self, generated_data_number, PF_add_threshold,  train_mode):
        # the number of data we want to generate for estimating
        self.generated_data_number = generated_data_number
        self.train_mode = train_mode
        self.PF_add_threshold = PF_add_threshold

    ####################################################################################################################
    def update_internal_state(self, radius,angle,level, selected_type, unknown_agent,po):
        u_agent = None 
        if not po:
            u_agent = unknown_agent.choose_target_state.main_agent.visible_agents[unknown_agent.index]
        else:
            memory_agents = unknown_agent.choose_target_state.main_agent.agent_memory
            for m_a in memory_agents:
                if m_a.index == unknown_agent.index:
                    u_agent = m_a
                    break


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

    def generate_data_for_update_parameter(self, previous_state, unknown_agent, selected_type, po):

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
            tmpAgent.memory = self.update_internal_state(tmp_radius,tmp_angle,tmp_level, selected_type, unknown_agent, po)

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
            for ds in self.data_set:
                x_train.append(ds['parameter'])
                y_train.append(ds['succeeded_steps'])
        elif self.generated_data_number is not None:
            if len(self.data_set) == 0:
                return x_train, y_train
            for i in range(0, self.generated_data_number):
                x_train.append(self.data_set[i][0:3])
                y_train.append(self.data_set[i][3])

        return x_train, y_train

    ###################################################################################################################
    def generate_data(self, unknown_agent):
        if self.data_set == [] or len(self.data_set) == 0:
            tmp_sim = copy(unknown_agent.choose_target_state)
            self.initialise_particle_data_set(unknown_agent, tmp_sim)

        # 1. Generating data (particles)
        previous_particles = [i for i in range(0,len(self.data_set))]
        none_count, none_thereshold = 0, self.generated_data_number
        while len(self.data_set) < self.generated_data_number:
            # a. Sampling a particle
            particle = {}
            if none_count < none_thereshold:
                particle = self.data_set[random.choice(previous_particles)]
                [tmp_level, tmp_radius, tmp_angle] = particle['parameter']
            else:
                break

            # b. Simulating and Filtering the particle
            # i. creating the new particle
            x,y,direction = unknown_agent.position[0], unknown_agent.position[1], unknown_agent.direction
            tmp_agent = agent.Agent(x,y,direction, self.type, -1)
            tmp_agent.set_parameters(unknown_agent.choose_target_state, tmp_level, tmp_radius, tmp_angle)

            # ii. defining route
            tmp_sim = copy(unknown_agent.choose_target_state)
            tmp_agent = tmp_sim.move_a_agent(tmp_agent)  # f(p)
            target = tmp_agent.get_memory()
            route_actions = tmp_agent.route_actions

            # iii. filtering
            if route_actions is not None and route_actions != []:
                #if p_action > self.PF_add_threshold:
                particle['target'] = target
                particle['parameter'] = [tmp_level, tmp_radius, tmp_angle]
                particle['route'] = tmp_agent.route_actions
                particle['succeeded_steps'] = 1
                particle['total_steps'] = 0
                self.data_set.append(particle)
            else:
                none_count += 1

    ###################################################################################################################
    def count_weird_movements(self,route):
        mark_weird = []
        for i in range(len(route)):
            for j in range(len(route)):
                if route[i] == 'N' and\
                route[j] == 'S' and\
                i not in mark_weird and\
                j not in mark_weird:
                    mark_weird.append(i)
                    mark_weird.append(j)
                if route[i] == 'W' and\
                route[j] == 'E' and\
                i not in mark_weird and\
                j not in mark_weird:
                    mark_weird.append(i)
                    mark_weird.append(j)

        return float(len(route) - len(mark_weird))

    ###################################################################################################################
    def compare_actions(self,ds_actions, actions_to_reach_target):
        # Running the comparison between the real and 
        # the support vector of the PF
        j, false_actions = 0, []
        for i in range(0,len(actions_to_reach_target)):
            if j == len(ds_actions):
                while i < len(actions_to_reach_target):
                    false_actions.append(actions_to_reach_target[i])
                    i += 1
                break

            if actions_to_reach_target[i] is None or\
            actions_to_reach_target[i] == ds_actions[j]:
                j += 1
            else:
                false_actions.append(actions_to_reach_target[i])

        #false_real_len = self.count_weird_movements(false_actions)
        if len(false_actions) < float(2*len(actions_to_reach_target)/3):
            return True
        else:
            return False

    ###################################################################################################################
    def update_data_set(self, unknown_agent, actions_to_reach_target,po):
        # 1. Getting the agent to update
        cts_agent = None 
        if not po:
            cts_agent = unknown_agent.choose_target_state.main_agent.visible_agents[unknown_agent.index]
        else:
            memory_agents = unknown_agent.choose_target_state.main_agent.agent_memory
            for m_a in memory_agents:
                if m_a.index == unknown_agent.index:
                    cts_agent = m_a
                    break

        # 2. Increasing the load count
        self.load_count += 1

        # 3. Running and updating the particle filter method
        remove_pf = []
        if unknown_agent.choose_target_state.items_left() != 0:
            for particle in self.data_set:
                # a. Creating a tmp agent to update
                x, y = cts_agent.position[0], cts_agent.position[1]
                direction = cts_agent.direction
                tmp_agent = agent.Agent(x,y,direction,self.type, -1)

                # b. Getting and setting the parameters data
                [tmp_level, tmp_radius, tmp_angle] = particle['parameter']
                tmp_agent.set_parameters(unknown_agent.choose_target_state, tmp_level, tmp_radius, tmp_angle)

                # c. Simulating the selected particle
                copy_state = copy(unknown_agent.choose_target_state)
                tmp_agent = copy_state.move_a_agent(tmp_agent)
                target = tmp_agent.get_memory()

                # d. Filtering the particle
                if self.compare_actions(particle['route'], actions_to_reach_target):
                    self.level_pool.append(tmp_level)
                    self.angle_pool.append(tmp_angle)
                    self.radius_pool.append(tmp_radius)

                    if tmp_agent.route_actions is not None:
                        particle['route'] = tmp_agent.route_actions
                        particle['target'] = target
                        particle['succeeded_steps'] += 1
                        particle['total_steps'] += 1

                elif particle['succeeded_steps'] > float((2/3) * particle['total_steps']):
                    self.level_pool.append(tmp_level)
                    self.angle_pool.append(tmp_angle)
                    self.radius_pool.append(tmp_radius)

                    tmp_agent = unknown_agent.choose_target_state.move_a_agent(tmp_agent)
                    target = tmp_agent.get_memory()

                    if tmp_agent.route_actions is not None:
                        particle['route'] = tmp_agent.route_actions
                        particle['target'] = target
                        particle['total_steps'] += 1
                else:
                    self.false_data_set.append(particle)
                    remove_pf.append(particle)

        # 4. Removing the marked data
        for marked_particle in remove_pf:
            if marked_particle in self.data_set:
                self.data_set.remove(marked_particle)

        # 5. Updating the succeeded steps
        particle_sum = sum([particle['succeeded_steps'] for particle in self.data_set])
        type_prob = float(particle_sum+1)/float(self.generated_data_number)
        return type_prob
