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
    def __init__(self, generated_data_number, PF_add_threshold, train_mode, train_data_type, mutation_rate, unknown_agent, sim):

        self.generated_data_number = generated_data_number
        self.PF_add_threshold = PF_add_threshold
        self.train_mode = train_mode
        self.type = train_data_type
        self.mutation_rate = mutation_rate

        self.data_set = []
        if train_mode == 'history_based':
            self.initialise_particle_data_set(unknown_agent, sim)
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

        none_count, none_threshold = 0, 500
        x, y, direction = unknown_agent.position[0], unknown_agent.position[1], unknown_agent.direction
        tmp_agent = agent.Agent(x, y, direction, self.type, -1)

        tmp_agent.set_parameters(sim, sim.agents[0].level, sim.agents[0].radius, sim.agents[0].angle)

        # 4. Defining route
        tmp_sim = sim.copy()
        tmp_agent = tmp_sim.move_a_agent(tmp_agent)
        target = tmp_agent.get_memory()
        route_actions = tmp_agent.route_actions
        particle = {}

        # 5. Adding to the data set
        if route_actions is not None:
            particle['target'] = target
            particle['choose_target_state'] = tmp_sim
            particle['parameter'] = [sim.agents[0].level, sim.agents[0].radius, sim.agents[0].angle]
            particle['succeeded_steps'] = 1
            particle['failed_steps'] = 0
            particle['index'] = len(self.data_set)
            particle['cts_type'] = 'e'
            self.data_set.append(particle)

        while len(self.data_set) < self.generated_data_number:
            if none_count == none_threshold:
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

            # 4. Calculating route
            tmp_sim = sim.copy()
            tmp_agent = tmp_sim.move_a_agent(tmp_agent)
            target = tmp_agent.get_memory()
            route_actions = tmp_agent.route_actions

            # 5. Adding to the data set
            if route_actions is not None:
                particle['target'] = target
                particle['choose_target_state'] = tmp_sim
                particle['parameter'] = [tmp_level, tmp_radius, tmp_angle]
                particle['succeeded_steps'] = 1
                particle['failed_steps'] = 0
                particle['index'] =len(self.data_set)
                particle['cts_type'] = 'e'
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
    def check_history(self, unknown_agent,level, radius, angle, selected_type):

        success_count = 0
        print 'begin history ---------------------------------------------'
        for hist in unknown_agent.choose_target_history:
            (x, y) = hist['pos']
            print hist
            old_state = hist['state'].copy()
            tmp_agent = agent.Agent(x, y, hist['direction'], selected_type, -1)

            tmp_agent.set_parameters(old_state, level, radius, angle)

            # find the target with
            tmp_agent = old_state.move_a_agent(tmp_agent)
            target = tmp_agent.get_memory()

            if target == hist['loaded_item']:
                print target, hist['loaded_item']
                success_count +=1
        print 'end history ---------------------------------------------'
        return success_count

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

    ###################################################################################################################
    def extract_train_set(self):
        x_train, y_train = [], []

        if self.train_mode == 'history_based':
            for ds in self.data_set:
                x_train.append(ds['parameter'])
                y_train.append(ds['succeeded_steps']- (ds['failed_steps']-1))
        elif self.generated_data_number is not None:
            if len(self.data_set) == 0:
                return x_train, y_train
            for i in range(0, self.generated_data_number):
                x_train.append(self.data_set[i][0:3])
                y_train.append(self.data_set[i][3])

        return x_train, y_train

    ###################################################################################################################
    def generate_data(self, unknown_agent, selected_type, current_state):

        if len(self.data_set)>0:
            max_index = max([particle['index'] for particle in self.data_set])+1
        else:
            max_index = 1
        particle_count = max_index

        if self.data_set == [] or len(self.data_set) == 0:
            tmp_sim = unknown_agent.choose_target_state.copy()
            self.initialise_particle_data_set(unknown_agent, tmp_sim)

        # 1. Generating data (particles)

        if len(self.radius_pool) > 0:
            random_creation = self.mutation_rate * (self.generated_data_number - len(self.data_set))
            pool_creation = (1 - self.mutation_rate) * (self.generated_data_number - len(self.data_set))
        else:
            random_creation = self.generated_data_number - len(self.data_set)
            pool_creation = 0

        none_count, none_threshold = 0, self.generated_data_number
        while len(self.data_set) < self.generated_data_number:
            # a. Sampling a particle
            particle = {}
            if none_count < pool_creation:
                tmp_radius = random.choice(self.radius_pool)
                tmp_angle = random.choice(self.angle_pool)
                tmp_level = random.choice(self.level_pool)

            # if [tmp_level, tmp_radius, tmp_angle] not in self.false_data_set:
            elif none_count < random_creation:  # Mutation
                tmp_radius = random.uniform(radius_min, radius_max)  # 'radius'
                tmp_angle = random.uniform(angle_min, angle_max)  # 'angle'
                tmp_level = random.uniform(level_min, level_max)  # 'level'
            else:
                break

            # b. Simulating and Filtering the particle
            # i. creating the new particle
            x, y, direction = unknown_agent.position[0], unknown_agent.position[1], unknown_agent.direction
            tmp_agent = agent.Agent(x, y, direction, selected_type, -1)
            tmp_agent.set_parameters(unknown_agent.choose_target_state, tmp_level, tmp_radius, tmp_angle)

            if self.check_history(unknown_agent,tmp_level,tmp_radius,tmp_angle,selected_type) > 0:

                # ii. defining route
                tmp_sim = unknown_agent.choose_target_state.copy()
                tmp_agent = tmp_sim.move_a_agent(tmp_agent)  # f(p)
                target = tmp_agent.get_memory()
                particle['target'] = target
                particle['index'] = particle_count + 1
                particle['parameter'] = [tmp_level, tmp_radius, tmp_angle]
                particle['choose_target_state'] = current_state.copy()
                particle['succeeded_steps'] = len(unknown_agent.choose_target_history)
                particle['failed_steps'] = 0
                particle['cts_type'] = 'e'

                self.data_set.append(particle)
            else:
                none_count += 1

    ###################################################################################################################
    def update_data_set(self,unknown_agent,loaded_items_list,current_state, po):
        # 1. Getting the agent to update
        cts_agent = None
        if not po:
            cts_agent = copy(current_state.main_agent.visible_agents[unknown_agent.index])
        else:
            memory_agents = current_state.main_agent.agent_memory
            for m_a in memory_agents:
                if m_a.index == unknown_agent.index:
                    cts_agent = m_a
                    break

        for particle in self.data_set:
            # print particle['target'], unknown_agent.last_loaded_item_pos

            for item in loaded_items_list:

                if particle['target'] == item.get_position():

                    [tmp_level, tmp_radius, tmp_angle] = particle['parameter']

                    x, y = cts_agent.position[0], cts_agent.position[1]
                    direction = cts_agent.direction
                    tmp_agent = agent.Agent(x, y, direction, self.type, -1)

                    # b. Getting and setting the parameters data

                    tmp_agent.set_parameters(current_state, tmp_level, tmp_radius, tmp_angle)

                    # c. Simulating the selected particle
                    copy_state = current_state.copy()
                    tmp_agent = copy_state.move_a_agent(tmp_agent)
                    target = tmp_agent.get_memory()

                    if tmp_agent.route_actions is not None or current_state.items_left() == 0:
                        particle['target'] = target
                        particle['choose_target_state'] = current_state.copy()
                        particle['cts_type'] = 'u'

    ###################################################################################################################

    def evaluate_data_set(self, unknown_agent, current_state, po):
        # 1. Getting the agent to update
        cts_agent = None
        max_succeed_cts = None
        if not po:
            cts_agent = copy(current_state.main_agent.visible_agents[unknown_agent.index])
        else:
            memory_agents = current_state.main_agent.agent_memory
            for m_a in memory_agents:
                if m_a.index == unknown_agent.index:
                    cts_agent = m_a
                    break

        # 2. Increasing the load count

        # 3. Running and updating the particle filter method
        remove_pf = []
        print '*************************************************************************************************'

        print 'last_loaded_item_pos',unknown_agent.last_loaded_item_pos
        # if unknown_agent.choose_target_state.items_left() != 0 and\
        if unknown_agent.is_item_nearby(current_state.items) != -1:
            self.load_count += 1
            for particle in self.data_set:
                print particle['target'], unknown_agent.last_loaded_item_pos

                # d. Filtering the particle
                if particle['target'] == unknown_agent.last_loaded_item_pos:
                    # and                     ds['succeeded_steps'] == max_succeeded_steps:
                    print 'before', particle
                    # particle['total_steps'] += 1
                    [tmp_level, tmp_radius, tmp_angle] = particle['parameter']

                    self.level_pool.append(tmp_level)
                    self.angle_pool.append(tmp_angle)
                    self.radius_pool.append(tmp_radius)

                    if current_state.items_left() != 0:
                        # a. Creating a tmp agent to update
                        x, y = cts_agent.position[0], cts_agent.position[1]
                        direction = cts_agent.direction
                        tmp_agent = agent.Agent(x, y, direction, self.type, -1)

                        # b. Getting and setting the parameters data

                        tmp_agent.set_parameters(current_state, tmp_level, tmp_radius, tmp_angle)

                        # c. Simulating the selected particle
                        copy_state = current_state.copy()
                        tmp_agent = copy_state.move_a_agent(tmp_agent)
                        target = tmp_agent.get_memory()

                        #Update particle filters target
                        if tmp_agent.route_actions is not None :
                            print 'new target',target
                            # particle['route'] = tmp_agent.route_actions
                            particle['target'] = target

                            particle['succeeded_steps'] = self.check_history(unknown_agent, tmp_level, tmp_radius,
                                                                             tmp_angle, self.type) + 1

                            particle['failed_steps'] = 0
                            print 'after', particle
                        else:
                            if int(particle['failed_steps']) > 0:
                                self.false_data_set.append(particle)
                                remove_pf.append(particle)
                            else:
                                particle['failed_steps'] += 1
                                particle['succeeded_steps'] -= 1
                else:
                    if int(particle['failed_steps']) > 0:
                        self.false_data_set.append(particle)

                        remove_pf.append(particle)
                    else:
                        particle['failed_steps'] += 1
                        particle['succeeded_steps'] -= 1

                # 4. Removing the marked data
            for marked_particle in remove_pf:
                if marked_particle in self.data_set:
                    self.data_set.remove(marked_particle)
                    self.false_data_set.append(marked_particle)

        # 5. Updating the succeeded steps
        succeeded_sum = sum([particle['succeeded_steps'] for particle in self.data_set])

        if len(self.data_set) > 0 :
            max_succeed_cts = max([particle['choose_target_state'] for particle in self.data_set])

        if float(self.load_count) == 0.0:
            type_prob = 0.0
        else:
            type_prob = succeeded_sum
        print '*************************************************************************************************'
        print 'estimated type:', self.type, succeeded_sum
        print 'true type:',unknown_agent.agent_type
        return type_prob,max_succeed_cts

    ####################################################################################################################
    # =================Generating  D = (p,f(p)) , f(p) = P(a|H_t_1,teta,p)==============================================

    def generate_data_for_update_parameter(self, previous_state, unknown_agent, selected_type, po):

        previous_agent = unknown_agent.previous_agent_status
        action = unknown_agent.next_action
        # D= (p,f(p)) , f(p) = P(a|H_t_1,teta,p)
        # print('------------------------------------------------------------')
        for i in range(0, self.generated_data_number):

            # Generating random values for parameters

            tmp_radius = radius_min + (1.0 * (radius_max - radius_min) / self.generated_data_number) * i
            tmp_angle = angle_min + (1.0 * (angle_max - angle_min) / self.generated_data_number) * i
            tmp_level = level_min + (1.0 * (level_max - level_min) / self.generated_data_number) * i
            x, y = previous_agent.get_position()
            tmpAgent = agent.Agent(x, y, previous_agent.direction, selected_type)
            tmpAgent.agent_type = selected_type
            tmpAgent.memory = self.update_internal_state(tmp_radius, tmp_angle, tmp_level, selected_type, unknown_agent,
                                                         po)

            tmpAgent.set_parameters(previous_state, tmp_level, tmp_radius, tmp_angle)

            tmpAgent = previous_state.move_a_agent(tmpAgent)  # f(p)
            p_action = tmpAgent.get_action_probability(action)
            # print p_action,action,tmp_radius,tmp_angle,tmp_level
            if p_action is not None:
                self.data_set.append([tmp_level, tmp_radius, tmp_angle, p_action])
