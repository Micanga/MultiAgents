# Types for agents are 'L1','L2','F1','F2'
import agent
import intelligent_agent
import item
import obstacle
import position
import a_star
from random import randint
from copy import copy
from numpy.random import choice
from collections import defaultdict


dx = [-1, 0, 1,  0]  # 0: W, 1:N, 2:E, 3:S
dy = [0,  1, 0, -1]
actions = ['L', 'N', 'E', 'S', 'W']


radius_max = 1
radius_min = 0.1
angle_max = 1
angle_min = 0.1
level_max = 1
level_min = 0


class Simulator:
    def __init__(self):
        self.the_map = []
        self.items = []
        self.agents = []
        self.obstacles = []
        self.main_agent = None  # M
        self.enemy_agent = None  # W
        self.suspect_agent = None
        self.dim_w = None  # Number of columns
        self.dim_h = None  # Number of rows

    ################################################################################################################
    @staticmethod
    def is_comment(string):
        for pos in range(len(string)):
            if string[pos] == ' ' or string[pos] == '\t':
                continue
            if string[pos] == '#':
                return True
            else:
                return False

    ################################################################################################################
    def loader(self, path,log_file):
        """
        Takes in a csv file and stores the necessary instances for the simulation object. The file path referenced
        should point to a file of a particular format - an example of which can be found in utils.py txt_generator().
        The exact order of the file is unimportant - the three if statements will extract the needed information.
        :param path: File path directory to a .csv file holding the simulation information
        :return:
        """
        # Load and store csv file
        i, j, l = 0, 0, 0
        info = defaultdict(list)
        print path
        with open(path) as info_read:
            for line in info_read:
                print line
                log_file.write(line )
                if not self.is_comment(line):
                    data = line.strip().split(',')
                    key, val = data[0], data[1:]

                    if key == 'grid':
                        self.dim_w = int(val[0])
                        self.dim_h = int(val[1])

                    if 'item' in key:
                        self.items.append(item.item(val[0], val[1], val[2], i))
                        i += 1
                    elif 'agent' in key:

                        #import ipdb; ipdb.set_trace()
                        #  x, y, direction, agent_type, index
                        agnt = agent.Agent(val[1], val[2], val[3], val[4], int(val[0]))
                        #level, radius, angle
                        agnt.set_parameters(self, val[5], val[6], val[7])
                        agnt.choose_target_state = copy(self)
                        self.agents.append(agnt)

                        j += 1
                    elif 'main' in key:
                        # x-coord, y-coord, direction, type, index
                        self.main_agent = intelligent_agent.Agent(val[0], val[1], val[2])
                        self.main_agent.level = float( val[4])

                    elif 'enemy' in key:
                        self.enemy_agent = intelligent_agent.Agent(val[0], val[1], val[2],True)
                        self.enemy_agent.level = float(val[4])

                    elif 'obstacle' in key:
                        self.obstacles.append(obstacle.Obstacle(val[0], val[1]))
                        l += 1

        # Run Checks
        assert len(self.items) == i, 'Incorrect Item Loading'
        assert len(self.agents) == j, 'Incorrect Ancillary Agent Loading'
        assert len(self.obstacles) == l, 'Incorrect Obstacle Loading'

        # Print Simulation Description
        print('Grid Size: {} \n{} Items Loaded\n{} Agents Loaded\n{} Obstacles Loaded'.format(self.dim_w,
                                                                                              len(self.items),
                                                                                              len(self.agents),
                                                                                              len(self.obstacles)))
        # Update the map
        self.update_the_map()

    ################################################################################################################

    def recreate_items(self):

        for item in self.items:
            item.loaded = False
            for agent in self.agents:
                if agent.get_position() == item.get_position():
                    x,y = self.move_to_empty_place()
                    item.set_position(x,y)

            if self.main_agent is not None:
                if self.main_agent.get_position() == item.get_position():
                    x,y = self.move_to_empty_place()
                    item.set_position(x,y)

            if self.enemy_agent is not None:
                if self.enemy_agent.get_position() == item.get_position():
                    x,y = self.move_to_empty_place()
                    item.set_position(x,y)

        return

    ################################################################################################################
    def move_to_empty_place(self):
        while True :
            testXValue = randint(0, self.dim_w - 1)
            testYValue = randint(0, self.dim_h - 1)
            if self.position_is_empty( testXValue, testYValue):
                return testXValue, testYValue

    ################################################################################################################
    def get_agents_and_enemy(self):
        agents = []
        for i in range(len(self.agents)):
            agents.append(self.agents[i])
        agents.append(self.enemy_agent)
        return agents

    ################################################################################################################
    def get_highest_enemy(self):
        agents = self.get_agents_and_enemy()
        highest = None
        belief = -1
        for i in range(len(agents)):
            if agents[i].get_w_percentage() > belief:
                belief = agents[i].get_w_percentage()
                highest = agents[i]
        return highest

    ################################################################################################################
    def is_there_item_in_position(self, x, y):

        for i in range(len(self.items)):
            if not self.items[i].loaded:
                (item_x, item_y) = self.items[i].get_position()
                if (item_x, item_y) == (x, y):
                    return i

        return -1

    ###############################################################################################################

    def create_empty_map(self):

        self.the_map = list()

        row = [0] * self.dim_w

        for i in range(self.dim_h):
            self.the_map.append(list(row))

    ####################################################################################################################

    def copy(self, for_UCT = True):

        copy_items = []

        for i in range(len(self.items)):
            copy_item = self.items[i].copy()
            copy_items.append(copy_item)

        copy_agents = list()

        for cagent in self.agents:
            if for_UCT:
                (x, y) = cagent.get_position()

                copy_agent = agent.Agent(x, y, cagent.direction, cagent.agent_type, cagent.index)
                copy_agent.level = cagent.level
                copy_agent.radius = cagent.radius
                copy_agent.index = cagent.index
                copy_agent.angle = cagent.angle
                copy_agent.co_radius = cagent.co_radius
                copy_agent.co_angle = cagent.co_angle
            else:
                copy_agent = cagent.copy()
            copy_agents.append(copy_agent)

        copy_obstacles = []
        for obs in self.obstacles:
            copy_obstacle = obs.copy()
            copy_obstacles.append(copy_obstacle)

        tmp_sim = Simulator()
        tmp_sim.dim_h = self.dim_h
        tmp_sim.dim_w = self.dim_w
        tmp_sim.agents = copy_agents
        tmp_sim.items = copy_items

        if self.main_agent is not None:
            copy_main_agent = self.main_agent.copy()
            tmp_sim.main_agent = copy_main_agent

        if self.enemy_agent is not None:
            copy_enemy_agent = self.enemy_agent.copy()
            tmp_sim.enemy_agent = copy_enemy_agent

        if self.suspect_agent is not None:
            copy_suspect = self.suspect_agent.copy()
            tmp_sim.suspect_agent = copy_suspect

        tmp_sim.obstacles = copy_obstacles
        tmp_sim.update_the_map()

        return tmp_sim

    ####################################################################################################################

    def equals(self, other_simulator):

        # If I reached here the maps are equal. Now let's compare the items and agents

        if len(self.items) != len(other_simulator.items):
            return False

        for i in range(len(self.items)):
            if not self.items[i].equals(other_simulator.items[i]):
                return False

        if len(self.agents) != len(other_simulator.agents):
            return False

        for i in range(len(self.items)):
            if not self.items[i].equals(other_simulator.items[i]):
                return False

        for i in range(len(self.agents)):
            if not self.agents[i].equals(other_simulator.agents[i]):
                return False

        if not self.main_agent.equals(other_simulator.main_agent):
            return False

        if self.enemy_agent is not None and not self.enemy_agent.equals(other_simulator.enemy_agent):
            return False

        return True

    ####################################################################################################################

    def create_log_file(self,path):
        file = open(path, 'w')
        return file

    ####################################################################################################################

    def log_map(self, file):
        line =''
        for y in range(self.dim_h - 1, -1, -1):
            for x in range(self.dim_w):
                xy = self.the_map[y][x]
                if xy == 0:
                    line = line + '.'  # space
                elif xy == 1:
                    line = line + 'I'  # Items
                elif xy == 2:
                    line = line + 'S'  # start
                elif xy == 3:
                    line = line + 'R'  # route
                elif xy == 4:
                    line = line + 'D'  # finish
                elif xy == 5:
                    line = line + '+'  # Obstacle
                elif xy == 8:
                    line = line + 'A'  # A Agent
                elif xy == 9:
                    line = line + 'M'  # Main Agent
                elif xy == 10:
                    line = line + 'W'  # Enemy Agent

            file.write(line+ '\n')
            line = ''
        file.write('*********************\n')

    ####################################################################################################################

    def create_result_file(self, path):
        file = open(path, 'w')
        return file

    ####################################################################################################################

    def get_first_action(self,route):
        #  This function is to find the first action afte finding the path by  A Star

        dir = route[0]

        if dir == '0':
            return 'W'
        if dir == '1':
            return 'N'
        if dir == '2':
            return 'E'
        if dir == '3':
            return 'S'

    ####################################################################################################################

    def convert_route_to_action(self, route):
        #  This function is to find the first action afte finding the path by  A Star
        actions = []

        for dir in route:

            if dir == '0':
                actions.append('W')
            if dir == '1':
                actions.append('N')
            if dir == '2':
                actions.append('E')
            if dir == '3':
                actions.append('S')
        return actions

    ###############################################################################################################

    def items_left(self):
        items_count= 0
        for i in range(0,len(self.items)):

            if not self.items[i].loaded:

                items_count += 1


        return items_count

    ###############################################################################################################

    def draw_map(self):

        for y in reversed(range(self.dim_h)):
            for x in range(self.dim_w):
                xy = self.the_map[y][x]
                if xy == 0:
                    print '.',  # space
                elif xy == 1:
                    print 'I',  # Items
                elif xy == 2:
                    print 'S',  # start
                elif xy == 3:
                    print '.',  # route
                elif xy == 4:
                    print 'D',  # finish
                elif xy == 5:
                    print '+',  # Obstacle
                elif xy == 8:
                    print 'A',  # A Agent
                elif xy == 9:
                    print 'M',  # Main Agent
                elif xy == 10:
                    print 'W',  # Enemy Agent
            print

    ###############################################################################################################

    def update_the_map(self):

        self.create_empty_map()

        for i in range(len(self.items)):
            (item_x, item_y) = self.items[i].get_position()
            if self.items[i].loaded :
                self.the_map[item_y][item_x] = 0
            else:
                self.the_map[item_y][item_x] = 1

        for i in range(len(self.agents)):
            (agent_x, agent_y) = self.agents[i].get_position()
            self.the_map[agent_y][agent_x] = 8

            (memory_x, memory_y) = self.agents[i].get_memory()
            if (memory_x, memory_y) != (-1, -1):
                self.the_map[memory_y][memory_x] = 4

        for i in range(len(self.obstacles)):
            (obs_x, obs_y) = self.obstacles[i].get_position()
            self.the_map[obs_y][obs_x] = 5

        if self.main_agent is not None:
            (m_agent_x, m_agent_y) = self.main_agent.get_position()
            self.the_map[m_agent_y][m_agent_x] = 9

        if self.enemy_agent is not None:
            (e_agent_x, e_agent_y) = self.enemy_agent.get_position()
            self.the_map[e_agent_y][e_agent_x] = 10

    ###############################################################################################################

    def find_agent_index(self,pos):

        agents_num = len(self.agents)
        for i in range(0, agents_num):
            if self.agents[i].position == pos:
                return i
        return -1

    ###############################################################################################################

    def remove_old_destination_in_map(self): #todo: check to delete

        for y in range(self.dim_h):
            for x in range(self.dim_w):
                xy = self.the_map[y][x]
                if xy == 4:
                    self.the_map[y][x] = 1

    ################################################################################################################

    def draw_map_with_level(self):

        for y in range(self.dim_h-1,-1,-1):

            line_str = ""
            for x in range(self.dim_w):
                item_index = self.find_item_by_location(x, y)

                xy = self.the_map[y][x]

                if xy == 0:
                    line_str += ' . '

                elif xy == 1:
                    line_str += str(self.items[item_index].index)

                elif xy == 2:
                    line_str += ' S '

                elif xy == 3:
                    line_str += ' R '

                elif xy == 4:
                    line_str += ' D '

                elif xy == 5:
                    line_str += ' O '  # Obstacle

                elif xy == 8:
                    line_str += ' A '

                elif xy == 9:
                    line_str += ' M '

            print line_str
            print

    ################################################################################################################

    def find_item_by_location(self, x, y):
        for i in range(len(self.items)):
            (item_x, item_y) = self.items[i].get_position()
            if item_x == x and item_y == y:
                return i
        return -1

    ################################################################################################################

    def load_item(self, agent, destination_item_index):

        self.items[destination_item_index].loaded = True
        (agent_x, agent_y) = agent.get_position()
        self.items[destination_item_index].remove_agent(agent_x, agent_y)
        agent.last_loaded_item = copy(agent.item_to_load)
        agent.last_loaded_item_pos = self.items[destination_item_index].get_position()
        agent.item_to_load = -1
        agent.reset_memory()

        return agent

    ################################################################################################################

    def run_and_update(self, a_agent):

        a_agent = self.move_a_agent(a_agent)

        next_action = choice(actions, p=a_agent.get_actions_probabilities())  # random sampling the action

        a_agent.next_action = next_action
        a_agent.actions_history.append(next_action)

        self.update(a_agent)

        return self.agents[a_agent.index]

    # ####################################################################################################################
    # def update_all_A_agents(self, simulated):
    #     reward = 0
    #
    #     for i in range(len(self.agents)):
    #         # if not (simulated and self.agents[i].get_position() == self.suspect_agent.get_position()):
    #
    #             next_action = choice(actions,
    #                                  p=self.agents[i].get_actions_probabilities())  # random sampling the action
    #
    #             self.agents[i].next_action = next_action
    #
    #             reward += self.update(i)
    #
    #     return reward
    #
    # ################################################################################################################
    # def update(self, a_agent_index):
    #     reward = 0
    #     loaded_item = None
    #     a_agent = self.agents[a_agent_index]
    #
    #     if a_agent.next_action == 'L' and a_agent.item_to_load != -1:
    #         # print 'loading :', a_agent.item_to_load.get_position()
    #         destination = a_agent.item_to_load
    #
    #         if destination.level <= a_agent.level:  # If there is a an item nearby loading process starts
    #
    #             # load item and and remove it from map  and get the direction of agent when reaching the item.
    #             a_agent = self.load_item(a_agent, destination.index)
    #             loaded_item = self.items[destination.index]
    #             reward += 1
    #         else:
    #             if not self.items[destination.index].is_agent_in_loaded_list(a_agent):
    #                 self.items[destination.index].agents_load_item.append(a_agent)
    #
    #     else:
    #         # If there is no item to collect just move A agent
    #         (new_position_x, new_position_y) = a_agent.new_position_with_given_action(self.dim_h,self.dim_w
    #                                                                                   , a_agent.next_action)
    #
    #         if self.position_is_empty(new_position_x, new_position_y):
    #             a_agent.position = (new_position_x, new_position_y)
    #         else:
    #             a_agent.change_direction_with_action(a_agent.next_action)
    #
    #     self.agents[a_agent_index] = a_agent
    #     self.update_the_map()
    #     return reward #, loaded_item

    ################################################################################################################
    def destination_loaded_by_other_agents(self, agent):
        # Check if item is collected by other agents so we need to ignore it and change the target.

        (memory_x, memory_y) = agent.get_memory()
        destination_index = self.find_item_by_location(memory_x, memory_y)

        item_loaded = False

        if destination_index != -1:
            item_loaded = self.items[destination_index].loaded

        return item_loaded

    ################################################################################################################
    def position_is_empty(self, x, y):

        for i in range(len(self.items)):
            (item_x, item_y) = self.items[i].get_position()
            if (item_x, item_y) == (x,y) and not self.items[i].loaded:
                return False

        for i in range(len(self.obstacles)):
            (obs_x, obs_y) = self.obstacles[i].get_position()
            if (obs_x, obs_y) == (x,y) :
                return False

        for i in range(len(self.agents)):
            (agent_x, agent_y) = self.agents[i].get_position()
            if (agent_x, agent_y) == (x, y):
                return False

        if self.main_agent is not None:
            (m_agent_x, m_agent_y) =self.main_agent.get_position()
            if (m_agent_x, m_agent_y) == (x, y):
                return False

        if self.enemy_agent is not None:
            (e_agent_x, e_agent_y) =self.enemy_agent.get_position()
            if (e_agent_x, e_agent_y) == (x, y):
                return False

        return True

    ################################################################################################################
    def do_collaboration(self):
        c_reward = 0

        for item in self.items:
            agents_total_level = 0
            for agent in item.agents_load_item:
                agents_total_level += agent.level
            if agents_total_level >= item.level and item.agents_load_item !=[]:
                item.loaded = True
                c_reward += 1
                for agent in item.agents_load_item:
                    if not agent.intelligent_agent:
                        self.agents[agent.index].last_loaded_item_pos = item.get_position()
                        self.agents[agent.index].reset_memory()
                    #print '2'
                item.agents_load_item = list()


        self.update_the_map()

        return float(c_reward)
    ###############################################################################################################

    def mark_route_map(self, route, xA, yA):  # todo: check to  delete

        x = xA
        y = yA

        if len(route) > 0:
            for i in range(len(route)):
                j = int(route[i])
                x += dx[j]
                y += dy[j]
                self.the_map[y][x] = 3
    ####################################################################################################################

    def move_a_agent(self, a_agent):

        location = a_agent.position  # Location of agent
        destination = position.position(-1, -1)
        target = position.position(-1, -1)

        if self.destination_loaded_by_other_agents(a_agent):  # item is loaded by other agents so reset the memory to choose new target.
            a_agent.reset_memory()

        # If the target is selected before we have it in memory variable and we can use it
        if a_agent.memory.get_position() != (-1, -1) and location != a_agent.memory: #here
            destination = a_agent.memory

        else:  # If there is no target we should choose a target based on visible items and agents.

            a_agent.visible_agents_items(self.items, self.agents)
            target = a_agent.choose_target(self.items, self.agents)
            a_agent.choose_target_state = copy(self)

            if target.get_position() != (-1, -1):
                destination = target

            a_agent.memory = destination

        # If there is no destination the probabilities for all of the actions are same.
        if destination.get_position() == (-1, -1):
            a_agent.set_actions_probability(0.2, 0.2, 0.2, 0.2, 0.2)
            a_agent.set_random_action()
            return a_agent
        else:

            (x_destination, y_destination) = destination.get_position()  # Get the target position
            destination_index = self.find_item_by_location(x_destination, y_destination)
            self.the_map[y_destination][x_destination] = 4  # Update map with target position

            load = a_agent.is_agent_near_destination(x_destination, y_destination)

            if load:  # If there is a an item nearby loading process starts
                a_agent.item_to_load = self.items[destination_index]

                a_agent.set_actions_probabilities('L')
            else:

                a = a_star.a_star(self, a_agent)  # Find the whole path  to reach the destination with A Star
                (x_agent, y_agent) = a_agent.get_position()  # Get agent's current position

                route = a.pathFind(x_agent, y_agent, x_destination, y_destination)
                if len(route) > 1:
                    self.mark_route_map(route,x_agent, y_agent)
                a_agent.route_actions = self.convert_route_to_action(route)

                if len(route) == 0:
                    a_agent.set_actions_probability(0.2, 0.2, 0.2, 0.2, 0.2)
                    a_agent.set_random_action()
                    return a_agent

                action = self.get_first_action(route)  # Get first action of the path
                a_agent.set_actions_probabilities(action)


            return a_agent



  ####################################################################################################################
    @staticmethod
    def update_all_A_agents(sim):
            reward = 0

            for i in range(len(sim.agents)):
                # if not (simulated and self.agents[i].get_position() == self.suspect_agent.get_position()):

                    next_action = choice(actions,
                                         p=sim.agents[i].get_actions_probabilities())  # random sampling the action

                    sim.agents[i].next_action = next_action
                    r,a =sim.update(sim,sim.agents[i])
                    reward += r

            return reward

    ################################################################################################################
    @staticmethod
    def update(sim, a_agent):
        reward = 0
        loaded_item = None
        # a_agent = sim.agents[a_agent_index]

        if a_agent.next_action == 'L' and a_agent.item_to_load != -1:
            # print 'loading :', a_agent.item_to_load.get_position()
            destination = a_agent.item_to_load

            if destination.level <= a_agent.level:  # If there is a an item nearby loading process starts

                # load item and and remove it from map  and get the direction of agent when reaching the item.
                a_agent = sim.load_item(a_agent, destination.index)
                loaded_item = sim.items[destination.index]
                reward += 1
            else:
                if not sim.items[destination.index].is_agent_in_loaded_list(a_agent):
                    sim.items[destination.index].agents_load_item.append(a_agent)

        else:
            # If there is no item to collect just move A agent
            (new_position_x, new_position_y) = a_agent.new_position_with_given_action(sim.dim_h,sim.dim_w
                                                                                      , a_agent.next_action)

            if sim.position_is_empty(new_position_x, new_position_y):
                a_agent.position = (new_position_x, new_position_y)
            else:
                a_agent.change_direction_with_action(a_agent.next_action)

        # sim.agents[a_agent_index] = a_agent
        sim.update_the_map()
        return reward, a_agent #, loaded_item