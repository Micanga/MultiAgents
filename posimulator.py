# Types for agents are 'L1','L2','F1','F2'
import agent
import intelligent_poagent
import item
import obstacle
import position
import a_star
from simulator import Simulator

from copy import copy

from numpy import pi
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


class POSimulator(Simulator,object):
    def __init__(self):
        # Common Parameters
        # >> the_map, items, agents, obstacles, main_agent,
        # enemy_agent, suspect_agent, dim_w, dim_h
        super(POSimulator,self).__init__()

    ###############################################################################################################
    @staticmethod
    def is_comment(string):
        for pos in range(len(string)):
            if string[pos] == ' ' or string[pos] == '\t':
                continue
            if string[pos] == '#':
                return True
            else:
                return False

    ###############################################################################################################
    def loader(self, path):
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
                        agnt = agent.Agent(val[1], val[2], val[3], val[4], int(val[0]))
                        agnt.set_parameters(self, val[5], val[6], val[7])
                        agnt.choose_target_state = copy(self)
                        self.agents.append(agnt)

                        j += 1
                    elif 'main' in key:
                        # x-coord, y-coord, direction, type, index
                        self.main_agent = intelligent_poagent.POAgent(val[0], val[1], val[2],float(val[5]),float(val[6]))
                        self.main_agent.level = val[4]

                    elif 'enemy' in key:
                        self.enemy_agent = intelligent_poagent.POAgent(val[0], val[1], val[2],float(val[5]),float(val[6]),True)
                        self.enemy_agent.level = val[4]

                    elif 'obstacle' in key:
                        self.obstacles.append(obstacle.Obstacle(val[0], val[1]))
                        l += 1

        # Run Checks
        assert len(self.items) == i, 'Incorrect Item Loading'
        assert len(self.agents) == j, 'Incorrect Ancillary Agent Loading'
        assert len(self.obstacles) == l, 'Incorrect Obstacle Loading'

        # Print Simulation Description
        self.show_description()

        # Update the map
        self.update_the_map()
    
    ###############################################################################################################
    def uniform_sim_sample(self):
        # 1. Defining the base simulation
        base_sim = self.copy()

        # 2. Refreshing the main agent vision
        m_agent = base_sim.main_agent
        m_agent.refresh_visibility(base_sim)

        # 3. Sampling the positions
        # a. setting on map (visible)
        base_sim.agents = []
        base_sim.items = []
        for vis_ag in m_agent.visible_agents:
            for sim_ag in self.agents:
                if sim_ag.index == vis_ag.index:
                    a = sim_ag.copy()
                    base_sim.agents.append(a)
                    break

        for vis_item in m_agent.visible_items:
            if not vis_item.loaded:
                base_sim.items.append(vis_item)

        # b. setting on map (invisible)
        for u_agent in m_agent.invisible_agents:
            for agent in self.agents:
                if u_agent.index == agent.index:
                    a = agent.copy()
                    base_sim.agents.append(a)
                    break

        for item in m_agent.invisible_items:
            for mem_i in m_agent.item_memory:
                if mem_i.index == item.index:
                    if not mem_i.loaded:
                        base_sim.items.append(item)
                        break

        base_sim.update_the_map()
        return base_sim

    ###############################################################################################################
    def take_m_observation(self):
        tmp_sim = self.copy()

        observation = list()

        self.main_agent.refresh_visibility(self)
        for agent in self.main_agent.visible_agents:
            observation.append(agent.position)
        for item in self.main_agent.visible_items:
            if not item.loaded:
                observation.append(item.get_position())

        return observation

    ###############################################################################################################
    def copy(self):
        copy_items = []
        for i in self.items:
            copy_item = i.copy()
            copy_items.append(copy_item)

        copy_agents = list()
        for a in self.agents:
            copy_agent = a.copy()
            copy_agents.append(copy_agent)

        copy_obstacles = []
        for o in self.obstacles:
            copy_obstacle = o.copy()
            copy_obstacles.append(copy_obstacle)

        tmp_sim = POSimulator()

        tmp_sim.dim_h = self.dim_h
        tmp_sim.dim_w = self.dim_w

        tmp_sim.agents = copy_agents
        tmp_sim.items = copy_items
        tmp_sim.obstacles = copy_obstacles

        if self.main_agent is not None:
            copy_main_agent = self.main_agent.copy()
            copy_main_agent.agent_memory = self.main_agent.agent_memory
            tmp_sim.main_agent = copy_main_agent

        if self.enemy_agent is not None:
            copy_enemy_agent = self.enemy_agent.copy()
            tmp_sim.enemy_agent = copy_enemy_agent

        if self.suspect_agent is not None:
            copy_suspect = self.suspect_agent.copy()
            tmp_sim.suspect_agent = copy_suspect

        tmp_sim.update_the_map()

        return tmp_sim

    ###############################################################################################################
    def show_description(self):
        print('Grid Size: {} \n{} Items Loaded\n{} Agents Loaded\n{} Obstacles Loaded'.format(self.dim_w,
                                                                                              len(self.items),
                                                                                              len(self.agents),
                                                                                              len(self.obstacles)))