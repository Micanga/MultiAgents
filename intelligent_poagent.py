from copy import copy, deepcopy
from math import sqrt
from numpy.random import choice
import numpy as np
from random import sample

import agent
from intelligent_agent import Agent
import parameter_estimation
import position
import UCT
import unknown_agent
import sensor

class POAgent(Agent, object): 

    # init methods
    def __init__(self, x, y, direction,radius,angle,is_enemy = False):
        # Common Parameters
        # >> position, level, apply_adevary, direction,
        # visible_agents, next_action
        super(POAgent,self).__init__(x, y, direction)

        # PO Parameters
        self.vision = sensor.VisionSensor(radius,angle)
        self.visible_items = []
        self.invisible_agents = []
        self.invisible_items = []
        self.already_seen_position = set()

        self.agent_memory = list()
        self.item_memory = list()
        self.history = list()

    def initialise_visible_agents(self, sim, generated_data_number, PF_add_threshold, train_mode , type_selection_mode,
                                  parameter_estimation_mode, polynomial_degree,apply_adversary,
                                  type_estimation_mode, mutation_rate):
        self.apply_adversary = apply_adversary

        # 1. getting the map empty positions
        empty_positions = []
        empty_positions = [(int(x),int(y)) for x in range(sim.dim_w) for y in range(sim.dim_h)]
        empty_positions.remove(sim.main_agent.get_position())

        # 2. taking the visible and invisible content
        self.agent_memory = list()
        self.item_memory = list()
        self.init_visibility(sim)

        # 3. setting memory and map
        # a. visible stuff
        for ag in self.visible_agents:
            ag.already_seen = True
            pos = ag.position
            empty_positions.remove(pos)
            self.agent_memory.append(ag)

        for item in self.visible_items:
            item.already_seen = True
            pos = item.get_position()
            empty_positions.remove(pos)
            self.item_memory.append(item)

        # b. removing the visible position
        for x in range(sim.dim_w):
            for y in range(sim.dim_h):
                if self.see_object((x,y)):
                    self.already_seen_position.add((x,y))
                    if (x,y) in empty_positions:
                        empty_positions.remove((x,y))

        # c. invisible stuff
        for item in self.invisible_items:
            item.already_seem = False
            x,y = sample(empty_positions,1)[0]
            item.position = position.position(x,y)
            empty_positions.remove((x,y))
            self.item_memory.append(item)

        for ag in self.invisible_agents:
            ag.already_seen = False
            pos = sample(empty_positions,1)[0]
            ag.position = pos
            empty_positions.remove(pos)
            self.agent_memory.append(ag)

        # 4. starting the param estimation
        for unknown_a in self.agent_memory:
            param_estim = parameter_estimation.ParameterEstimation(generated_data_number, PF_add_threshold, train_mode,
                                                                   mutation_rate,
                                                                   apply_adversary,unknown_a,sim)
            param_estim.estimation_initialisation()
            param_estim.estimation_configuration(type_selection_mode, parameter_estimation_mode, polynomial_degree,type_estimation_mode)
            
            unknown_a.agents_parameter_estimation = param_estim

    def init_visibility(self,sim):

        self.visible_agents = []
        self.invisible_agents = []

        if sim.agents is not None:
            for ag in sim.agents:
                x,y = ag.get_position()
                a = unknown_agent.Agent(x, y, ag.direction,ag.index)
                if self.see_object((x,y)):
                    a.next_action = ag.next_action
                    a.previous_agent_status = ag
                    self.visible_agents.append(a)
                else:
                    a.next_action = None
                    a.previous_agent_status = None
                    self.invisible_agents.append(a)

        if sim.enemy_agent is not None:
            x, y = sim.enemy_agent.get_position()
            a = unknown_agent.Agent(x, y, sim.enemy_agent.direction)
            if self.see_object((x,y)):
                self.visible_agents.append(a)
            else:
                self.invisible_agents.append(a)

        self.visible_items = []
        self.invisible_items = []
        if sim.items is not None:
            for item in sim.items:
                x, y = item.get_position()
                i = item.copy()
                if self.see_object((x,y)):
                    self.visible_items.append(i)
                else:
                    self.invisible_items.append(i)

    def refresh_visibility(self,sim):
        # 1. getting the map empty positions
        empty_positions = []
        empty_positions = [(int(x),int(y)) for x in range(sim.dim_w) for y in range(sim.dim_h)]
        pos = sim.main_agent.get_position()
        empty_positions.remove(pos)

        # 2. taking the current visibility
        self.init_visibility(sim)

        # 3. Sincronizing the vision with the main_agent memory
        # a. sinc vis items
        for mem_it in self.item_memory:
            for vis_it in self.visible_items:
                if mem_it.index == vis_it.index:
                    mem_it.already_seen = True
                    mem_it.loaded = vis_it.loaded

                    if not mem_it.loaded:
                        pos = vis_it.get_position()
                        mem_it.position = position.position(pos[0],pos[1])

                        if pos in empty_positions:
                            empty_positions.remove(pos)

                    break

        # b. sinc vis agents
        for mem_ag in self.agent_memory:
            for vis_ag in self.visible_agents:
                if mem_ag.index == vis_ag.index:
                    mem_ag.already_seen = True
                    vis_ag.agents_parameter_estimation = mem_ag.agents_parameter_estimation
                    vis_ag.choose_target_state = mem_ag.choose_target_state

                    mem_ag.position = vis_ag.get_position()
                    mem_ag.direction = vis_ag.direction
                    mem_ag.next_action = vis_ag.next_action
                    mem_ag.previous_agent_status = vis_ag

                    if pos in empty_positions:
                        empty_positions.remove(pos)

                    break

        # 4. Removing the visible position
        for x in range(sim.dim_w):
            for y in range(sim.dim_h):
                if self.see_object((x,y)):
                    self.already_seen_position.add((x,y))
                    if (x,y) in empty_positions:
                        empty_positions.remove((x,y))

        # 5. Sampling the invisible stuff
        # a. sinc inv items
        for mem_it in self.item_memory:
            for inv_it in self.invisible_items:
                if mem_it.index == inv_it.index:
                    if not mem_it.loaded:
                        if mem_it.already_seen:
                            pos = mem_it.get_position()
                            inv_it.position = position.position(pos[0],pos[1])
                            if pos in empty_positions:
                                empty_positions.remove(pos)
                        else:
                            pos = sample(empty_positions,1)[0]
                            while pos in self.already_seen_position:
                                pos = sample(empty_positions,1)[0]

                            inv_it.position = position.position(pos[0],pos[1])
                            if pos in empty_positions:
                                empty_positions.remove(pos)

        # b. sinc inv agents
        for mem_ag in self.agent_memory:
            for inv_ag in self.invisible_agents:
                if mem_ag.index == inv_ag.index:
                    pos = sample(empty_positions,1)[0]
                    inv_ag.position = pos
                    inv_ag.agents_parameter_estimation = mem_ag.agents_parameter_estimation
                    inv_ag.choose_target_state = mem_ag.choose_target_state
                    inv_ag.previous_agent_status = mem_ag.previous_agent_status
                    mem_ag.next_action = None
                    if pos in empty_positions:
                        empty_positions.remove(pos)

                    break
    
    def update_unknown_agents(self, sim):
        for sim_ag in sim.agents:
            for vis_ag in self.visible_agents:
                if sim_ag.index == vis_ag.index:
                    vis_ag.position = copy(sim_ag.position)
                    vis_ag.direction  = sim_ag.direction

                    vis_ag.next_action = sim_ag.next_action
                    vis_ag.previous_agent_status = sim_ag
                    vis_ag.choose_target_state = copy(sim)

                    for mem_ag in self.agent_memory:
                        if mem_ag.index == vis_ag.index:
                            mem_ag.position = copy(sim_ag.position)
                            mem_ag.direction  = sim_ag.direction

                            mem_ag.next_action = vis_ag.next_action
                            mem_ag.previous_agent_status = sim_ag
                            mem_ag.choose_target_state = copy(sim)

                            break

            for inv_ag in self.invisible_agents:
                if sim_ag.index == inv_ag.index:
                    for mem_ag in self.agent_memory:
                        if mem_ag.index == inv_ag.index:
                            inv_ag.previous_agent_status = mem_ag
                            break

    def update_unknown_agents_status(self, sim):
        for sim_ag in sim.agents:
            for vis_ag in self.visible_agents:
                if sim_ag.index == vis_ag.index:
                    vis_ag.position = copy(sim_ag.position)
                    vis_ag.direction  = sim_ag.direction

                    vis_ag.next_action = sim_ag.next_action
                    vis_ag.previous_agent_status = sim_ag
                    vis_ag.choose_target_state = copy(sim)

                    for mem_ag in self.agent_memory:
                        if mem_ag.index == vis_ag.index:
                            mem_ag.position = copy(sim_ag.position)
                            mem_ag.direction  = sim_ag.direction

                            mem_ag.next_action = vis_ag.next_action
                            mem_ag.previous_agent_status = sim_ag
                            mem_ag.choose_target_state = copy(sim)

                            break

    def get_memory_agent(self,unknown_agent):
        for m_a in self.agent_memory:
            if m_a.index == unknown_agent.index :
                return m_a
        
    def see_object(self,obj_position):
        agent_pos = self.position
        agent_dir = self.direction
        if self.vision.in_radius(agent_pos,obj_position):
            if self.vision.in_angle(agent_pos,agent_dir,obj_position):
                return True
        return False

    def items_left(self):
        count = 0
        for im in self.item_memory:
            if not im.loaded:
                count += 1
        return count

    def copy(self):

        (x, y) = self.position

        copy_agent = POAgent(x, y, self.direction,self.vision.radius,self.vision.angle)

        copy_agent.level = self.level

        copy_agent.next_action = self.next_action

        copy_agent.history = list()
        for action in self.history:
            copy_agent.history.append(action)

        for va in self.visible_agents:
            copy_agent.visible_agents.append(va.copy())
            idx = len(copy_agent.visible_agents)-1
            copy_agent.visible_agents[idx].agents_parameter_estimation = va.agents_parameter_estimation
            copy_agent.visible_agents[idx].choose_target_state = va.choose_target_state
            copy_agent.visible_agents[idx].previous_agent_status = va.previous_agent_status
        for ia in self.invisible_agents:
            copy_agent.invisible_agents.append(ia.copy())
            idx = len(copy_agent.invisible_agents)-1
            copy_agent.invisible_agents[idx].agents_parameter_estimation = ia.agents_parameter_estimation
            copy_agent.invisible_agents[idx].choose_target_state = ia.choose_target_state
            copy_agent.invisible_agents[idx].previous_agent_status = ia.previous_agent_status

        for vi in self.visible_items:
            copy_agent.visible_items.append(vi.copy())
        for ii in self.invisible_items:
            copy_agent.invisible_items.append(ii.copy())

        for mem_agent in self.agent_memory:
            copy_agent.agent_memory.append(mem_agent.copy())

        for mem_item in self.item_memory:
            copy_agent.item_memory.append(mem_item.copy())

        return copy_agent

    def show(self):
        print '**** POAGENT *************************'
        print 'position:',self.position
        print 'level:',self.level
        print 'direction:',self.direction
        print 'visible agents:',self.visible_agents
        print 'next action:',self.next_action
        print 'vision radius:',self.vision.radius
        print 'vision angle:',self.vision.angle
        print 'history:',self.history
        print '**************************************'

    ####################################################################################################################
    def generate_previous_state(self,unknown_agent,next_action,current_state):
        previous_state = deepcopy(current_state)
        for agent in current_state.agents:
            if agent.index == unknown_agent.index:
                pos = agent.position
                if next_action == 'N':
                    agent.position =  (pos[0] - 0, pos[1] - 1)
                elif next_action == 'S':
                    agent.position =  (pos[0] - 0, pos[1] + 1)
                elif next_action == 'W':
                    agent.position =  (pos[0] + 1, pos[1] - 0)
                elif next_action == 'E':
                    agent.position =  (pos[0] - 1, pos[1] - 0)
                break
        return previous_state

    def agent_is_visible(self,unknown_agent):
        for v_a in self.visible_agents:
            if v_a.index == unknown_agent.index:
                return True
        return False

    def estimation(self,time_step,main_sim,enemy_action_prob, types, actions,current_state):
        # For the unkown agents, estimating the parameters and types
        for unknown_agent in self.agent_memory:
            if unknown_agent is not None:
                # 1. Selecting the types
                parameter_estimation = unknown_agent.agents_parameter_estimation
                if parameter_estimation.type_selection_mode == 'AS':
                    selected_types = types
                if parameter_estimation.type_selection_mode == 'BS':
                    selected_types = parameter_estimation.UCB_selection(time_step)  # returns l1, l2, f1, f2,w
                
                # 2. Defining the next agent action or appending the action
                # for the history based method
                if parameter_estimation.train_mode == 'history_based':
                    parameter_estimation.action_history.append(unknown_agent.next_action)
                    if unknown_agent.next_action != 'L':
                        parameter_estimation.actions_to_reach_target.append(unknown_agent.next_action)

                # 3. Estimating
                print unknown_agent.next_action
                if self.agent_is_visible(unknown_agent) and unknown_agent.next_action is not None:
                    tmp_sim = copy(main_sim)
                    tmp_previous_state = copy(self.previous_state)
                    parameter_estimation.process_parameter_estimations(unknown_agent,\
                        tmp_previous_state, tmp_sim, enemy_action_prob, selected_types,True)
                elif current_state is not None:
                    
                    for agent in current_state.agents:
                        if agent.index == unknown_agent.index:
                            next_action = agent.next_action
                            break

                    previous_state = self.generate_previous_state(unknown_agent,next_action,current_state)

                    unknown_agent.next_action = next_action


                    #parameter_estimation.unseen_parameter_estimation_not_update(unknown_agent,selected_types)
                    parameter_estimation.process_parameter_estimations(unknown_agent,\
                        previous_state, current_state, enemy_action_prob, selected_types,True)