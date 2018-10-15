# -*- coding: utf-8 -*-
import sys
import operator, random
from copy import deepcopy

import POMDP
import poagent
import POUCT
import POMCP
import posimulator

import item
import agent

import numpy as np
from math import sqrt

def main():
    # MAP
    width, height = 10, 10
    grid = [[-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04], 
            [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
            [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
            [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
            [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
            [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
            [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
            [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
            [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
            [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04]]
    terminals = []
    evidence = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    # POMDP
    pomdp = POMDP.POMDP(width,height,grid,terminals,evidence,0.9)

    # MAIN AGENT
    main_agent = poagent.POAgent('M','l1',10,0,0,0.1,np.pi/2,(2*np.pi/2),pomdp)

    # POMCP
    pomcp = POMCP.POMCP(main_agent,None)

    # ITEMS
    items = []
    for i in range(1):
        items.append(item.item(3,2,10,'0'))

    # A AGENT
    agents = []
    for i in range(1):
        agents.append(agent.Agent(3,1,0,'l1','0'))
        agents[i].level = 8
        agents[i].radius = 1
        agents[i].co_radius = sqrt(10 ** 2 + 10 ** 2)
        agents[i].angle = 2 * np.pi
        agents[i].co_angle = 2 * np.pi

    # 1. Running the test
    print 'starting the simulation test'
    sim = posimulator.POSimulator(pomdp,items,agents,main_agent)
    while True:
        # a. running the  simulation
        print '----------------- a. sim + plannig'
        sim.show()
        next_a, next_root = pomcp.po_monte_carlo_planning(sim)
        print next_a

        # b. moving the main agt and walking on search tree
        print '----------------- b. main move'
        new_state, observation, reward = sim.real_run(next_a)
        sim.main_agent.position = new_state
        pomcp.pouct.search_tree.change_root(next_root)
        sim.show()

        # c. 

if __name__ == "__main__":
	main()