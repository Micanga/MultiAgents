
import simulator
import UCT
import numpy as np

# ========== main part  ====== ===========

# Map creation

# ==============simulator initialisation=====================================================

column_no = 10  # horizontal size ,column
row_no = 10  # vertical size ,row

main_sim = simulator.simulator(column_no, row_no)
main_sim.initialisation_fixed_values()

# ================create unknown agent  ================================================================

# true parameters
true_radius = 0.78
true_angle = 0.72
true_level = 1

true_parameters = [true_level, true_radius, true_angle]

a_agent = main_sim.agents[0]
a_agent.set_parameters(true_level, true_radius, true_angle)

main_agent = main_sim.main_agent

# ======================================================================================================


# real_sim.draw_map_with_level()
main_agent.direction = 0

main_sim.draw_map()

time_step = 0
while time_step < 100:

    print 'main run count: ', time_step

    # Initializing the simulator
    a_agent = main_sim.run_and_update(a_agent)
    tmp_sim = main_sim.copy()
    main_agent_next_action = UCT.m_agent_planning(tmp_sim, true_parameters)
    print 'main_agent_next_action: ', main_agent_next_action
    print 'main_agent_direction: ', main_agent.get_agent_direction()

    r = UCT.do_move(main_sim, main_agent_next_action)

    time_step = time_step + 1

    main_sim.draw_map()
    # real_sim.draw_map_with_level()

    if main_sim.items_left() == 0:
        break
    print "left items", main_sim.items_left()

print  time_step
print "True parameters: ",true_level,true_radius,true_angle
#print "last new_estimated_parameters", new_estimated_parameters


