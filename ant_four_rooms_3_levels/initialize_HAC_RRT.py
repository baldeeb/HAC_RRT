"""
This is the starting file for the Hierarchical Actor-Critc (HAC) algorithm.  The below script processes the command-line options specified
by the user and instantiates the environment and agent.
"""

from design_agent_and_env import design_agent_and_env
from options import parse_options
from agent import Agent
from run_HAC import run_HAC

# Determine training options specified by user.  The full list of available options can be found in "options.py" file.
FLAGS = parse_options()

# Instantiate the agent and Mujoco environment.  The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file.
agent, env = design_agent_and_env(FLAGS)

# # Begin training
# run_HAC(FLAGS,env,agent)

from rrt.hac_rrt import HAC_RRT
from rrt.rrt_utils import get_tree_branches

goal =  env.get_next_goal(FLAGS.test)
state = env.reset_sim(goal)
hac_rrt = HAC_RRT(agent.layers)
print("goal: ", goal)
success = hac_rrt.run(state, goal)




##########################################
### Visualize tree  ######################
##########################################
import matplotlib.pyplot as plt
import numpy as np

branches = get_tree_branches(hac_rrt.tree)
start = agent.layers[-1].project_state_to_end_goal(state)
# print("start: ", start)
# print("goal: ", goal)
# for k, v in hac_rrt.tree.items():
#     print(k.config.squeeze(), "   ->   ", end='')
#     if v is not None: print(v.config.squeeze())
#     else: print("None")
plt.plot(start[0], start[1], 'o', color='red')
plt.plot(goal[0], goal[1], 'o', color='green')
for b in branches:
    b = np.concatenate(b, axis=1)
    plt.plot(b[0], b[1], color='orange') #TODO: FIX! BAD
plt.show()
##########################################
##########################################



if success: 
    print("HAC RRT ran successfully!")
    path = hac_rrt.get_path()
    hac_rrt.execute_path(env)
else: print("HAC RRT Failed to find path...")



print("Done!")