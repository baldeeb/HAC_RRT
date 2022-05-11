from rrt.rrt_utils import RrtNode, take_euclidean_step
import numpy as np
from typing import List
import time
from random import random
from queue import PriorityQueue
from copy import copy



# TODO: make all parameters as inputs. 

class HAC_RRT_Layer():
    def __init__(self, agent, level, level_count, tree={}, downstream=None):

        self.goal_threshold = np.array([1, 1])  # TODO: remove, no longer used
        
        # NOTE: tolerance and collision checks ought to be based on the steps per layer.
        # NOTE: q-distance = -ive q-value
        # NOTE: Seems like when the collision q-value is lower than the tolerance in magnitude, panning works! why?
        # # TODO: After setting up the connect function, try set this to -5  
        self.__collision_threshold = -7  # NOTE: this is based on the number of steps per level "agent.time_limit"
        
        # Indicates the maximum q-distance of a target that would allow it to be added itself 
        self.tolerance      = 9  # NOTE: the use of this value is important. need to be better defined
        
        
        self._agent = agent
        self._downstream = downstream
        self._tree = tree
        self._level = level
        self._num_levels = level_count
        
        # RRT specific
        self.goal_bias      = 0.85
        self.timeout_after  = 60 * 1
        self.path           = None


        self.start_time     = None
        self.num_iterations = 0
        self.max_attempts_per_run = 20

    def __get_value(self, state_position, goal):
        state = self._agent.project_goal_to_state(state_position)
        if self._level < self._num_levels-1: 
            goal = self._agent.project_end_goal_to_subgoal(goal)
        return self._agent.get_value(state, goal)
        
    def __get_action(self, state_position, goal):
        state = self._agent.project_goal_to_state(state_position)
        if self._level < self._num_levels-1: 
            goal = self._agent.project_end_goal_to_subgoal(goal)
        return self._agent.get_action(state, goal)

    def _find_nearest(self, nodes:List[RrtNode], target:RrtNode) -> [RrtNode, float]: 
        nearest, best_q,  = None, -float('inf')
        for n in nodes:
            state, goal = n.as_vector(), target.as_vector()
            q_val = self.__get_value(state, goal)
            if q_val > best_q: 
                nearest, best_q = n, q_val
        return nearest, -best_q

    def _agent_distance(self, state, goal):
        state, goal = state.as_vector(), goal.as_vector()
        return self.__get_value(state, goal)

    def _step(self, state:RrtNode, goal:RrtNode) -> RrtNode: 
        state_t, goal_t = state.as_vector(), goal.as_vector()
        subgoal = self.__get_action(state_t, goal_t)
        step = self._agent.project_subgoal_to_end_goal(subgoal)
        return RrtNode(step)

    def _sample_random_node(self) -> RrtNode: 
        return RrtNode(self._agent.sample_goal())

    def _is_not_reachable(self, state:RrtNode, target:RrtNode) -> bool: 
        q_val = self._downstream.__get_value(state.as_vector(), target.as_vector())
        return q_val < self.__collision_threshold

    def _is_goal(self, n): 
        for v, t in zip(self.goal_threshold, (n - self.goal).as_vector().squeeze()):
            if v > t: return False
        return True

    def _time_to_terminate(self): 
        if not self.start_time: self.start_time = time.time()
        self.num_iterations += 1

        return (time.time() - self.start_time) > self.timeout_after \
                or self.num_iterations > self.max_attempts_per_run
        

    def _sample_target_node(self):
        if random() < self.goal_bias: return self.goal
        else: return self._sample_random_node()  


    def run(self, start:RrtNode ,goal:RrtNode):
        self.path = None
        self.goal = goal
        self.start_time = time.time()

        while not self._time_to_terminate():
            sampled_node = self._sample_target_node()
            _, new_nodes = self.connect(sampled_node, tree=self._tree, source_node=start)
            # _, new_nodes = self.extend(sampled_node, tree=self._tree, source_node=start)

            if new_nodes is None: continue
            
            for n in new_nodes:
                # if self._is_goal(n):
                #     return True
                if n == goal:
                    return True
        return False    


    def connect(self, target_node, tree=None, source_node=None):
        if source_node is not None: 
            assert source_node in tree, "Extending from unknown source node!!"
        elif tree is not None:
            source_node, nearest_dist = self._find_nearest(tree, target_node)
        else:
            raise(SyntaxError("Calling the RRT extend function with insufficient parameters!"))
        
        added_nodes = []
        self.num_iterations = 0
        while not self._time_to_terminate():
            
            if self._level > 1:
                # Q-value distance
                downstream_q =  -self._downstream.__get_value(source_node.as_vector(), target_node.as_vector())
                if downstream_q < self.tolerance: 
                    downstream_target = target_node
                    source_node, extended_nodes = self._downstream.connect(downstream_target, tree=tree, source_node=source_node)
                    added_nodes.extend(extended_nodes) 
                    return source_node, added_nodes
                else: 
                    downstream_target = self._step(source_node, target_node)
                    source_node, extended_nodes = self._downstream.connect(downstream_target, tree=tree, source_node=source_node)
                
                    if len(extended_nodes) == 0:
                        return source_node, added_nodes
                    elif extended_nodes[-1] == downstream_target: 
                        added_nodes.extend(extended_nodes)
                        source_node = extended_nodes[-1]
                        continue
                    else:
                        added_nodes.extend(extended_nodes) 
                        return source_node, added_nodes

            elif self._level == 1:
                source, extended_nodes = self.extend(target_node, tree=tree, source_node=source_node)
                if extended_nodes is None:
                    return source_node, added_nodes
                else:
                    added_nodes.extend(extended_nodes)
                    if target_node in extended_nodes: return source_node, added_nodes
                    else: source_node = added_nodes[-1]
            else: 
                raise(Exception(f"Error: Attempting to extend level {self._level}!"))

        return source, added_nodes


    def extend(self, target_node, tree=None, source_node=None):
        if source_node is not None: 
            assert source_node in tree, "Extending from unknown source node!!"
        elif tree is not None:
            source_node, nearest_dist = self._find_nearest(tree, target_node)
        else:
            raise(SyntaxError("Calling the RRT extend function with insufficient parameters!"))
        
        # Euclidean distance to target
        #TODO: Useful?
        source_pos, target_pos = source_node.as_vector(), target_node.as_vector()
        euclidean_dist = np.linalg.norm(source_pos - target_pos)

        # Q-value distance
        downstream_q =  -self._downstream.__get_value(source_pos, target_pos)
        
        if downstream_q < self.tolerance or euclidean_dist < 0.1: 
            extended_node = target_node
        else:
            if self._level > 1:
                extended_node = self._step(source_node, target_node)
            elif self._level == 1:
                proposed_target_node = self._step(source_node, target_node)

                delta = proposed_target_node - source_node
                extended_node = source_node + (delta / 3) # TODO: make a tunable parameter

        print("##############################")
        print("extending to: ", target_node.config.squeeze())
        print("nearest: ", source_node.config.squeeze())
        print("selected: ", extended_node.config.squeeze())
        print("level: ", self._level)
        print(" ")

        if self._level > 1:
            success = self._downstream.run(source_node, extended_node)
            return source_node, [extended_node] if success else None  
        elif self._level == 1:
            # TODO: Check if this is useful given the way the node is selected above
            # Use the q value from above maybe? 
            if self._is_not_reachable(source_node, extended_node): 
                return source_node, None
            else:
                tree[extended_node] = source_node
                return source_node, [extended_node]
        else:
            raise(RuntimeError(f"Error: Attempting to extend level {self._level}!"))

    def _trace_back_path(self, tree, node):
        path, n = [], node
        while n is not None:
            path.append(n.config)
            n = tree[n]
        return path 

    def get_path(self): 
        if self.path is None: raise(Exception("No Path Available!"))
        return self.path






class HAC_RRT:
    def __init__(self, layer_agents):
        self.tree = {}
        self._bottom_agent = layer_agents[0] 
        num_layers = len(layer_agents)
        self.layers = [HAC_RRT_Layer(layer_agents[0], level=0, level_count=num_layers, tree=self.tree)]
        for i in range(1, num_layers):
            upstream = HAC_RRT_Layer(layer_agents[i], i, num_layers, self.tree, self.layers[-1])
            self.layers.append(upstream)

    def run(self, state, goal):
        # TODO: this should be conditioned on state space 
        # being different than """goal space"""
        # TODO: Should the RRT tree hold things in subgoal space?
        start_position = self._bottom_agent.project_state_to_end_goal(state)
        start_node = RrtNode(start_position)
        goal_node = RrtNode(goal)

        print("Starting planner")
        print("run from: ", start_position)
        print("to: ", goal)
        print(" ")

        self.tree.update({start_node: None})
        
        if self.layers[-1].run(start_node, goal_node) is True:
            self.path = self.layers[-1]._trace_back_path(self.tree, goal_node)
            self.path.reverse()
            return True
        return False

    def get_path(self):
        return self.path
        
    def execute_path(self, env):
        
        class TerminationIndicator:
            def __init__(self, env, subgoal, agent):
                self.steps_taken = 0
                self.max_steps_per_segment = 50  # TODO: NOTE This is causing issues. 
                                                 #     Instead maybe trigger replan when agent is stuck 
                self.env = env
                self.agent = agent
                self.goal_position = self.agent.project_subgoal_to_end_goal(subgoal)
            
            def __goal_reached(self, state):
                current_position = self.agent.project_state_to_end_goal(state)
                return all(np.absolute(self.goal_position-current_position) < self.env.end_goal_thresholds)
            
            def __call__(self, state):
                self.steps_taken += 1
                if self.__goal_reached(state):
                    print("Reached subgoal!")
                    return True
                if (self.steps_taken > self.env.max_actions):
                    print("Reached Max iter!")
                    return True
                if (self.steps_taken > self.max_steps_per_segment):
                    print("Reached Max steps per episode")
                    return True
                return False

        assert(len(self.path) > 0)
        state = env.get_state()
        print("executing rrt path: ")
        for p in self.path:
            print("\theading towards ", p.squeeze())
            subgoal = self._bottom_agent.project_end_goal_to_subgoal(p)
            time_to_terminate = TerminationIndicator(env, subgoal, self._bottom_agent)
            while not time_to_terminate(state):
                action = self._bottom_agent.get_action(state, subgoal)
                state = env.execute_action(action)
            print('\treached: ', self._bottom_agent.project_state_to_subgoal(state))