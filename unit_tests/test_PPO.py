import pytest

import algos.multiagent.ppo as PPO

import numpy as np
import torch

# Functions
# combined_shape
# discount_cumsum

# Named tuples
# UpdateResult
# BpArgs
# OptimizationStorage

# Classes
# PPOBuffer
# AgentPPO

# Test helper functions
class Test_CombinedShape:    
    def test_CreateBufferofScalars(self)-> None:
        ''' Make a list of single values. Example: Make a buffer for advantages for an epoch. Size (x)'''
        max = 10
        buffer_dims = PPO.combined_shape(max)
        
        assert buffer_dims == (10,)
        adv_buff = np.zeros(buffer_dims, dtype=np.float32)
        
        assert len(adv_buff) == 10
        
        
    def test_CreateListofArrays(self)-> None:
        ''' Make a list of lists. Example: Make a buffer for source locations for an epoch (x, y). Size (x, y)'''
        max = 10
        coordinate_dimensions = (2)
        
        buffer_dims = PPO.combined_shape(max, coordinate_dimensions)
        assert buffer_dims == (10,2)
        
        source_buff = np.zeros(buffer_dims, dtype=np.float32)
        
        for step in source_buff:
            assert len(step) == 2
        
    def test_CreateListofTuples(self)-> None:
        ''' Make a list of multi-dimensional tuples. Example: Make a buffer for agent observations for an epoch. Size (x, y, z, ...)'''             
        
        max = 10
        agents = 2
        observation_dimensions = 11
        
        buffer_dims = PPO.combined_shape(max, (agents, observation_dimensions))
        assert buffer_dims == (10, 2, 11)
        
        source_buff = np.zeros(buffer_dims, dtype=np.float32)
        
        for step in source_buff:
            for agent_observation in step:
                assert len(agent_observation) == 11       
                

class Test_DiscountCumSum:    
    @pytest.fixture
    def init_parameters(self)-> dict:
        ''' Set up initialization parameters needed to test discount_cumsum '''
        last_state_value  = -0.26717135
        return dict(
            gamma = 0.99,
            lam = 0.90,
            reward_buffer = [-0.46, -0.48, -0.46, -0.45, -0.45, -0.47, -0.48, -0.48, -0.48, -0.49, last_state_value],
            values_buffer = [-0.26629043, -0.26634163, -0.26718464, -0.26631153, -0.26637784, -0.26601458, -0.26657045, -0.2666973, -0.26680088, -0.26717135, last_state_value],
        )
        
                
    def test_DiscountCumSum(self)-> None:
        
        
        def generalized_advantage_estimate(gamma, lamda, value_old_state, value_new_state, reward, done):
            """
            Get generalized advantage estimate of a trajectory
            gamma: trajectory discount (scalar)
            lamda: exponential mean discount (scalar)
            value_old_state: value function result with old_state input
            value_new_state: value function result with new_state input
            reward: agent reward of taking actions in the environment
            done: flag for end of episode
            """
            batch_size = done.shape[0]

            advantage = np.zeros(batch_size + 1)

            for t in reversed(range(batch_size)):
                delta = reward[t] + (gamma * value_new_state[t] * done[t]) - value_old_state[t]
                advantage[t] = delta + (gamma * lamda * advantage[t + 1] * done[t])

            value_target = advantage[:batch_size] + np.squeeze(value_old_state)

            return advantage[:batch_size], value_target