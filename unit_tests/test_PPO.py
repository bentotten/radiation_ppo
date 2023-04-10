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