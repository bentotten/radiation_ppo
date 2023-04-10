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
        ''' Make a list of single values. Example: Make a buffer for actions for an epoch'''
    def test_CreateListofScalars(self)-> None:
        ''' Make a list of single values. Example: Make a buffer for actions for an epoch'''        
        