import unittest
import torch
from dataclasses import field
import numpy as np

from algos.multiagent.core import RNNModelActorCritic

class TestAgentCreation(unittest.TestCase):
    def setUp(self):
        number_of_agents: int = 2
        hid_pol: int = 32
        l_pol: int = 1
        hid_val: int = 32
        l_val: int = 1
        hid_rec: int = 24
        hid_gru: int = 24
        net_type: str = 'rnn'
        minibatches: int = 1           
        observation_space = 11
        action_space = 9
            
        ac_kwargs=dict(
            hidden_sizes_pol=[[hid_pol]] * l_pol,
            hidden_sizes_val=[[hid_val]] * l_val,
            hidden_sizes_rec=[hid_rec],
            hidden=[[hid_gru]],
            net_type=net_type,
            batch_s=minibatches,
            seed=1,
            pad_dim=2
        )
              
        self.agents: dict[int, RNNModelActorCritic] = {i: RNNModelActorCritic(observation_space, action_space, **ac_kwargs) for i in range(number_of_agents)}

    def tearDown(self):
        self.agents.clear()

    def testSingleton(self):
        print('\n:: Test Agents Do not Share Class Objects ::')
        self.assertNotEqual(self.agents[0].pi, self.agents[1].pi)          
        self.agents[0].pi = None
        self.assertNotEqual(self.agents[0].pi, self.agents[1].pi)   

        


