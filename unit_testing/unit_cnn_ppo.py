import pytest
import unittest
import torch
from dataclasses import field
import numpy as np

from gym_rad_search.envs import StepResult
from algos.multiagent.NeuralNetworkCores.CNN_core import  MapsBuffer, Actor, PPO

# https://github.com/cgoldberg/python-unittest-tutorial

class UnitTestModule(unittest.TestCase):
    def test_always_passes(self):
        self.assertEqual(2+2, 4)

# New tests for new buffer - be sure to check changing an element in one array doesnt change it in the others

# class Unit_RolloutBuffer(unittest.TestCase):
#     def setUp(self):
#         self.buffer = RolloutBuffer()
#         self.buffer2 = RolloutBuffer()

#     def tearDown(self):
#         del self.buffer
#         del self.buffer2

#     def test_actions(self):
#         sample = torch.tensor(2)
#         self.buffer.actions.append(sample)
#         self.assertEqual(len(self.buffer.actions), 1)
#         self.assertEqual(self.buffer.actions[0], 2)
#         self.buffer.actions.append(torch.tensor([3]))
#         self.assertFalse(len(self.buffer.actions) == len(self.buffer2.actions))
#         self.assertFalse(self.buffer.actions == self.buffer2.actions)
#         self.buffer.clear()
#         self.assertEqual(len(self.buffer.actions), 0)
        
#     def test_states(self):
#         sample = {
#             0: {
#                 'id': 0, 
#                 'state': [
#                     1.44600000e+03, 3.33333333e-01, 3.33333333e-01, 0.00000000e+00,
#                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00
#                     ],
#                 'reward':-0.27, 
#                 'done':False, 
#                 'error':{'out_of_bounds': False, 'out_of_bounds_count': 0}
#                 }
#             }
#         self.buffer.states.append(sample)
#         self.assertEqual(len(self.buffer.states), 1)
#         self.assertEqual(len(self.buffer.states[0][0]), 5)   
#         self.buffer.states.append(4)
#         self.assertFalse(len(self.buffer.states) == len(self.buffer2.states))
#         self.assertFalse(self.buffer.states == self.buffer2.states) 
#         self.buffer.clear()
#         self.assertEqual(len(self.buffer.states), 0)
                
#     def test_logprobs(self):
#         sample = torch.Tensor([-1.4719])
#         self.buffer.logprobs.append(sample)
#         self.assertEqual(len(self.buffer.logprobs), 1)
#         self.assertEqual(self.buffer.logprobs[0], -1.4719) 
#         self.buffer.logprobs.append(4)
#         self.assertFalse(len(self.buffer.logprobs) == len(self.buffer2.logprobs))
#         self.assertFalse(self.buffer.logprobs == self.buffer2.logprobs)                     
#         self.buffer.clear()
#         self.assertEqual(len(self.buffer.logprobs), 0)
        
#     def test_rewards(self):
#         self.buffer.rewards.append(-0.3)
#         self.assertEqual(len(self.buffer.rewards), 1)
#         self.assertEqual(self.buffer.rewards[0], -0.3)    
#         self.buffer.rewards.append(4)
#         self.assertFalse(len(self.buffer.rewards) == len(self.buffer2.rewards))
#         self.assertFalse(self.buffer.rewards == self.buffer2.rewards)                 
#         self.buffer.clear()
#         self.assertEqual(len(self.buffer.rewards), 0)
                
#     def test_is_terminals(self):
#         self.buffer.is_terminals.append(False)
#         self.assertEqual(len(self.buffer.is_terminals), 1)
#         self.assertEqual(self.buffer.is_terminals[0], False)    
#         self.buffer.is_terminals.append(True)
#         self.assertFalse(len(self.buffer.is_terminals) == len(self.buffer2.is_terminals))
#         self.assertFalse(self.buffer.is_terminals == self.buffer2.is_terminals)                 
#         self.buffer.clear()
#         self.assertEqual(len(self.buffer.is_terminals), 0)
                
#     def test_mapstacks(self):
#         sample = torch.zeros(1, 4, 20, 20)
#         self.buffer.mapstacks.append(sample)
#         self.assertEqual(len(self.buffer.mapstacks), 1)
#         self.assertEqual(len(self.buffer.mapstacks[0].size()), len(sample.size()))     
#         self.buffer.mapstacks.append(4)
#         self.assertFalse(len(self.buffer.mapstacks) == len(self.buffer2.mapstacks))
#         self.assertFalse(self.buffer.mapstacks == self.buffer2.mapstacks)                   
#         self.buffer.clear()
#         self.assertEqual(len(self.buffer.mapstacks), 0)

#     def test_readings(self):
#         key = (0.3333333333333333, 0.3333333333333333)
#         sample = [1446.0]
#         self.buffer.readings[key] = sample
#         self.assertEqual(len(self.buffer.readings), 1)
#         self.assertEqual(self.buffer.readings[key][0], 1446.0)
#         self.assertFalse(len(self.buffer.readings.keys()) == len(self.buffer2.readings.keys()))                  
#         self.assertFalse(self.buffer.readings == self.buffer2.readings)
#         self.buffer.clear()
#         self.assertEqual(len(self.buffer.readings), 0)
                
#     def test_full_clear(self):
#         self.buffer.actions.append(3)


class ActorCritic(unittest.TestCase):
    def setUp(self):
        map_dim, state_dim, max_size = (10,10), 11, 10  
        input_map_stack = torch.zeros(1, 5, 22, 22)
        state = {0: StepResult()}
        state[0].id=0
        state[0].state= np.array(
            [7.10000000e+02, 4.54545455e-01, 4.54545455e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
        )
        state[0].reward=-0.35
        state[0].done=False
        state[0].error={'out_of_bounds': False, 'out_of_bounds_count': 0, 'blocked': False, 'scale': 0.00045454545454545455}
        
        self.state = state
        self.map_buffer = MapsBuffer(observation_dimension = state_dim, max_size = max_size)
        self.map_stack = self.map_buffer.observation_to_map(id=state[0].id, observation=state)
        
        self.buffer = Actor(map_dim=map_dim, state_dim=state_dim)
        
    def testConvolution(self):
        pass
        # First convolution
        # self.step1 = nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, stride=1, padding=1)  # output tensor with shape (batchs, 8, Height, Width)
        # self.relu = nn.ReLU()
        # self.step2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output height and width is floor(((Width - Size)/ Stride) +1)
        # self.step3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1) 
        # #nn.ReLU()
        # self.step4 = nn.Flatten(start_dim=0, end_dim= -1) # output tensor with shape (1, x)
        # self.step5 = nn.Linear(in_features=16 * batches * pool_output * pool_output, out_features=32) 
        # #nn.ReLU()
        # self.step6 = nn.Linear(in_features=32, out_features=16) 
        # #nn.ReLU()
        # self.step7 = nn.Linear(in_features=16, out_features=5) # TODO eventually make '5' action_dim instead
        # self.softmax = nn.Softmax(dim=0)  # Put in range [0,1] 
        # self.global_critic = global_critic

    def tearDown(self):
        del self.buffer
        del self.buffer2

if __name__ == '__main__':
    unittest.main()