import pytest

import algos.multiagent.ppo as PPO

import numpy as np


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
        return dict(
            gamma = 0.99,
            lamb = 0.90,
            done = np.array([False, False, False, False, False, False, False, False, False, False]),
            rewards = np.array([-0.46, -0.48, -0.46, -0.45, -0.45, -0.47, -0.48, -0.48, -0.48, -0.49]),
            values = np.array([-0.26629043, -0.26634163, -0.26718464, -0.26631153, -0.26637784, -0.26601458, -0.26657045, -0.2666973, -0.26680088, -0.26717135]),
            last_val  = -0.26717135
        )
        
                
    def test_DiscountCumSum(self, init_parameters)-> None:
        ''' test discount cumsum by testing GAE '''
        # Manual GAE calculation
        def generalized_advantage_estimate(gamma, lamb, done, rewards, values, last_val):
            """
            gamma: trajectory discount (scalar)
            lamda: exponential mean discount (scalar)
            values: value function results for each step
            rewards: rewards for each step
            done: flag for end of episode (ensures advantage only calculated for single epsiode, when multiple episodes are present)
            
            Thank you to https://nn.labml.ai/rl/ppo/gae.html
            """
            batch_size = done.shape[0]

            advantages = np.zeros(batch_size + 1)
            
            last_advantage = 0
            last_value = values[-1]

            for t in reversed(range(batch_size)):
                # Make mask to filter out values by episode
                mask = 1.0 - done[t] # convert bools into variable to multiply by
                
                # Apply terminal mask to values and advantages 
                last_value = last_value * mask
                last_advantage = last_advantage * mask
                
                # Calculate deltas
                delta = rewards[t] + gamma * last_value - values[t]

                # Get last advantage and add to proper element in advantages array
                last_advantage = delta + gamma * lamb * last_advantage                
                advantages[t] = last_advantage
                
                # Get new last value
                last_value = values[t]
                
            return advantages
                      
        manual_gae = generalized_advantage_estimate(**init_parameters)[:-1] # Remove last non-step element        
        
        # Setup for RAD-TEAM GAE from spinningup
        rews = np.append(init_parameters['rewards'], init_parameters['last_val'])
        vals = np.append(init_parameters['values'], init_parameters['last_val'])      
        
        # GAE
        deltas = rews[:-1] + init_parameters['gamma'] * vals[1:] - vals[:-1]        
        advantages = PPO.discount_cumsum(deltas, init_parameters['gamma'] * init_parameters['lamb'])

        for result, to_test in zip(manual_gae, advantages):
            assert result == to_test
            

class Test_PPOBuffer:

    def test_store(self)-> None:
        
        obs = [41.0, 0.42181818, 0.92181818, 0., 0., 0., 0., 0., 0., 0., 0.]
        act = 1
        rew = -0.46
        val 
        logp: npt.NDArray[np.float32],
        src: npt.NDArray[np.float32],


    def store_episode_length(self, episode_length: npt.NDArray) -> None:

            
    def finish_path(self, last_val: int = 0) -> None:

    def get(self, logger=None) -> Dict[str, Union[torch.Tensor, List, Dict]]:

    
# Classes
# PPOBuffer
# AgentPPO