# type: ignore
import pytest

import algos.multiagent.ppo as PPO

import numpy as np
import torch

# Helper functions
@pytest.fixture
def helpers():
    return Helpers
         

class Helpers:
    @staticmethod
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

    @staticmethod
    def rewards_to_go(batch_rews, gamma):
        ''' 
        Calculate the rewards to go. Gamma is the discount factor.
        Thank you to https://medium.com/swlh/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a
        '''
        # The rewards-to-go (rtg) per episode per batch to return and the shape will be (num timesteps per episode).
        batch_rtgs = [] 
        
        # Iterate through each episode backwards to maintain same order in batch_rtgs
        discounted_reward = 0 # The discounted reward so far
        
        for rew in reversed(batch_rews):
            discounted_reward = rew + discounted_reward * gamma
            batch_rtgs.insert(0, discounted_reward)
                
        return batch_rtgs     

    @staticmethod
    def normalization_trick(adv_buffer: np.array):
        adv_mean = adv_buffer.mean()
        adv_std = adv_buffer.std()
        return (adv_buffer - adv_mean) / adv_std        


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
               
    def test_DiscountCumSum(self, init_parameters, helpers)-> None:
        ''' test discount cumsum by testing GAE '''
       
        manual_gae = helpers.generalized_advantage_estimate(**init_parameters)[:-1] # Remove last non-step element        
        
        # Setup for RAD-TEAM GAE from spinningup
        rews = np.append(init_parameters['rewards'], init_parameters['last_val'])
        vals = np.append(init_parameters['values'], init_parameters['last_val'])      
        
        # GAE
        deltas = rews[:-1] + init_parameters['gamma'] * vals[1:] - vals[:-1]        
        advantages = PPO.discount_cumsum(deltas, init_parameters['gamma'] * init_parameters['lamb'])

        for result, to_test in zip(manual_gae, advantages):
            assert result == to_test
            

class Test_PPOBuffer:
    @pytest.fixture
    def init_parameters(self)-> dict:
        ''' Set up initialization parameters '''
        return dict(
            observation_dimension = 11,
            max_size = 2,
            max_episode_length = 2,
            number_agents = 2
        )
            
    def test_Init(self, init_parameters):
        _ = PPO.PPOBuffer(**init_parameters)    
        
    def test_QuickReset(self, init_parameters):
        buffer = PPO.PPOBuffer(**init_parameters)    
        
        buffer.ptr = 1
        buffer.path_start_idx = 1
        buffer.episode_lengths_buffer.append(1)
        buffer.quick_reset()
        
        assert buffer.ptr == 0     
        assert buffer.path_start_idx == 0                   
        assert len(buffer.episode_lengths_buffer) == 0
        
    def test_store(self, init_parameters)-> None:
        # Instatiate
        buffer = PPO.PPOBuffer(**init_parameters)    
        
        # Set up step results
        obs = np.array([41.0, 0.42181818, 0.92181818, 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
        act = 1
        rew = -0.46
        val = -0.26629042625427246
        logp = -1.777620792388916 
        src = np.array([788.0, 306.0])
        full_obs = {0: obs, 1: np.array([41.0, 0.42181818, 0.92181818, 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)}        
        test = np.zeros((11,), dtype=np.float32) # For comparison with empty
        
        # Store 1st set
        buffer.store(
            obs=obs,
            act=act,
            rew=rew,
            val=val,
            logp=logp,
            src=src,
            full_observation=full_obs
        )
        
        # Check stored correctly
        assert buffer.obs_buf.shape == (2,11)
        assert np.array_equal(buffer.obs_buf[0], obs)
        
        assert buffer.act_buf.shape == (2,)
        assert buffer.act_buf[0] == act
        
        assert buffer.rew_buf.shape == (2,)
        assert buffer.rew_buf[0] == pytest.approx(rew)
        
        assert buffer.val_buf.shape == (2,)
        assert buffer.val_buf[0] == pytest.approx(val)
        
        assert buffer.source_tar.shape == (2,2)
        assert np.array_equal(buffer.source_tar[0], src)      
        
        assert buffer.logp_buf.shape == (2,)
        assert buffer.logp_buf[0] == pytest.approx(logp)
        
        for agent_id, agent_obs in full_obs.items():
            assert np.array_equal(buffer.full_observation_buffer[0][agent_id], agent_obs)     
            
        # Check remainder are zeros        
        for i in range(1, init_parameters['max_size']):
            assert np.array_equal(buffer.obs_buf[i], test)
            assert buffer.act_buf[i] == 0
            assert buffer.rew_buf[i] == 0.0
            assert buffer.val_buf[i] == 0.0
            assert np.array_equal(buffer.source_tar[i], np.zeros((2,), dtype=np.float32))   
            assert buffer.logp_buf[i] == 0.0

            for id in range(1, init_parameters['number_agents']):
                assert np.array_equal(buffer.full_observation_buffer[i][id], test)

        # Check pointer updated
        assert buffer.ptr == 1
                                       
        # Store 2nd set
        buffer.store(
            obs=obs,
            act=act,
            rew=rew,
            val=val,
            logp=logp,
            src=src,
            full_observation=full_obs
        )
        
        # Check stored correctly
        assert buffer.obs_buf.shape == (2,11)
        assert np.array_equal(buffer.obs_buf[1], obs)
        
        assert buffer.act_buf.shape == (2,)
        assert buffer.act_buf[1] == act
        
        assert buffer.rew_buf.shape == (2,)
        assert buffer.rew_buf[1] == pytest.approx(rew)
        
        assert buffer.val_buf.shape == (2,)
        assert buffer.val_buf[1] == pytest.approx(val)
        
        assert buffer.source_tar.shape == (2,2)
        assert np.array_equal(buffer.source_tar[1], src)      
        
        assert buffer.logp_buf.shape == (2,)
        assert buffer.logp_buf[1] == pytest.approx(logp)
        
        for agent_id, agent_obs in full_obs.items():
            assert np.array_equal(buffer.full_observation_buffer[1][agent_id], agent_obs)     
            
        # Check remainder are zeros        
        for i in range(2, init_parameters['max_size']):
            assert np.array_equal(buffer.obs_buf[i], test)
            assert buffer.act_buf[i] == 0
            assert buffer.rew_buf[i] == 0.0
            assert buffer.val_buf[i] == 0.0
            assert np.array_equal(buffer.source_tar[i], np.zeros((2,), dtype=np.float32))   
            assert buffer.logp_buf[i] == 0.0

            for id in range(1, init_parameters['number_agents']):
                assert np.array_equal(buffer.full_observation_buffer[i][id], test)        

        # Check pointer updated
        assert buffer.ptr == 2
        
        # Check failure when ptr exceeds max_size
        with pytest.raises(AssertionError):
            buffer.store(
                obs=obs,
                act=act,
                rew=rew,
                val=val,
                logp=logp,
                src=src,
                full_observation=full_obs
            )           

    def test_store_episode_length(self, init_parameters)-> None:
        buffer = PPO.PPOBuffer(**init_parameters)    
        assert len(buffer.episode_lengths_buffer) == 0
        buffer.store_episode_length(7)
        assert len(buffer.episode_lengths_buffer) == 1
        assert buffer.episode_lengths_buffer[0] == 7
        
    def test_GAE_advantage_and_rewardsToGO_hardcoded(self, helpers)-> None:        
        # Manual test variables                
        test = dict(
            gamma = 0.99,
            lamb = 0.90,
            done = np.array([False, False, False, False, False, False, False, False, False, False]),
            rewards = np.array([-0.46, -0.48, -0.46, -0.45, -0.45, -0.47, -0.48, -0.48, -0.48, -0.49]),
            values = np.array([-0.26629043, -0.26634163, -0.26718464, -0.26631153, -0.26637784, -0.26601458, -0.26657045, -0.2666973, -0.26680088, -0.26717135]),
            last_val  = -0.26717135
        )     
        
        manual_gae = helpers.generalized_advantage_estimate(**test)[:-1] # Remove last non-step element                
        rewards = np.append(test['rewards'], test['last_val']).tolist()
        manual_rewardsToGo = helpers.rewards_to_go(batch_rews=rewards, gamma=test['gamma'])[:-1] # Remove last non-step element   
                        
        # setup PPO buffer
        init_parameters = dict(
            observation_dimension = 11,
            max_size = 10,
            max_episode_length = 2,
            number_agents = 2
        )
        
        buffer = PPO.PPOBuffer(**init_parameters)
                       
        buffer.rew_buf = test['rewards']
        buffer.val_buf = test['values']
        buffer.ptr = 10
             
        buffer.GAE_advantage_and_rewardsToGO(last_state_value=test['last_val'])
        
        for result, to_test in zip(manual_rewardsToGo, buffer.ret_buf):
            assert result == pytest.approx(to_test)         
            
        for result, to_test in zip(manual_gae, buffer.adv_buf):
            assert result == pytest.approx(to_test)    

    def test_GAE_advantage_and_rewardsToGO_with_storage(self, helpers)-> None:        
        # Manual test variables                
        test = dict(
            gamma = 0.99,
            lamb = 0.90,
            done = np.array([False, False, False]),
            rewards = np.array([-0.46, -0.48, -0.46]),
            values = np.array([-0.26629043, -0.26634163, -0.26718464]),
            last_val  = -0.26718464
        )     
        
        manual_gae = helpers.generalized_advantage_estimate(**test)[:-1] # Remove last non-step element                
        rewards = np.append(test['rewards'], test['last_val']).tolist()
        manual_rewardsToGo = helpers.rewards_to_go(batch_rews=rewards, gamma=test['gamma'])[:-1] # Remove last non-step element   
                        
        # setup PPO buffer
        init_parameters = dict(
            observation_dimension = 11,
            max_size = 10,
            max_episode_length = 2,
            number_agents = 2
        )
        
        buffer = PPO.PPOBuffer(**init_parameters)
            
        # Prime buffer
        # 1st step: 
        buffer.store(
            obs=np.zeros((11,), dtype=np.float32),
            act=0,
            rew=test['rewards'][0],
            val=test['values'][0],
            logp=0,
            src=np.zeros((1,2), dtype=np.float32),
            full_observation={0: np.zeros(11,), 1: np.zeros(11,)}
        )
        # 2nd step:
        buffer.store(
            obs=np.zeros((11,), dtype=np.float32),
            act=0,
            rew=test['rewards'][1],
            val=test['values'][1],
            logp=0,
            src=np.zeros((1,2), dtype=np.float32),
            full_observation={0: np.zeros(11,), 1: np.zeros(11,)}
        )  
        # 3rd step:
        buffer.store(
            obs=np.zeros((11,), dtype=np.float32),
            act=0,
            rew=test['rewards'][2],
            val=test['values'][2],
            logp=0,
            src=np.zeros((1,2), dtype=np.float32),
            full_observation={0: np.zeros(11,), 1: np.zeros(11,)}
        )               
              
        buffer.GAE_advantage_and_rewardsToGO(last_state_value=test['last_val'])
        
        for result, to_test in zip(manual_rewardsToGo, buffer.ret_buf):
            assert result == pytest.approx(to_test)         
            
        for result, to_test in zip(manual_gae, buffer.adv_buf):
            assert result == pytest.approx(to_test)                                                             
        
    def test_get(self, init_parameters)-> None:
        buffer = PPO.PPOBuffer(**init_parameters)    
                
        # Manual test variables                
        test = dict(
            gamma = 0.99,
            lamb = 0.90,
            obs = np.array([41.0, 0.42181818, 0.92181818, 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32),
            full_obs = {
                0: np.array([41.0, 0.42181818, 0.92181818, 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32), 
                1: np.array([41.0, 0.42181818, 0.92181818, 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
                },                    
            done = np.array([False, False]),
            rewards = np.array([-0.46, -0.48]),
            values = np.array([-0.26629043, -0.26634163]),
            src = np.array([788.0, 306.0]),
            act = np.array([1, 2]),
            logp = np.array([-1.777620792388916, -1.777620792388916]),
            last_val = -0.26634163
        )     
            
        # Prime buffer
        # 1st step: 
        buffer.store(
            obs=test['obs'],
            act=test['act'][0],
            rew=test['rewards'][0],
            val=test['values'][0],
            logp=test['logp'][0],
            src=test['src'],
            full_observation=test['full_obs']
        )
        # 2nd step:
        buffer.store(
            obs=test['obs'],
            act=test['act'][1],
            rew=test['rewards'][1],
            val=test['values'][1],
            logp=test['logp'][1],
            src=test['src'],
            full_observation=test['full_obs']
        )
        
        buffer.store_episode_length(2)
        buffer.GAE_advantage_and_rewardsToGO(test['last_val'])
        data = buffer.get()

        # Make sure reset happened
        assert buffer.ptr == 0     
        assert buffer.path_start_idx == 0                   
        assert len(buffer.episode_lengths_buffer) == 0
        
        # Check observations        
        i = 0
        obs_buffer_tensor =  data['obs'].tolist()        
        for x, y in zip(*obs_buffer_tensor):
            assert x == test['obs'][i]
            assert y == test['obs'][i]
            i += 1

        # Check actions
        i = 0
        act_buffer_tensor =  data['act'].tolist()        
        for x in act_buffer_tensor:
            assert x == test['act'][i]
            i += 1    

        # TODO Finish remaining checks when time. For now skipping to move on to more important checks            


class Test_PPOAgent:
    @pytest.fixture
    def init_parameters(self)-> dict:
        ''' Set up initialization parameters '''
        bpargs = dict(
            bp_decay=0.1,
            l2_weight=1.0,
            l1_weight=0.0,
            elbo_weight=1.0,
            area_scale=5
        )          
        ac_kwargs={
            'action_space': 8, 
            'observation_space': 11, 
            'steps_per_episode': 1, 
            'number_of_agents': 2, 
            'detector_step_size': 100.0, 
            'environment_scale': 0.00045454545454545455, 
            'bounds_offset': np.array([200., 500.]), 
            'enforce_boundaries': False, 
            'grid_bounds': (1, 1), 
            'resolution_multiplier': 0.01, 
            'GlobalCritic': None, 
            'save_path': ['../../models/train/test/2023-04-11-16:34:26', '2023-04-11-16:34:26_test_agents1_s2']
            }
                      
        return dict(
            id = 0,
            observation_space = 11,
            bp_args = bpargs,
            steps_per_epoch = 3,
            steps_per_episode = 2,
            number_of_agents = 2,
            env_height = 5,
            actor_critic_args = ac_kwargs  
        )
            
    def test_Init(self, init_parameters):
        _ = PPO.AgentPPO(**init_parameters)
        # TODO add custom checks for different combos with CNN/RAD-A2C/Global Critic       
        
    def test_reduce_pfgru_training(self, init_parameters):
        AgentPPO = PPO.AgentPPO(**init_parameters)        
        assert AgentPPO.reduce_pfgru_iters == True
        assert AgentPPO.train_pfgru_iters == 15
        AgentPPO.reduce_pfgru_training()
        assert AgentPPO.reduce_pfgru_iters == False
        assert AgentPPO.train_pfgru_iters == 5
        
    def test_step(self, init_parameters):
        ''' Wrapper between CNN and Train '''
        AgentPPO = PPO.AgentPPO(**init_parameters)    
        
        observations = {
            0: np.array([41.0, 0.42181818, 0.92181818, 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32), 
            1: np.array([41.0, 0.42181818, 0.92181818, 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
            }
        
        observations: Dict[int, List[Any]]
        hiddens: Union[None, Dict] = None
        save_map: bool = True
        message: Union[None, Dict] =None
        -> RADCNN_core.ActionChoice:
            
        ''' Wrapper for neural network action selection'''
        if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
            assert type(hiddens) == dict
            results = self.agent.step(observations[self.id], hidden=hiddens[self.id]) # type: ignore
        elif self.actor_critic_architecture == 'cnn':
            results = self.agent.select_action(observations, self.id, save_map=save_map)  # TODO add in hidden layer shenanagins for PFGRU use
        else:
            raise ValueError("Unknown architecture")
        return results           