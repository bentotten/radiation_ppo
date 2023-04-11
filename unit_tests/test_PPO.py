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
    @pytest.fixture
    def init_parameters(self)-> dict:
        ''' Set up initialization parameters needed to test discount_cumsum '''
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
        
    def test_GAE_advantage_and_rewardsToGO(self)-> None:        
        def generalized_advantage_estimate(gamma, lamb, done, rewards, values):
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

        def rewards_to_go2(rewards: np.array, gamma: float) -> np.array:
            """
                Calculates the sequence of discounted rewards-to-go.
                Args:
                    rewards: the sequence of observed rewards
                    gamma: the discount factor
                Returns:
                    discounted_rewards: the sequence of the rewards-to-go
            """
            discounted_rewards = np.empty_like(rewards, dtype=np.float)
            for i in range(rewards.shape[0]):
                gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=gamma)
                discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
                discounted_reward = np.sum(rewards[i:] * discounted_gammas)
                discounted_rewards[i] = discounted_reward
            return discounted_rewards
        
        def rewards_to_go3(rews):
            n = len(rews)
            rtgs = np.zeros_like(rews)
            for i in reversed(range(n)):
                rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
            return rtgs            

        init_parameters = dict(
            observation_dimension = 11,
            max_size = 3,
            max_episode_length = 2,
            number_agents = 2
        )
                                    
        buffer = PPO.PPOBuffer(**init_parameters)
          
        # Manual test variables
        gamma = 0.99
        lamb = 0.90
        done = np.asarray([False, False])
        values = np.asarray([-0.4, -0.4, 0.3])
        rewards =[-1, -1, -1]
        
        # Manually setup buffer
        # 1st step: 
        buffer.store(
            obs=np.zeros((11,), dtype=np.float32),
            act=0,
            rew=rewards[0],
            val=values[0],
            logp=0,
            src=np.zeros((1,2), dtype=np.float32),
            full_observation={0: np.zeros(11,), 1: np.zeros(11,)}
        )
        # 2nd step:
        buffer.store(
            obs=np.zeros((11,), dtype=np.float32),
            act=0,
            rew=rewards[1],
            val=values[1],
            logp=0,
            src=np.zeros((1,2), dtype=np.float32),
            full_observation={0: np.zeros(11,), 1: np.zeros(11,)}
        )  
        # 3rd step:
        buffer.store(
            obs=np.zeros((11,), dtype=np.float32),
            act=0,
            rew=rewards[2],
            val=values[2],
            logp=0,
            src=np.zeros((1,2), dtype=np.float32),
            full_observation={0: np.zeros(11,), 1: np.zeros(11,)}
        )                     
             
        # Get GAE and r2g
        manual_gae = generalized_advantage_estimate(gamma=gamma, lamb=lamb, done=done, rewards= np.asarray(rewards), values=values)     
        manual_rewardsToGo = rewards_to_go(batch_rews=rewards, gamma=gamma)
        manual_rewardsToGo2 = rewards_to_go2(rewards=np.asarray(rewards), gamma=gamma)
        manual_rewardsToGo3 = rewards_to_go3(rews=rewards)
        
        buffer.GAE_advantage_and_rewardsToGO(last_state_value=values[-1])

        for result, to_test in zip(manual_rewardsToGo, manual_rewardsToGo2):
            assert result == pytest.approx(to_test) 
            
        for result, to_test in zip(manual_rewardsToGo, buffer.ret_buf):
            assert result == pytest.approx(to_test)         
            
        for result, to_test in zip(manual_gae, buffer.adv_buf):
            assert result == pytest.approx(to_test)        
                            
        # Input: vector x,
        #     [x0,
        #     x1,
        #     x2]

        # Output:
        #     [x0 + discount * x1 + discount^2 * x2,
        #     x1 + discount * x2,
        #     x2]                                 
        
    
    def test_get(self, init_parameters)-> None:
        pass

    
# Classes
# PPOBuffer
# AgentPPO