# type: ignore
import pytest

import algos.multiagent.ppo as PPO
import algos.multiagent.NeuralNetworkCores.RADTEAM_core as RADTEAM_core

import numpy as np
import torch
import copy

# Helper functions
@pytest.fixture
def helpers():
    return Helpers

@pytest.fixture
def rada2c():
    return RADA2C


class RADA2C:
    @staticmethod
    def get_hiddens():
        return {0: ((
            torch.Tensor([[0.5960, 0.1294, 0.2328, 0.9597, 0.8043, 0.4710, 0.5223, 0.6586, 0.2308, 0.6779, 0.1494, 0.1594, 0.4553, 0.6984, 0.9075, 0.1983, 
                            0.0502, 0.6131,0.0998, 0.5853, 0.1522, 0.4587, 0.7399, 0.8631],
                [0.4578, 0.4242, 0.9543, 0.5092, 0.2156, 0.0329, 0.1127, 0.9901, 0.4490, 0.0687, 0.9534, 0.9364, 0.2560, 0.0931, 0.6246, 0.5947, 0.4765, 0.1815, 
                            0.6810, 0.5026, 0.5180, 0.0270, 0.1005, 0.6143],
                [0.6932, 0.0314, 0.3692, 0.9569, 0.2258, 0.0847, 0.5165, 0.3565, 0.7375, 0.6349, 0.5715, 0.0441, 0.8497, 0.6552, 0.7374, 0.3136, 0.2569, 0.5107,
                            0.0480, 0.0807, 0.9456, 0.8801, 0.9488, 0.5441],
                [0.9551, 0.7699, 0.9073, 0.3029, 0.9872, 0.9591, 0.9568, 0.8517, 0.5288, 0.5498, 0.5092, 0.2473, 0.5126, 0.8796, 0.2021, 0.2906, 0.6036, 0.0222,
                            0.0740, 0.3175, 0.7116, 0.0065, 0.8193, 0.1974],
                [0.3525, 0.1181, 0.7855, 0.4840, 0.0300, 0.5698, 0.4288, 0.5912, 0.0497, 0.0213, 0.1170, 0.7829, 0.4659, 0.4232, 0.3393, 0.7283, 0.8030, 0.4923,
                            0.7269, 0.3998, 0.6015, 0.5285, 0.8133, 0.1119],
                [0.2192, 0.1234, 0.4010, 0.9795, 0.1758, 0.5957, 0.4006, 0.5597, 0.2715, 0.2462, 0.2833, 0.1379, 0.4604, 0.1961, 0.8498, 0.6885, 0.8273, 0.7247,
                            0.2719, 0.3894, 0.0391, 0.4743, 0.8216, 0.0468],
                [0.0902, 0.9550, 0.0102, 0.5335, 0.3594, 0.2101, 0.6089, 0.1407, 0.6335, 0.0962, 0.1255, 0.3371, 0.9254, 0.3853, 0.9351, 0.6343, 0.6703, 0.4213,
                            0.8829, 0.7195, 0.6433, 0.5683, 0.7538, 0.2140],
                [0.0355, 0.4612, 0.6198, 0.9220, 0.4047, 0.7279, 0.2587, 0.9709, 0.4811, 0.8858, 0.6023, 0.5551, 0.3801, 0.5044, 0.7103, 0.5401, 0.2607, 0.0991,
                            0.1617, 0.4613, 0.3447, 0.9495, 0.1033, 0.4312],
                [0.8247, 0.6928, 0.4792, 0.5099, 0.1597, 0.8334, 0.9597, 0.6380, 0.6075, 0.4614, 0.0485, 0.5877, 0.7488, 0.6350, 0.3500, 0.9496, 0.0471, 0.7085,
                            0.6833, 0.4187, 0.7860, 0.0172, 0.2952, 0.7459],
                [0.3342, 0.5645, 0.6029, 0.2024, 0.2107, 0.8261, 0.3797, 0.3645, 0.2964, 0.0873, 0.0649, 0.5609, 0.2923, 0.8861, 0.8928, 0.8840, 0.5392, 0.2097,
                            0.5651, 0.0418, 0.3697, 0.7854, 0.6541, 0.4903],
                [0.6171, 0.2636, 0.6777, 0.0546, 0.8745, 0.4127, 0.8338, 0.4991, 0.7509, 0.0381, 0.8575, 0.0836, 0.1720, 0.6040, 0.8471, 0.1805, 0.7351, 0.4874,
                            0.9166, 0.6779, 0.5450, 0.9221, 0.0336, 0.8973],
                [0.1363, 0.1415, 0.7300, 0.1735, 0.7214, 0.0174, 0.3437, 0.6226, 0.2300, 0.3422, 0.1989, 0.5739, 0.3921, 0.7452, 0.4018, 0.6545, 0.4463, 0.1193,
                            0.1393, 0.6111, 0.4141, 0.5470, 0.3661, 0.8666],
                [0.2485, 0.0947, 0.3281, 0.6861, 0.1122, 0.4404, 0.0820, 0.9033, 0.9078, 0.4609, 0.8323, 0.0568, 0.2724, 0.9620, 0.7255, 0.9361, 0.5811, 0.2602,
                            0.1794, 0.7487, 0.7893, 0.5693, 0.7754, 0.5455],
                [0.8735, 0.6869, 0.6180, 0.1678, 0.0366, 0.7461, 0.9503, 0.1300, 0.3809, 0.6720, 0.0720, 0.9623, 0.5840, 0.6259, 0.0625, 0.6601, 0.2188, 0.1459,
                            0.5180, 0.1346, 0.1483, 0.3696, 0.3310, 0.9422],
                [0.0561, 0.6852, 0.2638, 0.2834, 0.9459, 0.4741, 0.1654, 0.8378, 0.2885, 0.9453, 0.4173, 0.3663, 0.3763, 0.7589, 0.5963, 0.1762, 0.9766, 0.3712,
                            0.6327, 0.3391, 0.0582, 0.3220, 0.1224, 0.6125],
                [0.9044, 0.3853, 0.6692, 0.7209, 0.7927, 0.0635, 0.1556, 0.9033, 0.2111, 0.3856, 0.6970, 0.7816, 0.2915, 0.8947, 0.3233, 0.8913, 0.0556, 0.9911,
                            0.0550, 0.7025, 0.7594, 0.4900, 0.8025, 0.0959],
                [0.8081, 0.3924, 0.2132, 0.5028, 0.9025, 0.2942, 0.0811, 0.8372, 0.1672, 0.3709, 0.7024, 0.3603, 0.7351, 0.1989, 0.1013, 0.5510, 0.6296, 0.3238,
                            0.9628, 0.0118, 0.5755, 0.3488, 0.4129, 0.1795],
                [0.3267, 0.6291, 0.4864, 0.9485, 0.9734, 0.1139, 0.8251, 0.6974, 0.9253, 0.6829, 0.6008, 0.4429, 0.4744, 0.1328, 0.5856, 0.8829, 0.9203, 0.5340,
                            0.5392, 0.0831, 0.3062, 0.6378, 0.6559, 0.2433],
                [0.5509, 0.3371, 0.5997, 0.2519, 0.0466, 0.0775, 0.6734, 0.4620, 0.0787, 0.5320, 0.3250, 0.2687, 0.3961, 0.0104, 0.1973, 0.9008, 0.9272, 0.2958,
                            0.0828, 0.2144, 0.4987, 0.9692, 0.3491, 0.2862],
                [0.1933, 0.8450, 0.7052, 0.4317, 0.9019, 0.3616, 0.3053, 0.1156, 0.5733, 0.1063, 0.7276, 0.6964, 0.2614, 0.4067, 0.5532, 0.8007, 0.8049, 0.4303,
                            0.0303, 0.5492, 0.2844, 0.5230, 0.0384, 0.9787],
                [0.7085, 0.6480, 0.6395, 0.0513, 0.2986, 0.6069, 0.0857, 0.0262, 0.9149, 0.7031, 0.7239, 0.8703, 0.8567, 0.9068, 0.9105, 0.3832, 0.7900, 0.1645,
                            0.5511, 0.0934, 0.0351, 0.6164, 0.3577, 0.1925],
                [0.2495, 0.1008, 0.5973, 0.2076, 0.1264, 0.9087, 0.4625, 0.6989, 0.8590, 0.2455, 0.8400, 0.7244, 0.2658, 0.5239, 0.1844, 0.5176, 0.5830, 0.2754,
                            0.2787, 0.4096, 0.6550, 0.4370, 0.5867, 0.7202],
                [0.4837, 0.6402, 0.3507, 0.7754, 0.4875, 0.8679, 0.5999, 0.3756, 0.3099, 0.3432, 0.0922, 0.7282, 0.7010, 0.9750, 0.9760, 0.1892, 0.4242, 0.6436,
                            0.5828, 0.7985, 0.6023, 0.0987, 0.4684, 0.7116],
                [0.5054, 0.4755, 0.1035, 0.7480, 0.9886, 0.0395, 0.6889, 0.5548, 0.1583, 0.9330, 0.1642, 0.5580, 0.6136, 0.5736, 0.9036, 0.4497, 0.2225, 0.6623,
                            0.2590, 0.3041, 0.4865, 0.0995, 0.5726, 0.0050],
                [0.9828, 0.5563, 0.9650, 0.1408, 0.1819, 0.8570, 0.0189, 0.6947, 0.0684, 0.7809, 0.1326, 0.9848, 0.9521, 0.4097, 0.3360, 0.8250, 0.7503, 0.8707,
                            0.8649, 0.8670, 0.9437, 0.4103, 0.1327, 0.7052],
                [0.7527, 0.2064, 0.2631, 0.6846, 0.4070, 0.6834, 0.0221, 0.1969, 0.6166, 0.6701, 0.4003, 0.6246, 0.6919, 0.2010, 0.8798, 0.7067, 0.9341, 0.4816,
                            0.8076, 0.3055, 0.3024, 0.0820, 0.8494, 0.6428],
                [0.9802, 0.9807, 0.3355, 0.6394, 0.9088, 0.1824, 0.9019, 0.5206, 0.0117, 0.0405, 0.6681, 0.2600, 0.7696, 0.8736, 0.6628, 0.4516, 0.5631, 0.6748,
                            0.8669, 0.7211, 0.0550, 0.7768, 0.4471, 0.5403],
                [0.1939, 0.9231, 0.5315, 0.6322, 0.9908, 0.0830, 0.8561, 0.6448, 0.0369, 0.1597, 0.5459, 0.2420, 0.0246, 0.4307, 0.6403, 0.1936, 0.1941, 0.7140,
                            0.4071, 0.0759, 0.0520, 0.2342, 0.0702, 0.7840],
                [0.6727, 0.2999, 0.2255, 0.8004, 0.6255, 0.1189, 0.6252, 0.7828, 0.5363, 0.6993, 0.4551, 0.5287, 0.2608, 0.6326, 0.1266, 0.2737, 0.2516, 0.2361,
                            0.9247, 0.0781, 0.5805, 0.6635, 0.3370, 0.8458],
                [0.5589, 0.3120, 0.0020, 0.1425, 0.4681, 0.9565, 0.7500, 0.6574, 0.3452, 0.3766, 0.6318, 0.0815, 0.5963, 0.5182, 0.6264, 0.3243, 0.0161, 0.6879,
                            0.5780, 0.1477, 0.6669, 0.8241, 0.4046, 0.0208],
                [0.2687, 0.5057, 0.5632, 0.1090, 0.1089, 0.4890, 0.3656, 0.0614, 0.1814, 0.8366, 0.8600, 0.2652, 0.1350, 0.0921, 0.3481, 0.7560, 0.3569, 0.7932,
                            0.5448, 0.1984, 0.7435, 0.1496, 0.0568, 0.2901],
                [0.4884, 0.5162, 0.4743, 0.8506, 0.8228, 0.7208, 0.1058, 0.9282, 0.3453, 0.9952, 0.9217, 0.7645, 0.1634, 0.7962, 0.7447, 0.1284, 0.5818, 0.5034,
                            0.5228, 0.1321, 0.8705, 0.1641, 0.7962, 0.4627],
                [0.4202, 0.7798, 0.3932, 0.4204, 0.5275, 0.6536, 0.1834, 0.1285, 0.0824, 0.3344, 0.8870, 0.8475, 0.7817, 0.7075, 0.0318, 0.8868, 0.7451, 0.9834,
                            0.4974, 0.6245, 0.6255, 0.7946, 0.8496, 0.9010],
                [0.9652, 0.3956, 0.1050, 0.1767, 0.0593, 0.0510, 0.8843, 0.7289, 0.3024, 0.0954, 0.4380, 0.2082, 0.6943, 0.7368, 0.4838, 0.8985, 0.8796, 0.8208,
                            0.1209, 0.3601, 0.4848, 0.9199, 0.1142, 0.4542],
                [0.7668, 0.0715, 0.7863, 0.9822, 0.2678, 0.5350, 0.6979, 0.5892, 0.0344, 0.9681, 0.3370, 0.6006, 0.4645, 0.0732, 0.3919, 0.3588, 0.7970, 0.7278,
                            0.9280, 0.2555, 0.0209, 0.7351, 0.6269, 0.7913],
                [0.6178, 0.2074, 0.6927, 0.0072, 0.7748, 0.3817, 0.2500, 0.9313, 0.6957, 0.9541, 0.5181, 0.7388, 0.4153, 0.5241, 0.7453, 0.5096, 0.1079, 0.7967,
                            0.4078, 0.0411, 0.8942, 0.8385, 0.4225, 0.1201],
                [0.8529, 0.4536, 0.5305, 0.7889, 0.4223, 0.5468, 0.4148, 0.9753, 0.7650, 0.4372, 0.0011, 0.4322, 0.5818, 0.8112, 0.8783, 0.1995, 0.8088, 0.5469,
                            0.4735, 0.4295, 0.1186, 0.8527, 0.7372, 0.8099],
                [0.7037, 0.3475, 0.8436, 0.5441, 0.1537, 0.3008, 0.6778, 0.8959, 0.5664, 0.7971, 0.7920, 0.5873, 0.4904, 0.1903, 0.5416, 0.3995, 0.3103, 0.7504,
                            0.9271, 0.6132, 0.4785, 0.5563, 0.7996, 0.4690],
                [0.9187, 0.6374, 0.3682, 0.4944, 0.9383, 0.1878, 0.8060, 0.8385, 0.5846, 0.7997, 0.4994, 0.0585, 0.8069, 0.9117, 0.3499, 0.6382, 0.8731, 0.4008,
                            0.4192, 0.6609, 0.4299, 0.2359, 0.1653, 0.4489],
                [0.3780, 0.5182, 0.3467, 0.7116, 0.9659, 0.2994, 0.4209, 0.8733, 0.2753, 0.2569, 0.1401, 0.8685, 0.1660, 0.2229, 0.3610, 0.9913, 0.3179, 0.0335,
                            0.2476, 0.8223, 0.7824, 0.7181, 0.9983, 0.2542]]), 
            torch.Tensor([[-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889],
                [-3.6889]])),
            torch.Tensor([[[ 0.0443,  0.1509,  0.0280, -0.1628,  0.0428,  0.1115, -0.1386, -0.0474,  0.1737,  0.1868, -0.1770,  0.1078,  0.0150,  0.1685,
                    0.0116, -0.0410, -0.0254, -0.1502,  0.1679,  0.1237,  0.0376, -0.0082, -0.0871,  0.0100]]]))}

    def get_init():
        return {   
            'obs_dim': 11, 'act_dim': 8,'hidden_sizes_pol': [[32]], 'hidden_sizes_val': [[32]], 'hidden_sizes_rec': [24], 
            'hidden': [[24]], 'net_type': 'rnn', 'batch_s': 1, 'seed': 0, 'pad_dim': 2
            }                   

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
        heatmap_stack = RADTEAM_core.HeatMaps(torch.tensor([0]), torch.tensor([1]))
        test = np.zeros((11,), dtype=np.float32) # For comparison with empty
        
        # Store 1st set
        buffer.store(
            obs=obs,
            act=act,
            rew=rew,
            val=val,
            logp=logp,
            src=src,
            full_observation=full_obs,
            heatmap_stacks=heatmap_stack,
            terminal=False
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
        
        # TODO write tests for prio_memory mode
        # for agent_id, agent_obs in full_obs.items():
        #     assert np.array_equal(buffer.full_observation_buffer[0][agent_id], agent_obs)     
        # assert buffer.full_observation_buffer[0]['terminal'] == False
            
        assert torch.equal(buffer.heatmap_buffer['actor'][0], heatmap_stack.actor)
        assert torch.equal(buffer.heatmap_buffer['critic'][0], heatmap_stack.critic)
        
            
        # Check remainder are zeros        
        for i in range(1, init_parameters['max_size']):
            assert np.array_equal(buffer.obs_buf[i], test)
            assert buffer.act_buf[i] == 0
            assert buffer.rew_buf[i] == 0.0
            assert buffer.val_buf[i] == 0.0
            assert np.array_equal(buffer.source_tar[i], np.zeros((2,), dtype=np.float32))   
            assert buffer.logp_buf[i] == 0.0

            # for id in range(1, init_parameters['number_agents']):
            #     assert np.array_equal(buffer.full_observation_buffer[i][id], test)

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
            full_observation=full_obs,
            heatmap_stacks=heatmap_stack,
            terminal=False            
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
        
        # for agent_id, agent_obs in full_obs.items():
        #     assert np.array_equal(buffer.full_observation_buffer[1][agent_id], agent_obs)     
        # assert buffer.full_observation_buffer[1]['terminal'] == False            
            
        assert torch.equal(buffer.heatmap_buffer['actor'][1], heatmap_stack.actor)
        assert torch.equal(buffer.heatmap_buffer['critic'][1], heatmap_stack.critic)
        
            
        # Check remainder are zeros        
        for i in range(2, init_parameters['max_size']):
            assert np.array_equal(buffer.obs_buf[i], test)
            assert buffer.act_buf[i] == 0
            assert buffer.rew_buf[i] == 0.0
            assert buffer.val_buf[i] == 0.0
            assert np.array_equal(buffer.source_tar[i], np.zeros((2,), dtype=np.float32))   
            assert buffer.logp_buf[i] == 0.0

            # for id in range(1, init_parameters['number_agents']):
            #     assert np.array_equal(buffer.full_observation_buffer[i][id], test)        

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
                full_observation=full_obs,
                heatmap_stacks=heatmap_stack,
                terminal=False                      
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
                        
        obs = np.zeros((11,), dtype=np.float32)
        
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
            full_observation={0: np.zeros(11,), 1: np.zeros(11,)},
            terminal=False,
            heatmap_stacks= RADTEAM_core.HeatMaps(torch.tensor(obs), torch.tensor(obs))            
        )
        
        # 2nd step:
        buffer.store(
            obs=np.zeros((11,), dtype=np.float32),
            act=0,
            rew=test['rewards'][1],
            val=test['values'][1],
            logp=0,
            src=np.zeros((1,2), dtype=np.float32),
            full_observation={0: np.zeros(11,), 1: np.zeros(11,)},
            terminal=False,
            heatmap_stacks= RADTEAM_core.HeatMaps(torch.tensor(obs), torch.tensor(obs))            
        )  
        # 3rd step:
        buffer.store(
            obs=np.zeros((11,), dtype=np.float32),
            act=0,
            rew=test['rewards'][2],
            val=test['values'][2],
            logp=0,
            src=np.zeros((1,2), dtype=np.float32),
            full_observation={0: np.zeros(11,), 1: np.zeros(11,)},
            terminal=False,
            heatmap_stacks= RADTEAM_core.HeatMaps(torch.tensor(obs), torch.tensor(obs))                 
        )               
              
        buffer.GAE_advantage_and_rewardsToGO(last_state_value=test['last_val'])
        
        for result, to_test in zip(manual_rewardsToGo, buffer.ret_buf):
            assert result == pytest.approx(to_test)         
            
        for result, to_test in zip(manual_gae, buffer.adv_buf):
            assert result == pytest.approx(to_test)                                                             
        
    def test_get(self, init_parameters)-> None:
        buffer = PPO.PPOBuffer(**init_parameters)    
        
        map = RADTEAM_core.MapsBuffer(observation_dimension=11, number_of_agents=2,steps_per_episode=5)
                
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
            last_val = -0.26634163,
            terminal=False               
        )     
        
        # Get mapstack
        stack = map.observation_to_map(test['full_obs'], id=0)
        actor_map_stack: torch.Tensor = torch.stack(
            [torch.tensor(stack[0]), torch.tensor(stack[1]), torch.tensor(stack[2]), torch.tensor(stack[3]),  torch.tensor(stack[4])]
        )
        critic_map_stack: torch.Tensor = torch.stack(
            [torch.tensor(stack[2]), torch.tensor(stack[3]),  torch.tensor(stack[4]), torch.tensor(stack[5])]
        )            
        
        # Add single batch tensor dimension for action selection
        batched_actor_mapstack: torch.Tensor = torch.unsqueeze(actor_map_stack, dim=0)      
        batched_critic_mapstack: torch.Tensor = torch.unsqueeze(critic_map_stack, dim=0)           
        
        test['heat'] = RADTEAM_core.HeatMaps(batched_actor_mapstack, batched_critic_mapstack)
            
        # Prime buffer
        # 1st step: 
        buffer.store(
            obs=test['obs'],
            act=test['act'][0],
            rew=test['rewards'][0],
            val=test['values'][0],
            logp=test['logp'][0],
            src=test['src'],
            full_observation=test['full_obs'],
            terminal=test['terminal'],
            heatmap_stacks=test['heat']            
        )
        # 2nd step:
        buffer.store(
            obs=test['obs'],
            act=test['act'][1],
            rew=test['rewards'][1],
            val=test['values'][1],
            logp=test['logp'][1],
            src=test['src'],
            full_observation=test['full_obs'],
            terminal=test['terminal'],
            heatmap_stacks=test['heat']                     
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
            'save_path': ['.', 'unit_test']
            }
                      
        return dict(
            id = 0,
            observation_space = 11,
            bp_args = bpargs,
            steps_per_epoch = 3,
            steps_per_episode = 2,
            number_of_agents = 2,
            env_height = 5,
            actor_critic_args = ac_kwargs,
            actor_critic_architecture = 'cnn'
        )
            
    def test_Init(self, init_parameters, rada2c):
        _ = PPO.AgentPPO(**init_parameters)
        
        rad_a2c_kwargs = rada2c.get_init()
        
        init_parameters['actor_critic_args'] = rad_a2c_kwargs
        init_parameters['actor_critic_architecture'] = 'rnn'
        
        
        _ = PPO.AgentPPO(**init_parameters)            
        # TODO add custom checks for different combos with CNN/RAD-A2C/Global Critic       
        
    def test_reduce_pfgru_training(self, init_parameters):
        AgentPPO = PPO.AgentPPO(**init_parameters)        
        assert AgentPPO.reduce_pfgru_iters == True
        assert AgentPPO.train_pfgru_iters == 15
        AgentPPO.reduce_pfgru_training()
        assert AgentPPO.reduce_pfgru_iters == False
        assert AgentPPO.train_pfgru_iters == 5
        
    def test_step(self, init_parameters, rada2c):
        ''' Wrapper between CNN and Train '''        
        hiddens = rada2c.get_hiddens() 
        # Test RAD-A2c
        rad_a2c_kwargs= rada2c.get_init()
        
        observations = {
            0: np.array([41.0, 0.42181818, 0.92181818, 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32), 
            1: np.array([41.0, 0.42181818, 0.92181818, 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
            }
        message = None
        
        # Test CNN
        AgentPPO = PPO.AgentPPO(**init_parameters)    
        
        agent_thoughts, heatmaps = AgentPPO.step(observations=observations, hidden=hiddens, message=message)
        
        assert heatmaps.actor.shape == torch.Size([1, 5, 28, 28])
        assert heatmaps.critic.shape == torch.Size([1, 4, 28, 28])
        assert agent_thoughts.action_logprob != None
        assert agent_thoughts.id != None
        assert agent_thoughts.state_value != None
        assert 0 <= agent_thoughts.action and agent_thoughts.action < int(8)        
        assert agent_thoughts.hiddens == None
        assert agent_thoughts.loc_pred == None
     
                
        rada2c_params = copy.deepcopy(init_parameters)
        rada2c_params['actor_critic_architecture'] = 'rnn'
        rada2c_params['actor_critic_args'] = rad_a2c_kwargs
        
        AgentPPO = PPO.AgentPPO(**rada2c_params)    
        
        agent_thoughts, heatmaps = AgentPPO.step(observations=observations, hidden=hiddens, message=message)
        
        assert heatmaps == None
        assert agent_thoughts.action_logprob != None
        assert agent_thoughts.id != None
        assert agent_thoughts.state_value != None
        assert 0 <= agent_thoughts.action and agent_thoughts.action < int(8)        
        assert agent_thoughts.hiddens != None
        assert agent_thoughts.loc_pred.shape == (2,)
        
        # Test invalid architecture 
        init_parameters['actor_critic_architecture'] = 'foo'
        with pytest.raises(ValueError):    
            AgentPPO = PPO.AgentPPO(**init_parameters)    
        
    def test_reset_agent(self, init_parameters, rada2c):
        hiddens = rada2c.get_hiddens()         

        observations = {
            0: np.array([41.0, 0.42181818, 0.92181818, 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32), 
            1: np.array([41.0, 0.42181818, 0.92181818, 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
            }
        message = None
        
        # Test CNN
        AgentPPO = PPO.AgentPPO(**init_parameters)    
        _  = AgentPPO.step(observations=observations, hidden=hiddens, message=message)
        assert AgentPPO.agent.reset_flag == 0
        assert AgentPPO.agent.maps.reset_flag == 1
        assert AgentPPO.agent.maps.tools.reset_flag == 1
                
        _ = AgentPPO.reset_agent()
        assert AgentPPO.agent.reset_flag == 1
        assert AgentPPO.agent.maps.reset_flag == 2
        assert AgentPPO.agent.maps.tools.reset_flag == 2

        # Test RAD-A2c
        rad_a2c_kwargs= rada2c.get_init()
        rada2c_params = copy.deepcopy(init_parameters)
        rada2c_params['actor_critic_architecture'] = 'rnn'
        rada2c_params['actor_critic_args'] = rad_a2c_kwargs        
        
        AgentPPO = PPO.AgentPPO(**rada2c_params)    
        _ = AgentPPO.step(observations=observations, hidden=hiddens, message=message)
        # TODO add check for RAD-A2C


        