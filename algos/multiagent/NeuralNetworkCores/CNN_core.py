'''
Implementation of "Target Localization using Multi-Agent Deep Reinforcement Learning with Proximal Policy Optimization" by Alagha et al.
Partially adapted from:
    - towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
    - github.com/nikhilbarhate99/PPO-PyTorch
    - https://towardsdatascience.com/a-graphic-guide-to-implementing-ppo-for-atari-games-5740ccbe3fbc
    - https://www.youtube.com/watch?v=KHZVXao4qXs 
    - https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl 

'''
from os import stat, path, mkdir, getcwd

from numpy import dtype
import numpy as np
import numpy.typing as npt

import scipy.signal

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torchinfo import summary

import pytorch_lightning as pl

from dataclasses import dataclass, field, asdict
from typing import Any, List, Tuple, Union, Literal, NewType, Optional, TypedDict, cast, get_args, Dict
from typing_extensions import TypeAlias

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.streamplot import Grid

# from epoch_logger import EpochLogger TODO not importing correctly

from gym_rad_search.envs import StepResult

# Maps
Point: TypeAlias = NewType("Point", tuple[float, float])  # Array indicies to access a GridSquare
Map: TypeAlias = NewType("Map", npt.NDArray[np.float32]) # 2D array that holds gridsquare values
CoordinateStorage: TypeAlias = NewType("Storage", list[dict, Point])

# Helpers
Shape: TypeAlias = int | tuple[int, ...]

DIST_TH = 110.0  # Detector-obstruction range measurement threshold in cm for inflating step size to obstruction

################################## Helper Functions and Classes ##################################
def combined_shape(length: int, shape: Optional[Shape] = None) -> Shape:
    if shape is None:
        return (length,)
    elif np.isscalar(shape):
        shape = cast(int, shape)
        return (length, shape)
    else:
        shape = cast(tuple[int, ...], shape)
        return (length, *shape)


def discount_cumsum(
    x: npt.NDArray[np.float64], discount: float
) -> npt.NDArray[np.float64]:
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


@dataclass()
class ActionChoice():
    id: int 
    action: torch.int # size (1)
    action_logprob: torch.float # size (1)
    state_value: torch.float # size(1)


################################## PPO Buffer ##################################
@dataclass
class PPOBuffer:
    # Required input
    observation_dimension: int  # Shape of state space aka how many elements in the observation returned from the environment
    max_size: int  # steps_per_epoch
    map_dimensions: tuple
    
    # Hyperparameters
    gamma: float = 0.99
    lamda: float = 0.90
    beta: float = 0.005
    
    # Path Tracking
    ptr: int = 0
    path_start_idx: int = 0

    # TODO should these use default factory?
    # Buffers
    states_buffer: npt.NDArray[np.float32] = field(init=False)
    actions_buffer: npt.NDArray[np.float32] = field(init=False)
    adv_buf: npt.NDArray[np.float32] = field(init=False)
    rewards_buffer: npt.NDArray[np.float32] = field(init=False)
    ret_buf: npt.NDArray[np.float32] = field(init=False)
    state_value_buffer: npt.NDArray[np.float32] = field(init=False)
    source_tar: npt.NDArray[np.float32] = field(init=False)
    logprobs_buffer: npt.NDArray[np.float32] = field(init=False)
    terminal_mask_buffer: npt.NDArray[np.bool] = field(init=False)

    # Additional buffers
    mapstacks: list = field(default_factory=list)  # TODO Change to numpy arrays like above  # TODO do we really want to store these or rerender them when needed for update?
    readings: Dict[Any, list] = field(default_factory=dict)  # TODO Change to numpy arrays like above
    coordinate_buffer: list = field(default_factory=list)  # TODO Change to numpy arrays like above
    episode_length_buffer: list = field(default_factory=list)      
    
    # Used for Location Prediction; 
    # TODO not implemented yet
    obs_win: npt.NDArray[np.float32] = field(init=False)
    obs_win_std: npt.NDArray[np.float32] = field(init=False)    

    """
    A buffer for storing histories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __post_init__(self):
        # TODO do these need to be made with default factory?
        self.states_buffer: npt.NDArray[np.float32] = np.zeros(
            combined_shape(self.max_size, self.observation_dimension), dtype=np.float32
        )
        self.actions_buffer: npt.NDArray[np.float32] = np.zeros(
            combined_shape(self.max_size), dtype=np.float32
        )
        self.adv_buf: npt.NDArray[np.float32] = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.rewards_buffer: npt.NDArray[np.float32] = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.ret_buf: npt.NDArray[np.float32] = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.state_value_buffer: npt.NDArray[np.float32] = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.source_tar: npt.NDArray[np.float32] = np.zeros(
            (self.max_size, 2), dtype=np.float32
        )
        self.logprobs_buffer: npt.NDArray[np.float32] = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.obs_win: npt.NDArray[np.float32] = np.zeros(
            self.observation_dimension, dtype=np.float32
        )
        self.obs_win_std: npt.NDArray[np.float32] = np.zeros(
            self.observation_dimension, dtype=np.float32
        )
        # TODO do we want to initialize these to 1?
        self.terminal_mask_buffer: npt.NDArray[np.int32] = np.full(
            self.max_size, 1, dtype=np.int32
        )

        # TODO Update these
        self.mapstacks: List = []  # TODO Change to numpy arrays like above
        self.readings: Dict[Any, list] = {} # For heatmap resampling
        self.coordinate_buffer: CoordinateStorage = []    
        self.episode_length_buffer = []
        
        ################################## set device ##################################
        print("============================================================================================")
        # set device to cpu or cuda
        self.device = torch.device('cpu')
        if(torch.cuda.is_available()): 
            self.device = torch.device('cuda:0') 
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(self.device)))
        else:
            print("Device set to : cpu")
        print("============================================================================================")

    def store(
        self,
        obs: npt.NDArray[Any],
        act: npt.NDArray[np.int32],
        rew: npt.NDArray[np.float32],
        val: npt.NDArray[np.float32],
        logp: npt.NDArray[np.float32],
        src: npt.NDArray[np.float32],
        terminal: npt.NDArray[np.bool],
    ) -> None:
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size
        self.states_buffer[self.ptr, :] = obs
        self.actions_buffer[self.ptr] = act
        self.rewards_buffer[self.ptr] = rew
        self.state_value_buffer[self.ptr] = val
        self.source_tar[self.ptr] = src
        self.logprobs_buffer[self.ptr] = logp
        self.terminal_mask_buffer = 0 if terminal else 1  # If terminal, multiple remaining values by 0 to mask out of calculation, as they are for future episodes.
        self.ptr += 1

    def store_episode_length( self, episode_length: npt.NDArray[np.int32]) -> None:
        ''' Only stores if episode has ended '''
        self.episode_length_buffer.append(episode_length)

    def compute_advantages(self, vals, rews, path_slice):
        ''' 
            uses rewards and value estimates from
            the whole trajectory to compute advantage estimates with GAE-Lambda 
        '''
        # the next two lines implement GAE-Lambda advantage calculation
        # gamma determines scale of value function, introduces bias regardless of VF accuracy
        # lambda introduces bias when VF is inaccurate
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        return discount_cumsum(deltas, self.gamma * self.lamda)      

    def finish_path_and_compute_advantages(self, last_val: int = 0) -> None:
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rewards_buffer[path_slice], last_val)
        vals = np.append(self.state_value_buffer[path_slice], last_val)

        self.adv_buf[path_slice] = self.compute_advantages(rews, vals, path_slice)

        # the next line computes rewards-to-go, to be targets for the value function
        # TODO discounted returns?
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get_buffers_for_epoch_and_reset(self, logger = None) -> dict[str, Union[torch.Tensor, list]]:  # logger will eventually be EpochLogger from epoch_logger
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer and clears the 
        map and reading buffers
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        
        # the next two lines implement the advantage normalization trick
        adv_mean = self.adv_buf.mean()
        adv_std = self.adv_buf.std()
        self.adv_buf: npt.NDArray[np.float32] = (self.adv_buf - adv_mean) / adv_std
        
        # TODO what is this and why is it commented out?
        # ret_mean, ret_std = self.ret_buf.mean(), self.ret_buf.std()
        # self.ret_buf = (self.ret_buf) / ret_std
        # obs_mean, obs_std = self.states_buffer.mean(), self.states_buffer.std()
        # self.states_buffer_std_ind[:,1:] = (self.states_buffer[:,1:] - obs_mean[1:]) / (obs_std[1:])

        epLens: list[int] = self.episode_length_buffer
        numEps = len(epLens)
        epLenTotal = sum(epLens)
        
        data = dict(
            maps=torch.stack(self.mapstacks), # TODO do we need states if we already have the maps?
            obs=torch.as_tensor(self.states_buffer, dtype=torch.float32), # TODO do we need states if we already have an old-map stack?
            act=torch.as_tensor(self.actions_buffer, dtype=torch.float32),
            ret=torch.as_tensor(self.ret_buf, dtype=torch.float32),
            adv=torch.as_tensor(self.adv_buf, dtype=torch.float32),
            logp=torch.as_tensor(self.logprobs_buffer, dtype=torch.float32),
            loc_pred=torch.as_tensor(self.obs_win_std, dtype=torch.float32),
            ep_len=torch.as_tensor(epLenTotal, dtype=torch.float32),
            ep_form = []
        )

        epLenSize = (
            # If they're equal then we don't need to do anything
            # Otherwise we need to add one to make sure that numEps is the correct size
            numEps + int(epLenTotal != len(self.states_buffer))
        )
        states_buffer = np.hstack(
            (
                self.states_buffer,
                self.adv_buf[:, None],
                self.ret_buf[:, None],
                self.logprobs_buffer[:, None],
                self.actions_buffer[:, None],
                self.source_tar,
            )
        )
        epForm: list[list[torch.Tensor]] = [[] for _ in range(epLenSize)]
        slice_b: int = 0
        slice_f: int = 0
        jj: int = 0
        
        # TODO: This is essentially just a sliding window over states_buffer; use a built-in function to do this
        for ep_i in epLens:
            slice_f += ep_i
            epForm[jj].append(
                torch.as_tensor(states_buffer[slice_b:slice_f], dtype=torch.float32)
            )
            slice_b += ep_i
            jj += 1
        if slice_f != len(self.states_buffer):
            epForm[jj].append(
                torch.as_tensor(states_buffer[slice_f:], dtype=torch.float32)
            )

        data["ep_form"] = epForm
        
        # Reset readings, mapstacks, and buffer pointers
        # TODO should this be a clear() instead?
        self.ptr, self.path_start_idx = 0, 0
        del self.mapstacks[:]
        self.readings.clear()        

        return data

    def clear(self):
        # Reset readings, mapstacks, and buffer pointers
        self.ptr, self.path_start_idx = 0, 0
        del self.mapstacks[:]
        del self.coordinate_buffer[:]     
        del self.episode_length_buffer[:]
        self.readings.clear()
        
        # Reset numpy arrays
        # sanity_check
        hold_s = self.states_buffer.shape
        hold_a = self.actions_buffer.shape
        hold_adv = self.adv_buf.shape
        hold_r = self.rewards_buffer.shape
        hold_ret = self.ret_buf.shape
        hold_sv = self.state_value_buffer.shape
        hold_src = self.source_tar.shape
        hold_l = self.logprobs_buffer.shape
        hold_t = self.terminal_mask_buffer.shape
        hold_ow = self.obs_win.shape
        hold_owstd = self.obs_win_std.shape
        
        self.states_buffer[:] = 0.0
        self.actions_buffer[:] = 0.0
        self.adv_buf[:] = 0.0
        self.rewards_buffer[:] = 0.0
        self.ret_buf[:] = 0.0
        self.state_value_buffer[:] = 0.0
        self.source_tar[:] = 0.0
        self.logprobs_buffer[:] = 0.0
        self.terminal_mask_buffer[:] = 0.0
        
        # TODO move to a unit test
        # Add asserts for del self.mapstacks[:] and self.readings.clear()
        assert hold_s == self.states_buffer.shape
        assert hold_a ==self.actions_buffer.shape
        assert hold_adv == self.adv_buf.shape
        assert hold_r == self.rewards_buffer.shape
        assert hold_ret == self.ret_buf.shape
        assert hold_sv == self.state_value_buffer.shape
        assert hold_src == self.source_tar.shape
        assert hold_l == self.logprobs_buffer.shape
        assert hold_t == self.terminal_mask_buffer.shape
        assert hold_ow == self.obs_win.shape
        assert hold_owstd == self.obs_win_std.shape   
        # TODO add episode length buffer check     
    
    
@dataclass()
class MapsBuffer:        
    '''
    5 maps: 
        1. Location Map: a 2D matrix showing the individual agent's location.
        2. Map of Other Locations: a grid showing the number of agents located in each grid element (excluding current agent).
        3. Readings map: a grid of the last reading collected in each grid square - unvisited squares are given a reading of 0.
        4. Visit Counts Map: a grid of the number of visits to each grid square from all agents combined.
        5. Obstacles Map: a grid of how far from an obstacle each agent was when they detected it
    '''
    # Inputs
    observation_dimension: int  # Shape of state space aka how many elements in the observation returned from the environment
    max_size: int  # steps_per_epoch   
        
    # Hyperparameters
    gamma: float = 0.99
    lamda: float = 0.90
    beta: float = 0.005
            
    # Parameters
    grid_bounds: tuple = field(default_factory= lambda: (1,1))  # Initial grid bounds for state x and y coordinates. For RADPPO, these are scaled to be below 0, so bounds are 1x1
    resolution_accuracy: int = field(default=100) # How much to multiply grid bounds and state coordinates by. 100 will return to full accuracy for RADPPO
    obstacle_state_offset: int = field(default=3) # Number of initial elements in state return that do not indicate there is an obstacle. First element is intensity, second two are x and y coords
    
    # Initialized elsewhere
    x_limit_scaled: int = field(init=False)  # maximum x value in maps
    y_limit_scaled: int = field(init=False)  # maximum y value in maps
    map_dimensions: Tuple = field(init=False)  # Scaled dimensions of each map - used to create the CNNs
    
    # TODO make work with max_step_count so boundaries dont need to be enforced on grid. Basically take the grid bounds and add the max step count to make the map sizes
    
    # Maps
    location_map: Map = field(init=False)  # Location Map: a 2D matrix showing the individual agent's location.
    others_locations_map: Map = field(init=False)  # Map of Other Locations: a grid showing the number of agents located in each grid element (excluding current agent).
    readings_map: Map = field(init=False)  # Readings map: a grid of the last reading collected in each grid square - unvisited squares are given a reading of 0.
    visit_counts_map: Map = field(init=False) # Visit Counts Map: a grid of the number of visits to each grid square from all agents combined.
    obstacles_map: Map = field(init=False) # bstacles Map: a grid of how far from an obstacle each agent was when they detected it
    
    # Buffers for PPO update to neural networks at end of epoch
    buffer: PPOBuffer = field(init=False)

    def __post_init__(self):      
        # Scaled maps

        self.map_dimensions = (int(self.grid_bounds[0] * self.resolution_accuracy), int(self.grid_bounds[1] * self.resolution_accuracy))
        self.x_limit_scaled: int = self.map_dimensions[0]
        self.y_limit_scaled: int = self.map_dimensions[1] 
        
        #self.buffer: RolloutBuffer = field(default_factory=lambda: RolloutBuffer())
        # TODO does this need to be made with default factory?
        self.buffer: PPOBuffer = PPOBuffer(
                observation_dimension=self.observation_dimension, 
                max_size=self.max_size, 
                gamma = self.gamma, 
                lamda = self.lamda, 
                beta = self.beta,
                map_dimensions=self.map_dimensions
            )        
        
        self.clear()

    def clear(self):
        self.location_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  # TODO rethink this, this is very slow - potentially change to torch?
        self.others_locations_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  # TODO rethink this, this is very slow
        self.readings_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  # TODO rethink this, this is very slow
        self.visit_counts_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  # TODO rethink this, this is very slow
        self.obstacles_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  # TODO rethink this, this is very slow
        self.buffer.clear()
        
    def observation_to_map(self, observation: dict[int, StepResult], id: int
                     ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:  
        '''
        observation: observations from environment for all agents
        id: ID of agent to reference in observation object 
        Returns a tuple of 2d map arrays
        '''
        
        # TODO Remove redundant calculations
        
        # Process observation for current agent's locations map
        scaled_coordinates = (int(observation[id][1] * self.resolution_accuracy), int(observation[id][2] * self.resolution_accuracy))        
        # Capture current and reset previous location
        if self.buffer.coordinate_buffer:
            last_state = self.buffer.coordinate_buffer[-1][id]
            scaled_last_coordinates = (int(last_state[0] * self.resolution_accuracy), int(last_state[1] * self.resolution_accuracy))
            x_old = int(scaled_last_coordinates[0])
            y_old = int(scaled_last_coordinates[1])
            self.location_map[x_old][y_old] -= 1 # In case agents are at same location, usually the start-point
            assert self.location_map[x_old][y_old] > -1, "location_map grid coordinate reset where agent was not present. The map location that was reset was already at 0."
        # Set new location
        x = int(scaled_coordinates[0])
        y = int(scaled_coordinates[1])
        self.location_map[x][y] = 1.0 
        
        # Process observation for other agent's locations map

        for other_agent_id in observation:
            # Do not add current agent to other_agent map
            if other_agent_id != id:
                others_scaled_coordinates = (int(observation[other_agent_id][1] * self.resolution_accuracy), int(observation[other_agent_id][2] * self.resolution_accuracy))
                # Capture current and reset previous location
                if self.buffer.coordinate_buffer:
                    last_state = self.buffer.coordinate_buffer[-1][other_agent_id]
                    scaled_last_coordinates = (int(last_state[0] * self.resolution_accuracy), int(last_state[1] * self.resolution_accuracy))
                    x_old = int(scaled_last_coordinates[0])
                    y_old = int(scaled_last_coordinates[1])
                    self.others_locations_map[x_old][y_old] -= 1 # In case agents are at same location, usually the start-point
                    assert self.others_locations_map[x_old][y_old] > -1, "Location map grid coordinate reset where agent was not present"
        
                # Set new location
                x = int(others_scaled_coordinates[0])
                y = int(others_scaled_coordinates[1])
                self.others_locations_map[x][y] += 1.0  # Initial agents begin at same location        
                 
        # Process observation for readings_map
        for agent_id in observation:
            scaled_coordinates = (int(observation[agent_id][1] * self.resolution_accuracy), int(observation[agent_id][2] * self.resolution_accuracy))            
            x = int(scaled_coordinates[0])
            y = int(scaled_coordinates[1])
            unscaled_coordinates = (observation[agent_id][1], observation[agent_id][2])
                        
            assert len(self.buffer.readings[unscaled_coordinates]) > 0
            # TODO onsider using a particle filter for resampling            
            estimated_reading = np.median(self.buffer.readings[unscaled_coordinates])
            self.readings_map[x][y] = estimated_reading  # Initial agents begin at same location

        # Process observation for visit_counts_map
        for agent_id in observation:
            scaled_coordinates = (int(observation[agent_id][1] * self.resolution_accuracy), int(observation[agent_id][2] * self.resolution_accuracy))            
            x = int(scaled_coordinates[0])
            y = int(scaled_coordinates[1])

            self.visit_counts_map[x][y] += 1
            
        # Process observation for obstacles_map 
        # Agent detects obstructions within 110 cm of itself
        for agent_id in observation:
            scaled_agent_coordinates = (int(observation[agent_id][1] * self.resolution_accuracy), int(observation[agent_id][2] * self.resolution_accuracy))            
            if np.count_nonzero(observation[agent_id][self.obstacle_state_offset:]) > 0:
                indices = np.flatnonzero(observation[agent_id][self.obstacle_state_offset::]).astype(int)
                for index in indices:
                    real_index = int(index + self.obstacle_state_offset)
                    
                    # Inflate to actual distance, then convert and round with resolution_accuracy
                    inflated_distance = (-(observation[agent_id][real_index] * DIST_TH - DIST_TH))
                    
                    # scaled_obstacle_distance = int(inflated_distance / self.resolution_accuracy)
                    # step: int = field(init=False)
                    # match index:
                        # Access the obstacle detection portion of observation and see what direction an obstacle is in
                        # These offset indexes correspond to:
                        # 0: left
                        # 1: up and left
                        # 2: up
                        # 3: up and right
                        # 4: right
                        # 5: down and right
                        # 6: down
                        # 7: down and left                    
                    #     # 0: Left
                    #     case 0:
                    #         step = (-1 ,0)
                    #     # 1: up and left
                    #     case 1:
                    #         step = (-scaled_obstacle_distance, scaled_obstacle_distance)
                    #     # 2: up
                    #     case 2:
                    #         step = (0, scaled_obstacle_distance)
                    #     # 3: up and right
                    #     case 3:
                    #         step = (scaled_obstacle_distance, scaled_obstacle_distance)
                    #     # 4: right
                    #     case 4:
                    #         step = (scaled_obstacle_distance, 0)                        
                    #     # 5: down and right
                    #     case 5:
                    #         step = (scaled_obstacle_distance, -scaled_obstacle_distance)                           
                    #     # 6: down
                    #     case 6:
                    #         step = (0, -scaled_obstacle_distance)                           
                    #     # 7: down and left
                    #     case 7:
                    #         step = (-scaled_obstacle_distance, -scaled_obstacle_distance)                                                     
                    #     case _:
                    #         raise Exception('Obstacle index is not within valid [0,7] range.')                         
                    # x = int(scaled_agent_coordinates[0] + step[0])
                    # y = int(scaled_agent_coordinates[1] + step[1])
                    x = int(scaled_coordinates[0])
                    y = int(scaled_coordinates[1])
                    
                    # Semi-arbritrary, but should make the number higher as the agent gets closer to the object, making heatmap look more correct
                    self.obstacles_map[x][y] = DIST_TH - inflated_distance
        
        return self.location_map, self.others_locations_map, self.readings_map, self.visit_counts_map, self.obstacles_map


#TODO make a reset function, similar to self.ac.reset_hidden() in RADPPO
class Actor(nn.Module):
    def __init__(self, map_dim, state_dim, batches: int=6, map_count: int=5, action_dim: int=5, global_critic: bool=False):
        super(Actor, self).__init__()
        
        ''' Actor Input tensor shape: (batch size, number of channels, height of grid, width of grid)
                1. batch size: 1
                2. (map_count) number of channels: 5 input maps
                3. Height: grid height
                4. Width: grid width
            
                5 maps: 
                    1. Location Map: a 2D matrix showing the agents location.
                    2. Map of Other Locations: a 2D matrix showing the number of agents located in each grid element (excluding current agent).
                    3. Readings map: a 2D matrix showing the last reading collected in each grid element. Grid elements that have not been visited are given a reading of 0.
                    4. Visit Counts Map: a 2D matrix showing the number of visits each grid element has received from all agents combined.
                    5. Obstacle Map: a 2D matrix of obstacles detected by agents
                    
            Critic Input tensor shape: (batch size, number of channels, height of grid, width of grid)
                1. batch size: 1 maps
                2. (map_count) number of channels: 5 input maps, same as Actor
                3. Height: grid height
                4. Width: grid width

        '''

        assert map_dim[0] > 0 and map_dim[0] == map_dim[1], 'Map dimensions mismatched. Must have equal x and y bounds.'
        
        channels = map_count
        pool_output = int(((map_dim[0]-2) / 2) + 1) # Get maxpool output height/width and floor it

        # Actor network
        self.step1 = nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, stride=1, padding=1)  # output tensor with shape (batchs, 8, Height, Width)
        self.relu = nn.ReLU()
        self.step2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output height and width is floor(((Width - Size)/ Stride) +1)
        self.step3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1) 
        #nn.ReLU()
        self.step4 = nn.Flatten(start_dim=0, end_dim= -1) # output tensor with shape (1, x)
        self.step5 = nn.Linear(in_features=16 * batches * pool_output * pool_output, out_features=32) 
        #nn.ReLU()
        self.step6 = nn.Linear(in_features=32, out_features=16) 
        #nn.ReLU()
        self.step7 = nn.Linear(in_features=16, out_features=5) # TODO eventually make '5' action_dim instead
        self.softmax = nn.Softmax(dim=0)  # Put in range [0,1] 
        self.global_critic = global_critic

        # TODO uncomment after ready to combine
        self.actor = nn.Sequential(
                        nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, stride=1, padding=1),  # output tensor with shape (4, 8, Height, Width)
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),  # output tensor with shape (4, 8, 2, 2)
                        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1),  # output tensor with shape (4, 16, 2, 2)
                        nn.ReLU(),
                        nn.Flatten(start_dim=0, end_dim= -1),  # output tensor with shape (1, x)
                        nn.Linear(in_features=16 * batches * pool_output * pool_output, out_features=32), # output tensor with shape (32)
                        nn.ReLU(),
                        nn.Linear(in_features=32, out_features=16), # output tensor with shape (16)
                        nn.ReLU(),
                        nn.Linear(in_features=16, out_features=5), # output tensor with shape (5)
                        nn.Softmax(dim=0)  # Put in range [0,1]
                    )

        # If decentralized critic
        # TODO uncomment after ready to combine
        if not self.global_critic:
            self.local_critic = nn.Sequential(
                        # Starting shape (batch_size, 4, Height, Width)
                        nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, stride=1, padding=1),  # output tensor with shape (batch_size, 8, Height, Width)
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),  # output tensor with shape (batch_size, 8, x, x) x is the floor(((Width - Size)/ Stride) +1)
                        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1),  # output tensor with shape (batch_size, 16, 2, 2)
                        nn.ReLU(),
                        nn.Flatten(start_dim=0, end_dim= -1),  # output tensor with shape (1, x)
                        nn.Linear(in_features=16 * batches * pool_output * pool_output, out_features=32), # output tensor with shape (32)
                        nn.ReLU(),
                        nn.Linear(in_features=32, out_features=16), # output tensor with shape (16)
                        nn.ReLU(),
                        nn.Linear(in_features=16, out_features=1), # output tensor with shape (1)
                        nn.ReLU(),
                    )

    def test(self, state_map_stack): 
        print("Starting shape, ", state_map_stack.size())
        x = self.step1(state_map_stack) # conv1
        x = self.relu(x)
        print("shape, ", x.size()) 
        x = self.step2(x) # Maxpool
        print("shape, ", x.size()) 
        x = self.step3(x) # conv2
        x = self.relu(x)
        print("shape, ", x.size()) 
        x = self.step4(x) # Flatten
        print("shape, ", x.size()) 
        x = self.step5(x) # linear
        x = self.relu(x) 
        print("shape, ", x.size()) 
        x = self.step6(x) # linear
        x = self.relu(x)
        print("shape, ", x.size()) 
        x = self.step7(x) # Output layer
        print("shape, ", x.size()) 
        x = self.softmax(x)
        
        print(x)
        pass
   
    def act(self, state_map_stack: torch.float) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:  # Tesnor Shape [batch_size, map_size, scaled_grid_x_bound, scaled_grid_y_bound] ([1, 5, 22, 22])

        # Select Action from Actor
        self.test(state_map_stack=state_map_stack)
        action_probs = self.actor(state_map_stack)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        # Get q-value from critic
        #state_value = self.local_critic(state_map_stack) if not self.global_critic else global_critic(state_map_stack) # TODO implement global critic
        state_value = self.local_critic(state_map_stack)
        
        return action.detach(), action_logprob.detach(), state_value.detach()
    
    def evaluate(self, state_map_stack, action):       
        
        self.actor.train()
        self.local_critic.train()
        
        # TODO Delete 
        print(summary(self.actor, input=state_map_stack.shape))
        self.test(state_map_stack=state_map_stack)
        exit()
        
        action_probs = self.actor(state_map_stack)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state_map_stack)
            
        return action_logprobs, state_values, dist_entropy        


class PPO:
    def __init__(self, state_dim, action_dim, grid_bounds, lr_actor, lr_critic, gamma, K_epochs, 
                 eps_clip, resolution_accuracy, steps_per_epoch, id, lamda=0.95, beta: float = 0.005, epsilon=0.2, random_seed=None):
        '''
        state_dim: The dimensions of the return from the environment
        action_dim: How many actions the actor chooses from
        grid_bounds: The grid bounds for the state returned by the environment. For RAD-PPO, this is (1, 1). This value will be scaled by the resolution_accuracy variable
        lr_actor: learning rate for actor neural network
        lr_critic: learning rate for critic neural network
        gamma: discount rate for expected return and Generalize Advantage Estimate (GAE) calculations
        K_epochs:
        eps_clip: 
        resolution_accuracy: How much to scale the convolution maps by (higher rate means more accurate, but more memory usage)
        lamda: smoothing parameter for Generalize Advantage Estimate (GAE) calculations
        '''
        # Testing
        if random_seed:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)                 
        
        # Hyperparameters
        self.gamma: float = gamma  # Discount factor
        self.lamda = lamda  # Smoothing parameter for GAE calculation
        self.beta = beta  # Entropy for loss function, encourages exploring different policies
        self.epsilon = epsilon  # clipping parameter to ensure we only make the maximum of ε% change to our policy at a time.
        self.eps_clip = eps_clip  # clip_ratio
        self.K_epochs = K_epochs
        self.resolution_accuracy = resolution_accuracy
        self.id = id
        self.steps_per_epoch = steps_per_epoch
        
        # Initialize
        self.maps = MapsBuffer(
                observation_dimension = state_dim,
                max_size=steps_per_epoch,
                gamma=gamma,
                lamda=lamda,      
                beta=beta,                    
                grid_bounds=grid_bounds, 
                resolution_accuracy=resolution_accuracy
            )

        self.policy = Actor(map_dim=self.maps.buffer.map_dimensions, state_dim=state_dim, action_dim=action_dim).to(self.maps.buffer.device) # TODO these are really slow

        self.optimizer_actor = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        
                    ])
        self.optimizer_critic = torch.optim.Adam([
                {'params': self.policy.local_critic.parameters(), 'lr': lr_critic}           
            ])
        
        self.policy_old = Actor(map_dim=self.maps.map_dimensions, state_dim=state_dim, action_dim=action_dim).to(self.maps.buffer.device)  # TODO why is this here
        self.policy_old.load_state_dict(self.policy.state_dict()) # TODO why is this here 
        
        self.MseLoss = nn.MSELoss()
        
    def select_action(self, state_observation: dict[int, StepResult], id: int) -> ActionChoice:         

        # Add intensity readings to a list if reading has not been seen before at that location.
        for observation in state_observation.values():
            key = (observation[1], observation[2])
            if key in self.maps.buffer.readings:
                if observation[0] not in self.maps.buffer.readings[key]:
                    self.maps.buffer.readings[key].append(observation[0])
            else:
                self.maps.buffer.readings[key] = [observation[0]]
            assert observation[0] in self.maps.buffer.readings[key], "Observation not recorded into readings buffer"

        with torch.no_grad():
            (
                location_map,
                others_locations_map,
                readings_map,
                visit_counts_map,
                obstacles_map
            ) = self.maps.observation_to_map(state_observation, id)
            
            map_stack = torch.stack([torch.tensor(location_map), torch.tensor(others_locations_map), torch.tensor(readings_map), torch.tensor(visit_counts_map),  torch.tensor(obstacles_map)]) # Convert to tensor
            
            # Add to mapstack buffer to eventually be converted into tensor with minibatches
            self.maps.buffer.mapstacks.append(map_stack)  # TODO if we're tracking this, do we need to track the observations?
            self.maps.buffer.coordinate_buffer.append({})
            for i, observation in state_observation.items():
                self.maps.buffer.coordinate_buffer[-1][i] = (observation[1], observation[2])
                
            # Add single minibatch for action selection
            map_stack = torch.unsqueeze(map_stack, dim=0) 
            
            #state = torch.FloatTensor(state).to(device) # Convert to tensor TODO already a tensor, is this necessary?
            #action, action_logprob = self.policy_old.act(state) # Choose action
            
            # TODO Delete 
            summary(self.policy_old.actor, input=map_stack.shape)        
            #self.policy.test(map_stack)
            
            action, action_logprob, state_value = self.policy_old.act(map_stack) # Choose action # TODO why old policy?

        return ActionChoice(id=id, action=action.item(), action_logprob=action_logprob.item(), state_value=state_value.item())
        #return action.item(), action_logprob.item(), state_value.item()
    
    def store(
            self,
            obs: npt.NDArray[Any],
            act: npt.NDArray[np.int32],
            rew: npt.NDArray[np.float32],
            val: npt.NDArray[np.float32],
            logp: npt.NDArray[np.float32],
            src: npt.NDArray[np.float32],
            terminal: npt.NDArray[np.bool],
        ) -> None:
        ''' Wrapper for inner buffer storage '''
        self.maps.buffer.store(obs=obs, act=act, rew=rew, val=val, logp=logp, src=src, terminal=terminal)

    def update(self):
        '''   
            Compute the new policy and log probabilities
                new_policy, log_probs = actor(states)

            Compute the advantages
                advantages = rewards + gamma * critic(next_states) - critic(states)

            Compute the PPO loss
                loss = ppo_loss(old_policy, new_policy, actions, rewards, advantages, epsilon)

            Perform the backpropagation
                actor_optimizer.zero_grad()
                loss.backward()

            Update the network's parameters
                actor_optimizer.step()            
        '''
        # Rewards have already been normalized and advantages already calculated with finish_path() function
        
        # Get data from buffers for old policy    
        data: dict[str, torch.Tensor] = self.maps.buffer.get_buffers_for_epoch_and_reset()  
        
        # Reset gradients (actor and critic will be set to train mode in evaluate())
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
            
        # Optimize policy for K epochs (TODO minibatches?)
        print(data['maps'])
        print(data["act"])

        for _ in range(self.K_epochs):
            for i, (observations, actions, advantages) in enumerate(train_loader):         
                logprobs, state_values, dist_entropy = self.policy.evaluate(data['maps'], data['act']) # Actor-critic            
        
        surrogate_loss = calculate_surrogate_loss()
        actor_loss = calculate_policy_loss(self, data)
        value_loss = calculate_critic_loss(self, data)
        
        actor_loss.backward()
        self.optimizer_actor.step()
        
        value_loss.backward()
        self.optimizer_critic.step()        
        
    
        # critic
        def calculate_value_loss(data):
            obs, ret = data['obs'], data['ret']
            obs = obs.to(self.maps.buffer.device)
            ret = ret.to(self.maps.buffer.device)
            return ((self.actor.local_critic(obs) - ret)**2).mean()    
        
        def calculate_policy_loss(self, data):
            '''
            Calculate how much the policy has changed: 
                ratio = policy_new / policy_old
            Take log form of this: 
                ratio = [log(policy_new) - log(policy_old)].exp()
            Calculate Actor loss as the minimum of two functions: 
                p1 = ratio * advantage
                p2 = clip(ratio, 1-epsilon, 1+epsilon) * advantage
                actor_loss = min(p1, p2)
                
            Calculate critic loss with MSE between returns and critic value
                critic_loss = (R - V(s))^2
                
            Caculcate total loss:
                total_loss = critic_loss * critic_discount + actor_loss - entropy
            '''            
            obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
            
            # Policy loss
            
            pi, logp = self.policy.actor(obs.to(self.maps.buffer.device), act.to(self.maps.buffer.device))
            logp = logp.cpu()
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * adv  # clipped ratio
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

            # TODO why not using surrogate?
            # surrogate = torch.exp(ratio) * advantages
            # surrogate_clipped = torch.clamp(surrogate, 1 - epsilon, 1 + epsilon) * advantages
            # surrogate_loss = -torch.min(surrogate, surrogate_clipped)
            # entropy = new_policy.entropy()
            # entropy_loss = -entropy_coef * entropy
            # loss = surrogate_loss + entropy_loss

            # Useful extra info
            approx_kl = (logp_old - logp).mean().item()
            ent = pi.entropy().mean().item()
            clipped = ratio.gt(1+self.eps_clip) | ratio.lt(1-self.eps_clip)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

            return loss_pi, pi_info            
         
    def update_old(self):
        # TODO I believe this is wrong; see vanilla_PPO.py TODO comment
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.maps.buffer.rewards), reversed(self.maps.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)  # Puts back in correct order
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions) # Actor-critic

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict()) # Actor-critic

        # clear buffer
        self.buffer.clear()
   
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path) # Actor-critic
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)) # Actor-critic
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)) # Actor-critic
        
    def render(self, savepath=getcwd(), save_map=True, add_value_text=False, interpolation_method='nearest', epoch_count: int=0):
        # TODO x and y are swapped - investigate if reading that way or a part of  imshow()
        if save_map:
            if not path.isdir(str(savepath) + "/heatmaps/"):
                mkdir(str(savepath) + "/heatmaps/")
        else:
            plt.show()                
     
        loc_transposed = self.maps.location_map.T # TODO this seems expensive
        other_transposed = self.maps.others_locations_map.T 
        readings_transposed = self.maps.readings_map.T
        visits_transposed = self.maps.visit_counts_map.T
        obstacles_transposed = self.maps.obstacles_map.T
     
        fig, (loc_ax, other_ax, intensity_ax, visit_ax, obs_ax) = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
        
        loc_ax.imshow(loc_transposed, cmap='viridis', interpolation=interpolation_method)
        loc_ax.set_title('Agent Location')
        loc_ax.invert_yaxis()        
        
        other_ax.imshow(other_transposed, cmap='viridis', interpolation=interpolation_method)
        other_ax.set_title('Other Agent Locations') 
        other_ax.invert_yaxis()  
        
        intensity_ax.imshow(readings_transposed, cmap='viridis', interpolation=interpolation_method)
        intensity_ax.set_title('Radiation Intensity')
        intensity_ax.invert_yaxis()
        
        visit_ax.imshow(visits_transposed, cmap='viridis', interpolation=interpolation_method)
        visit_ax.set_title('Visit Counts') 
        visit_ax.invert_yaxis()
        
        obs_ax.imshow(obstacles_transposed, cmap='viridis', interpolation=interpolation_method)
        obs_ax.set_title('Obstacles Detected (cm from Agent)') 
        obs_ax.invert_yaxis()
        
        #divider = make_axes_locatable(loc_ax)
        #cax = divider.append_axes('right', size='5%', pad=0.05)             
        #fig.colorbar(loc_ax, cax=cax, orientation='vertical')
        
        # Add values to gridsquares if value is greater than 0 #TODO if large grid, this will be slow
        if add_value_text:
            for i in range(loc_transposed.shape[0]):
                for j in range(loc_transposed.shape[1]):
                    if loc_transposed[i, j] > 0: 
                        loc_ax.text(j, i, loc_transposed[i, j].astype(int), ha="center", va="center", color="black", size=6)
                    if other_transposed[i, j] > 0: 
                        other_ax.text(j, i, other_transposed[i, j].astype(int), ha="center", va="center", color="black", size=6)
                    if readings_transposed[i, j] > 0:
                        intensity_ax.text(j, i, readings_transposed[i, j].astype(int), ha="center", va="center", color="black", size=4)
                    if visits_transposed[i, j] > 0:
                        visit_ax.text(j, i, visits_transposed[i, j].astype(int), ha="center", va="center", color="black", size=6)
                    if obstacles_transposed[i, j] > 0:
                        obs_ax.text(j, i, obstacles_transposed[i, j].astype(int), ha="center", va="center", color="black", size=6)                        
        
        fig.savefig(f'{str(savepath)}/heatmaps/heatmap_agent{self.id}_epoch_{epoch_count}-{self.render_counter}.png')
        
        self.render_counter += 1
        plt.close(fig)