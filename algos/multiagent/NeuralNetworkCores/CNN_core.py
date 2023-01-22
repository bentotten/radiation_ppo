from os import stat, path, mkdir, getcwd

from numpy import dtype
import numpy as np
import numpy.typing as npt

import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical
from torchinfo import summary

import pytorch_lightning as pl

from dataclasses import dataclass, field, asdict
from typing import Any, List, Tuple, Union, Literal, NewType, Optional, TypedDict, cast, get_args, Dict, Callable, overload
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


def mlp(
    sizes: list[Shape],
    activation,
    output_activation=nn.Identity,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers = []
    for j in range(len(sizes) - 1):
        layer = [nn.Linear(sizes[j], sizes[j + 1])]

        if layer_norm:
            ln = nn.LayerNorm(sizes[j + 1]) if j < len(sizes) - 1 else None
            layer.append(ln)

        layer.append(activation() if j < len(sizes) - 1 else output_activation())
        layers += layer

    if layer_norm and None in layers:
        layers.remove(None)

    return nn.Sequential(*layers)


@dataclass()
class ActionChoice():
    id: int 
    action: npt.NDArray # size (1)
    action_logprob: npt.NDArray # size (1)
    state_value: npt.NDArray # size(1)

    # For compatibility with RAD-PPO
    hiddens: Union[torch.Tensor, None] = field(default=None)
    loc_pred: Union[npt.NDArray, None] = field(default=None)


@dataclass
class RolloutBuffer:      
    # Buffers
    mapstacks: list = field(init=False)
    coordinate_buffer: list = field(init=False)
    readings: Dict[Any, list] =field(init=False)
    
    
    def __init__(self):
        self.mapstacks: List = []  # TODO Change to numpy arrays
        self.coordinate_buffer: CoordinateStorage = []    
        self.readings: Dict[Any, list] = {} # For heatmap resampling        
    
    def clear(self):
        # Reset readings, mapstacks, and buffer pointers
        del self.mapstacks[:]
        del self.coordinate_buffer[:]     
        self.readings.clear()


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
    
    buffer: RolloutBuffer = field(default_factory=lambda: RolloutBuffer())

    def __post_init__(self):      
        # Scaled maps
        self.map_dimensions = (int(self.grid_bounds[0] * self.resolution_accuracy), int(self.grid_bounds[1] * self.resolution_accuracy))
        self.x_limit_scaled: int = self.map_dimensions[0]
        self.y_limit_scaled: int = self.map_dimensions[1]    
        
        self.clear()

    def clear(self):
        self.location_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  # TODO rethink this, this is very slow - potentially change to torch?
        self.others_locations_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  # TODO rethink this, this is very slow
        self.readings_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  # TODO rethink this, this is very slow
        self.visit_counts_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  # TODO rethink this, this is very slow
        self.obstacles_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  # TODO rethink this, this is very slow
        
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
    def __init__(self, map_dim, state_dim, batches: int=1, map_count: int=5, action_dim: int=5):
        super(Actor, self).__init__()
        
        ''' Actor Input tensor shape: (batch size, number of channels, height of grid, width of grid)
                1. batch size: 1 mapstack
                2. (map_count) number of channels: 5 input maps
                3. Height: grid height
                4. Width: grid width
            
                5 maps: 
                    1. Location Map: a 2D matrix showing the agents location.
                    2. Map of Other Locations: a 2D matrix showing the number of agents located in each grid element (excluding current agent).
                    3. Readings map: a 2D matrix showing the last reading collected in each grid element. Grid elements that have not been visited are given a reading of 0.
                    4. Visit Counts Map: a 2D matrix showing the number of visits each grid element has received from all agents combined.
                    5. Obstacle Map: a 2D matrix of obstacles detected by agents
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
        #self.test(state_map_stack=state_map_stack)
        action_probs = self.actor(state_map_stack)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action, action_logprob
    
    def evaluate(self, state_map_stack, action):       
        
        self.actor.train()
        self.local_critic.train()
        
        action_probs = self.actor(state_map_stack)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state_map_stack)
            
        return action_logprobs, state_values, dist_entropy        

    def _reset_state(self):
        return self._get_init_states()

class Critic(nn.Module):
    def __init__(self, map_dim, state_dim, batches: int=1, map_count: int=5, action_dim: int=5, global_critic: bool=False):
        super(Critic, self).__init__()
        
        '''
            Critic Input tensor shape: (batch size, number of channels, height of grid, width of grid)
                1. batch size: 1 mapstack
                2. (map_count) number of channels: 5 input maps, same as Actor
                3. Height: grid height
                4. Width: grid width                
            
            5 maps: 
                1. Location Map: a 2D matrix showing the agents location.
                2. Map of Other Locations: a 2D matrix showing the number of agents located in each grid element (excluding current agent).
                3. Readings map: a 2D matrix showing the last reading collected in each grid element. Grid elements that have not been visited are given a reading of 0.
                4. Visit Counts Map: a 2D matrix showing the number of visits each grid element has received from all agents combined.
                5. Obstacle Map: a 2D matrix of obstacles detected by agents
        '''

        assert map_dim[0] > 0 and map_dim[0] == map_dim[1], 'Map dimensions mismatched. Must have equal x and y bounds.'
        
        channels = map_count
        pool_output = int(((map_dim[0]-2) / 2) + 1) # Get maxpool output height/width and floor it

        # Critic network
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
        self.step7 = nn.Linear(in_features=16, out_features=1) # output tensor with shape (1)
        #nn.ReLU()
        
        self.critic = nn.Sequential(
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
        x = self.relu(x)
        
        print(x)
        pass
   
    def act(self, state_map_stack: torch.float) -> Tuple[torch.tensor, torch.tensor, torch.tensor]: 
        # Get q-value from critic
        #self.test(state_map_stack)
        state_value = self.critic(state_map_stack)
        return state_value
    
    def evaluate(self, state_map_stack, action):       
        self.critic.train()
        state_values = self.critic(state_map_stack)
            
        return state_values        

    def _reset_state(self):
        return self._get_init_states()

@dataclass
class PPO:
    id: int    
    state_dim: int
    action_dim: int
    grid_bounds: tuple[float]
    resolution_accuracy: float
    steps_per_epoch: int
    random_seed: int = field(default=None)
    critic: Any = field(default=None)  # Eventually allows for a global critic
    
    # Moved to PPO Buffer    
    #steps_per_epoch    
    # lamda=0.95
    # beta: float = 0.005
    # epsilon=0.2
    # lr_actor
    # lr_critic
    # gamma
    # K_epochs
    # eps_clip        
    '''
    state_dim: The dimensions of the return from the environment
    action_dim: How many actions the actor chooses from
    grid_bounds: The grid bounds for the state returned by the environment. For RAD-PPO, this is (1, 1). This value will be scaled by the resolution_accuracy variable
    resolution_accuracy: How much to scale the convolution maps by (higher rate means more accurate, but more memory usage)
    lamda: smoothing parameter for Generalize Advantage Estimate (GAE) calculations
    '''
    def __post_init__(self):      
        # Initialize buffers and neural networks
        self.maps = MapsBuffer(
                observation_dimension = self.state_dim,
                max_size=self.steps_per_epoch,                  
                grid_bounds=self.grid_bounds, 
                resolution_accuracy=self.resolution_accuracy
            )

        self.pi = Actor(map_dim=self.maps.map_dimensions, state_dim=self.state_dim, action_dim=self.action_dim)#.to(self.maps.buffer.device)
        
        if not self.critic:
            self.critic = Critic(map_dim=self.maps.map_dimensions, state_dim=self.state_dim, action_dim=self.action_dim)#.to(self.maps.buffer.device) # TODO these are really slow
        
        # For PFGRU (Developed from RAD-A2C https://github.com/peproctor/radiation_ppo)
        bpf_hsize: int = 64
        bpf_num_particles: int = 40
        bpf_alpha: float = 0.7            
        # TODO rename this (this is the PFGRU module); naming this "model" for compatibility reasons (one refactor at a time!), but the true model is the maps buffer
        self.model = PFGRUCell(bpf_num_particles, self.state_dim - 8, self.state_dim - 8, bpf_hsize, bpf_alpha, True, "relu") 
        
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
            #self.maps.buffer.mapstacks.append(map_stack)  # TODO if we're tracking this, do we need to track the observations?
            self.maps.buffer.coordinate_buffer.append({})
            for i, observation in state_observation.items():
                self.maps.buffer.coordinate_buffer[-1][i] = (observation[1], observation[2])
                
            # Add single batch tensor dimension for action selection
            map_stack = torch.unsqueeze(map_stack, dim=0) 
            
            # Get actions and values                          
            action, action_logprob,  = self.pi.act(map_stack) # Choose action
            state_value = self.critic.act(map_stack)  # Should be a pointer to either local critic or global critic

        return ActionChoice(id=id, action=action.numpy(), action_logprob=action_logprob.numpy(), state_value=state_value.numpy())
        #return action.item(), action_logprob.item(), state_value.item()
    
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
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)) # Actor-critic
        
    def render(self, savepath=getcwd(), save_map=True, add_value_text=False, interpolation_method='nearest', epoch_count: int=0):
        ''' Renders heatmaps from maps buffer '''
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
        


# Developed from RAD-A2C https://github.com/peproctor/radiation_ppo
class PFRNNBaseCell(nn.Module):
    """parent class for PFRNNs"""

    def __init__(
        self,
        num_particles: int,
        input_size: int,
        hidden_size: int,
        resamp_alpha: float,
        use_resampling: bool,
        activation: str,
    ):
        """init function

        Arguments:
            num_particles {int} -- number of particles
            input_size {int} -- input size
            hidden_size {int} -- particle vector length
            resamp_alpha {float} -- alpha value for soft-resampling
            use_resampling {bool} -- whether to use soft-resampling
            activation {str} -- activation function to use
        """
        super().__init__()
        self.num_particles: int = num_particles
        self.samp_thresh: float = num_particles * 1.0
        self.input_size: int = input_size
        self.h_dim: int = hidden_size
        self.resamp_alpha: float = resamp_alpha
        self.use_resampling: bool = use_resampling
        self.activation: str = activation
        self.initialize: str = "rand"
        if activation == "relu":
            self.batch_norm: nn.BatchNorm1d = nn.BatchNorm1d(
                self.num_particles, track_running_stats=False
            )

    @overload
    def resampling(self, particles: torch.Tensor, prob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @overload
    def resampling(
        self, particles: tuple[torch.Tensor, torch.Tensor], prob: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        ...

    def resampling(
        self, particles: torch.Tensor | tuple[torch.Tensor, torch.Tensor], prob: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor] | torch.Tensor, torch.Tensor]:
        """soft-resampling

        Arguments:
            particles {tensor} -- the latent particles
            prob {tensor} -- particle weights

        Returns:
            tuple -- particles
        """

        resamp_prob = (
            self.resamp_alpha * torch.exp(prob)
            + (1 - self.resamp_alpha) * 1 / self.num_particles
        )
        resamp_prob = resamp_prob.view(self.num_particles, -1)
        flatten_indices = (
            torch.multinomial(
                resamp_prob.transpose(0, 1),
                num_samples=self.num_particles,
                replacement=True,
            )
            .transpose(1, 0)
            .contiguous()
            .view(-1, 1)
            .squeeze()
        )

        # PFLSTM
        if type(particles) == tuple:
            particles_new = (
                particles[0][flatten_indices],
                particles[1][flatten_indices],
            )
        # PFGRU
        else:
            particles_new = particles[flatten_indices]

        prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (
            self.resamp_alpha * prob_new + (1 - self.resamp_alpha) / self.num_particles
        )
        prob_new = torch.log(prob_new).view(self.num_particles, -1)
        prob_new = prob_new - torch.logsumexp(prob_new, dim=0, keepdim=True)

        return particles_new, prob_new

    def reparameterize(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Implements the reparameterization trick introduced in [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

        Arguments:
            mu {tensor} -- learned mean
            var {tensor} -- learned variance

        Returns:
            tensor -- sample
        """
        std: torch.Tensor = F.softplus(var)
        eps: torch.Tensor = torch.FloatTensor(std.shape).normal_()
        return mu + eps * std


# Developed from RAD-A2C https://github.com/peproctor/radiation_ppo
class PFGRUCell(PFRNNBaseCell):
    def __init__(
        self,
        num_particles: int,
        input_size: int,
        obs_size: int,
        hidden_size: int,
        resamp_alpha: float,
        use_resampling: bool,
        activation: str,
    ):
        super().__init__(
            num_particles,
            input_size,
            hidden_size,
            resamp_alpha,
            use_resampling,
            activation,
        )

        self.fc_z: nn.Linear = nn.Linear(self.h_dim + self.input_size, self.h_dim)
        self.fc_r: nn.Linear = nn.Linear(self.h_dim + self.input_size, self.h_dim)
        self.fc_n: nn.Linear = nn.Linear(self.h_dim + self.input_size, self.h_dim * 2)

        self.fc_obs: nn.Linear = nn.Linear(self.h_dim + self.input_size, 1)
        self.hid_obs: nn.Sequential = mlp([self.h_dim, 24, 2], nn.ReLU)
        self.hnn_dropout: nn.Dropout = nn.Dropout(p=0)

    def forward(
        self, input_: torch.Tensor, hx: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One step forward for PFGRU

        Arguments:
            input_ {tensor} -- the input tensor
            hx {tuple} -- previous hidden state (particles, weights)

        Returns:
            tuple -- new tensor
        """
        h0, p0 = hx
        obs_in = input_.repeat(h0.shape[0], 1)
        obs_cat = torch.cat((h0, obs_in), dim=1)

        z = torch.sigmoid(self.fc_z(obs_cat))
        r = torch.sigmoid(self.fc_r(obs_cat))
        n = self.fc_n(torch.cat((r * h0, obs_in), dim=1))

        mu_n, var_n = torch.split(n, split_size_or_sections=self.h_dim, dim=1)
        n: torch.Tensor = self.reparameterize(mu_n, var_n)

        if self.activation == "relu":
            # if we use relu as the activation, batch norm is require
            n = n.view(self.num_particles, -1, self.h_dim).transpose(0, 1).contiguous()
            n = self.batch_norm(n)
            n = n.transpose(0, 1).contiguous().view(-1, self.h_dim)
            n = torch.relu(n)
        elif self.activation == "tanh":
            n = torch.tanh(n)
        else:
            raise ModuleNotFoundError

        h1: torch.Tensor = (1 - z) * n + z * h0

        p1 = self.observation_likelihood(h1, obs_in, p0)

        if self.use_resampling:
            h1, p1 = self.resampling(h1, p1)

        p1 = p1.view(-1, 1)
        mean_hid = torch.sum(torch.exp(p1) * self.hnn_dropout(h1), dim=0)
        loc_pred: torch.Tensor = self.hid_obs(mean_hid)

        return loc_pred, (h1, p1)

    def observation_likelihood(self, h1: torch.Tensor, obs_in: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
        """observation function based on compatibility function"""
        logpdf_obs: torch.Tensor = self.fc_obs(torch.cat((h1, obs_in), dim=1))
        p1: torch.Tensor = logpdf_obs + p0
        p1 = p1.view(self.num_particles, -1, 1)
        p1 = F.log_softmax(p1, dim=0)
        return p1

    def init_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        initializer: Callable[[int, int], torch.Tensor] = (
            torch.rand if self.initialize == "rand" else torch.zeros
        )
        h0 = initializer(batch_size * self.num_particles, self.h_dim)
        p0: torch.Tensor = torch.ones(batch_size * self.num_particles, 1) * np.log(
            1 / self.num_particles
        )
        hidden = (h0, p0)
        return hidden


