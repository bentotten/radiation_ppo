from os import stat, path, mkdir, getcwd
import sys

from dataclasses import dataclass, field, asdict
from typing import Any, List, Tuple, Union, NewType, Optional, TypedDict, cast, get_args, Dict, Callable, overload, Union, List, Dict, NamedTuple

import numpy as np
import numpy.typing as npt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt # type: ignore


# Maps
#: [New Type] Array indicies to access a GridSquare (x, y). Type: Tuple[float, float]
Point = NewType("Point", Tuple[float, float])
#: [New Type] Heatmap - a two dimensional array that holds heat values for each gridsquare. Note: the number of gridsquares is scaled with a resolution accuracy variable. Type: numpy.NDArray[np.float32]
Map = NewType("Map", npt.NDArray[np.float32])
#: [New Type] Tracks previous coordinates for all agents in order to update them on the current-location and others-locations heatmaps. Type: List[Dict[int, Point]]
CoordinateStorage = NewType("CoordinateStorage", List[Dict[int, Point]])

# Helpers
#: [Type Alias] Used in multi-layer perceptron in the prediction module (PFGRU). Type: int | Tuple[int, ...]
Shape = Union[int, Tuple[int, ...]]

#: [Global] Detector-obstruction range measurement threshold in cm for inflating step size for obstruction heatmap. Type: float
DIST_TH = 110.0
#: [Global] Toggle for simple value/max normalization vs stdbuffer for radiation intensity map and log-based for visit-counts map. Type: bool
SIMPLE_NORMALIZATION = False


def _log_and_normalize_test(max: int = 120):
    ''' For modeling expected values as visit count increases. This puts greater emphasis on lower step counts.'''
    
    x = [x for x in range(max)]
    y = []    
    
    for i in range(max):
        y.append((( 
            np.log(
                2 + i * (
                            max** ( np.log(2) / np.log(max) )
                        )
                    )
            ) / np.log(max)) * 1/(np.log(2 *max)/ np.log(max))
        )

    print(x)
    print(y)

    # Plot the function
    plt.plot(x, y)

    # Add labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of log(x)')

    # Show the plot
    plt.show()


class ActionChoice(NamedTuple):
    ''' Named Tuple - Standardized response/return template from Actor-Critic for action selection '''
    #: An Agent's unique identifier that serves as a hash key. 
    id: int 
    #: A single integer that represents an agent action in the environment. Stored in a single-element numpy array for processing convinience.
    action: npt.NDArray[np.int32] # size (1)
    #: The log of the policy distribution. Taking the gradient of the log probability is more stable than using the actual density.
    action_logprob: npt.NDArray[np.float32] # size (1)
    #: The estimated value of being in this state. Note: Using GAE for advantage, this is the state-value, not the q-value
    state_value: npt.NDArray[np.float32] # size(1)
    #: Coordinates predicted by the location prediction model (PFGRU). TODO: Implement for CNN
    loc_pred: Union[torch.Tensor, None] = None

    # For compatibility with RAD-PPO
    #: Hidden state (for compatibility with RAD-PPO)
    hiddens: Union[torch.Tensor, None] = None


@dataclass
class StatisticsBuffer:
    ''' 
    Statistics buffer for standardizing intensity readings from environment (B. Welford, "Note on a method for calculating corrected sums of squares and products").
    Because an Agent does not know the maximum radiation intensity it may encounter, it uses this estimated running sample mean and variance instead. 
    '''
    #: Running mean
    mu: float = 0.0
    #: Squared distance from the mean
    sigma: float = 0.0 
    #: Sample standard-deviation
    sample_std: float = 1.0 
    #: Count of how many samples have been seen so far
    count: int = 0
    
    def update(self, reading: float) -> None:
        ''' Update estimate running sample mean and variance for standardizing radiation intensity readings.
            
            #. The existing mean is subtracted from the new reading to get the initial delta. 
            #. This delta is then divided by the number of samples seen so far and added to the existing mean to create a new mean.
            #. This new mean is then subtracted from the reading to get new delta.
            #. This new delta is multiplied by the old delta and added to the existing squared distance from the mean. 
            #. To get the sample variance, the new existing squared distance from the mean is divided by the number of samples seen so far minus 1. 
            #. To get the sample standard deviation, the square root of this value is taken.
            
            Thank you to `Wiki - Algorithms for calculating variance <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#cite_ref-5>`_ 
            and `NZMaths - Sample Variance <https://nzmaths.co.nz/category/glossary/sample-variance>`_
            Original: B. Welford, "Note on a method for calculating corrected sums of squares and products"
        '''
        self.count += 1
        if self.count == 1:
            self.mu = reading  # For first reading, mu is equal to that reading
        else:
            mu_new = self.mu + (reading - self.mu) / (self.count) 
            sigma_new = self.sigma + (reading - self.mu) * (reading - mu_new)
            self.mu = mu_new
            self.sigma = sigma_new
            self.sample_std = max(math.sqrt(sigma_new / (self.count - 1)), 1)

    def reset(self) -> None:
        ''' Reset statistics buffer '''
        self = StatisticsBuffer()


@dataclass
class RolloutBuffer:      
    # Buffers
    coordinate_buffer: CoordinateStorage = field(init=False)
    readings: Dict[Union[str, Tuple[float, float]], Union[np.floating[Any], List[np.floating[Any]]]] = field(init=False)
    
    def __post_init__(self)-> None:
        self.coordinate_buffer: CoordinateStorage = CoordinateStorage(list())
        self.readings = {'max': np.float32(0.0), 'min': np.float32(0.0)} # For heatmap resampling        
    
    def clear(self)-> None:
        # Reset readings and coordinates buffers
        del self.coordinate_buffer[:]     
        self.readings.clear()
        self.readings = {'max': np.float32(0.0), 'min': np.float32(0.0)} # For heatmap resampling                


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
    steps_per_episode: int # Used for normalizing visits count in visits map
    number_of_agents: int # Used for normalizing visists count in visits map
            
    # Parameters
    grid_bounds: Tuple = field(default_factory= lambda: (1,1))  # Initial grid bounds for state x and y coordinates. For RADPPO, these are scaled to be below 0, so bounds are 1x1
    resolution_accuracy: float = field(default=100) # How much to multiply grid bounds and state coordinates by. 100 will return to full accuracy for RADPPO
    obstacle_state_offset: int = field(default=3) # Number of initial elements in state return that do not indicate there is an obstacle. First element is intensity, second two are x and y coords
    offset: float = field(default=0)  # Offset for when boundaries are different than "search area".
    
    # Initialized elsewhere
    x_limit_scaled: int = field(init=False)  # maximum x value in maps
    y_limit_scaled: int = field(init=False)  # maximum y value in maps
    map_dimensions: Tuple = field(init=False)  # Scaled dimensions of each map - used to create the CNNs
    base: int = field(init=False) # Base for log() for visit count map normalization
        
    # Maps
    location_map: Map = field(init=False)  # Location Map: a 2D matrix showing the individual agent's location.
    others_locations_map: Map = field(init=False)  # Map of Other Locations: a grid showing the number of agents located in each grid element (excluding current agent).
    readings_map: Map = field(init=False)  # Readings map: a grid of the last reading collected in each grid square - unvisited squares are given a reading of 0.
    obstacles_map: Map = field(init=False) # bstacles Map: a grid of how far from an obstacle each agent was when they detected it
    visit_counts_map: Map = field(init=False) # Visit Counts Map: a grid of the number of visits to each grid square from all agents combined.
    visit_counts_shadow: Dict = field(default_factory=lambda: dict()) # Due to lazy allocation and python floating point precision, it is cheaper to calculate the log on the fly with a second sparce matrix than to inflate a log'd number
    
    # Buffers
    buffer: RolloutBuffer = field(default_factory=lambda: RolloutBuffer())
    observation_buffer: List = field(default_factory=lambda: list())  # TODO move to PPO buffer
    intensity_standardization: StatisticsBuffer = field(default_factory=lambda: StatisticsBuffer())

    def __post_init__(self)-> None:
        self.base = self.steps_per_episode * self.number_of_agents
        
        # Scaled maps
        self.map_dimensions = (
            int(self.grid_bounds[0] * self.resolution_accuracy) + int(self.offset  * self.resolution_accuracy),
            int(self.grid_bounds[1] * self.resolution_accuracy) + int(self.offset  * self.resolution_accuracy)
        )
        self.x_limit_scaled: int = self.map_dimensions[0]
        self.y_limit_scaled: int = self.map_dimensions[1]    
        self.clear()

    def clear_maps(self)-> None:
        ''' Clear maps. Often called at the end of an episode to reset the maps for a new starting location and source location'''
        # TODO rethink this, this is very slow
        self.location_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  # TODO rethink this, this is very slow - potentially change to torch or keep a ref count?
        self.others_locations_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  
        self.readings_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  
        self.obstacles_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32)) 
        self.visit_counts_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32)) 
        self.visit_counts_shadow.clear() # Stored tuples (x, y, 2(i)) where i increments every hit
        self.buffer.clear()
        self.intensity_standardization.reset()
        
    def clear(self)-> None:
        ''' Clear maps and buffers. Often called at the end of an Epoch when updates have been applied and its time for new observations'''
        del self.observation_buffer[:]
        self.clear_maps()        
        
    def observation_to_map(self, observation: Dict[int, list], id: int
                     ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:  
        '''
        Convert an 11-element observation dictionary from all agents into maps. Normalize Data
        
        observation (Dict): observations from environment for all agents
            id (int): ID of agent to reference in observation object 
            
            Returns a Tuple of 2d map arrays
        '''
        
        # TODO Remove redundant calculations and massively consolidate
        
        # Process observation for current agent's locations map
        deflated_x = observation[id][1]
        deflated_y = observation[id][2]
        current_a_scaled_coordinates: Tuple[int, int] = (int(deflated_x * self.resolution_accuracy), int(deflated_y * self.resolution_accuracy))        
        # Capture current and reset previous location
        if self.buffer.coordinate_buffer:
            last_state = self.buffer.coordinate_buffer[-1][id]
            scaled_last_coordinates = (int(last_state[0] * self.resolution_accuracy), int(last_state[1] * self.resolution_accuracy))
            x_old = int(scaled_last_coordinates[0])
            y_old = int(scaled_last_coordinates[1])
            self.location_map[x_old][y_old] -= 1 # In case agents are at same location, usually the start-point
            assert self.location_map[x_old][y_old] > -1, "location_map grid coordinate reset where agent was not present. The map location that was reset was already at 0."
        # Set new location
        current_a_x: int = int(current_a_scaled_coordinates[0])
        current_a_y: int = int(current_a_scaled_coordinates[1])
        self.location_map[current_a_x][current_a_y] = 1
        
        ### Process observation for other agent's locations map
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
                other_a_x: int = int(others_scaled_coordinates[0])
                other_a_y: int = int(others_scaled_coordinates[1])
                self.others_locations_map[other_a_x][other_a_y] += 1  # Initial agents begin at same location        

        ### Process observation for readings_map
        # Update stat buffer for later normalization
        if not SIMPLE_NORMALIZATION:
            for agent_id in observation:
                self.intensity_standardization.update(observation[agent_id][0])
        for agent_id in observation:
            readings_scaled_coordinates: Tuple[int, int] = (int(observation[agent_id][1] * self.resolution_accuracy), int(observation[agent_id][2] * self.resolution_accuracy))            
            readings_x: int = int(readings_scaled_coordinates[0])
            readings_y: int = int(readings_scaled_coordinates[1])
            
            # Inflate coordinates
            unscaled_coordinates: Tuple[float, float] = (observation[agent_id][1], observation[agent_id][2])
            assert len(self.buffer.readings[unscaled_coordinates]) > 0 # type: ignore
            
            # Get estimated reading and save new max for later normalization
            estimated_reading = np.median(self.buffer.readings[unscaled_coordinates])
            if estimated_reading > self.buffer.readings['max']:
                self.buffer.readings['max'] = estimated_reading            
            
            # Normalize
            if SIMPLE_NORMALIZATION:
                normalized_reading = estimated_reading / self.buffer.readings['max']
                assert normalized_reading <= 1.0 and normalized_reading >= 0.0 # Stat buffer can give 0 back as a reading
                
            else:
                normalized_reading = np.clip((observation[agent_id][0] - self.intensity_standardization.mu) / self.intensity_standardization.sig_obs, -8, 8)     
                # TODO Get range for this tool and add an assert        
            if estimated_reading > 0:
                self.readings_map[readings_x][readings_y] = normalized_reading 
            else:
                assert estimated_reading >= 0

        ### Process observation for visit_counts_map
        for agent_id in observation:
            visits_scaled_coordinates = (int(observation[agent_id][1] * self.resolution_accuracy), int(observation[agent_id][2] * self.resolution_accuracy))            
            visits_x = int(visits_scaled_coordinates[0])
            visits_y = int(visits_scaled_coordinates[1])

            # Increment shadow table
            # If visited before, fetch counter from shadow table, else create shadow table entry
            if visits_scaled_coordinates in self.visit_counts_shadow.keys():
                current = self.visit_counts_shadow[visits_scaled_coordinates]
                self.visit_counts_shadow[visits_scaled_coordinates] += 2
            else:
                current = 0
                self.visit_counts_shadow[visits_scaled_coordinates] = 2
                    
            if SIMPLE_NORMALIZATION:
                self.visit_counts_map[visits_x][visits_y] = current / self.base
            else: 
                with np.errstate(all='raise'):
                    # Using 2 due to log(1) == 0
                    self.visit_counts_map[visits_x][visits_y] = (
                            (
                                np.log(2 + current, dtype=np.float128) / np.log(self.base, dtype=np.float128) # Change base to max steps * num agents
                            ) * 1/(np.log(2 * self.base)/ np.log(self.base)) # Put in range [0, 1]
                        )          
                    assert self.visit_counts_map.max() <= 1.0 and self.visit_counts_map.min() >= 0.0, "Normalization error" 
            
        ### Process observation for obstacles_map 
        for agent_id in observation:
            scaled_agent_coordinates = (int(observation[agent_id][1] * self.resolution_accuracy), int(observation[agent_id][2] * self.resolution_accuracy))   
            if np.count_nonzero(observation[agent_id][self.obstacle_state_offset:]) > 0:
                indices = np.flatnonzero(observation[agent_id][self.obstacle_state_offset::]).astype(int)
                for index in indices:
                    real_index = int(index + self.obstacle_state_offset)
                    obstruct_x = int(scaled_agent_coordinates[0])
                    obstruct_y = int(scaled_agent_coordinates[1])  
                                        
                    # Inflate to actual distance, then convert and round with resolution_accuracy
                    #inflated_distance = (-(observation[agent_id][real_index] * DIST_TH - DIST_TH))
                    
                    # Semi-arbritrary, but should make the number higher as the agent gets closer to the object, making heatmap look more correct
                    self.obstacles_map[obstruct_x][obstruct_y] = observation[agent_id][real_index]
        
        return self.location_map, self.others_locations_map, self.readings_map, self.visit_counts_map, self.obstacles_map


#TODO make a reset function, similar to self.ac.reset_hidden() in RADPPO
class Actor(nn.Module):
    def __init__(self, map_dim, observation_space, batches: int=1, map_count: int=5, action_dim: int=5):
        super(Actor, self).__init__()
        ''' 
            When an observation is fed to this base class, it is transformed into a series of stackable observation maps (numerican matrices/tensors). As these maps are fed 
            through the network, Convolutional and pooling layers train a series of filters that operate on the data and extract features from it. 
            These features are then distilled through linear layers to produce an array that contains probabilities, where each element cooresponds to an action.
            
            Actor Input tensor shape: (batch size, number of channels, height of grid, width of grid)
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
                        nn.Linear(in_features=16, out_features=action_dim), # output tensor with shape (5)
                        nn.Softmax(dim=0)  # Put in range [0,1]
                    )

    def test(self, state_map_stack): 
        print("Starting shape, ", state_map_stack.size())
        torch.set_printoptions(threshold=sys.maxsize)
        
        with open('0_starting_mapstack.txt', 'w') as f:
            print(state_map_stack, file=f)
        
        x = self.step1(state_map_stack) # conv1
        with open('1_1st_covl.txt', 'w') as f:
            print(x, file=f)
               
        x = self.relu(x)
        with open('2_1st_relu.txt', 'w') as f:
            print(x, file=f)              
        print("shape, ", x.size()) 
        
        x = self.step2(x) # Maxpool
        with open('3_maxpool.txt', 'w') as f:
            print(x, file=f)                            
        print("shape, ", x.size())
        
        x = self.step3(x) # conv2
        with open('4_2nd_convl.txt', 'w') as f:
            print(x, file=f)          
                   
        x = self.relu(x)
        with open('5_2nd_relu.txt', 'w') as f:
            print(x, file=f)           
        print("shape, ", x.size()) 
        
        x = self.step4(x) # Flatten
        with open('6_flatten.txt', 'w') as f:
            print(x, file=f)           
        print("shape, ", x.size()) 
        
        x = self.step5(x) # linear
        with open('7_1st_linear.txt', 'w') as f:
            print(x, file=f)           
                                     
        x = self.relu(x) 
        with open('8_3rd_relu_.txt', 'w') as f:
            print(x, file=f)         
        print("shape, ", x.size()) 
        
        x = self.step6(x) # linear
        with open('9_2nd_linear.txt', 'w') as f:
            print(x, file=f)           
        
        x = self.relu(x)
        with open('10_4th_relu.txt', 'w') as f:
            print(x, file=f)              
        print("shape, ", x.size()) 
        
        x = self.step7(x) # Output layer
        with open('11_3rd_linear_output.txt', 'w') as f:
            print(x, file=f)               
        print("shape, ", x.size()) 
        
        x = self.softmax(x)
        with open('11_softmax.txt', 'w') as f:
            print(x, file=f)          
        
        print(x)
        pass
   
    def act(self, state_map_stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # Tesnor Shape [batch_size, map_size, scaled_grid_x_bound, scaled_grid_y_bound] ([1, 5, 22, 22])
        ''' Select action from action probabilities returned by actor.'''
        action_probs: torch.Tensor = self.actor(state_map_stack)
        dist = Categorical(action_probs)
        action: torch.Tensor = dist.sample()
        action_logprob: torch.Tensor = dist.log_prob(action) # log of the policy distribution. Taking the gradient of the log probability is more stable than using the actual density
        
        return action, action_logprob

    def forward(self, observation = None, act = None):
        raise NotImplementedError
    
    def evaluate(self, state_map_stack: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' Put actor into "train" mode and get action logprobabilities for an observation mapstack. Then calculate a particular actions entropy.'''
        self.actor.train()
        
        action_probs: torch.Tensor = self.actor(state_map_stack)
        dist = Categorical(action_probs)
        action_logprobs: torch.Tensor  = dist.log_prob(action)
        dist_entropy: torch.Tensor  = dist.entropy()
            
        return action_logprobs, dist_entropy        

    def _reset_state(self):
        raise NotImplementedError("Not implemented")   


class Critic(nn.Module):
    def __init__(self, map_dim, observation_space, batches: int=1, map_count: int=5, action_dim: int=5, global_critic: bool=False):
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
        self.tanh = nn.Tanh()
        
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
                    #nn.ReLU(),
                    nn.Tanh(),
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
        x = self.tanh(x)
        
        print(x)
        pass
   
    def act(self, state_map_stack: torch.Tensor) -> torch.Tensor: 
        # Get state-value from critic
        #self.test(state_map_stack)
        state_value = self.critic(state_map_stack)
        return state_value
    
    def evaluate(self, state_map_stack):       
        self.critic.train()
        state_values = self.critic(state_map_stack)
            
        return state_values    

    def _reset_state(self):
        raise NotImplementedError("Not implemented")
    
    def forward(self, observation = None, act = None):
        raise NotImplementedError    


# Developed from RAD-A2C https://github.com/peproctor/radiation_ppo
class PFRNNBaseCell(nn.Module):
    """ Parent class for Particle Filter Recurrent Neural Networks """

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
    def resampling(self, particles: torch.Tensor, prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @overload
    def resampling(
        self, particles: Tuple[torch.Tensor, torch.Tensor], prob: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        ...

    def resampling(
        self, particles: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], prob: torch.Tensor
    ) -> Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor]:
        """soft-resampling

        Arguments:
            particles {tensor} -- the latent particles
            prob {tensor} -- particle weights

        Returns:
            Tuple -- particles
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
        if type(particles) == Tuple:
            particles_new = (
                particles[0][flatten_indices],
                particles[1][flatten_indices],
            )
        # PFGRU
        else:
            particles_new = particles[flatten_indices]  # type: ignore

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
    ''' Particle Filter Gated Recurrent Unit '''
    def __init__(
        self,
        input_size: int,
        obs_size: int,
        use_resampling: bool,
        activation: str,        
        num_particles: int = 40,        
        hidden_size: int = 64,
        resamp_alpha: float = 0.7,
    ):
        super().__init__(
            num_particles,
            input_size,
            hidden_size,
            resamp_alpha,
            use_resampling,
            activation,
        )
        
        def mlp(
            sizes: List[Shape],
            activation,
            output_activation=nn.Identity,
            layer_norm: bool = False,
        ) -> nn.Sequential:
            ''' Create a Multi-Layer Perceptron '''
            layers = []
            for j in range(len(sizes) - 1):
                layer = [nn.Linear(sizes[j], sizes[j + 1])] # type: ignore

                if layer_norm:
                    ln = nn.LayerNorm(sizes[j + 1]) if j < len(sizes) - 1 else None  # type: ignore
                    layer.append(ln)  # type: ignore

                layer.append(activation() if j < len(sizes) - 1 else output_activation())
                layers += layer
            if layer_norm and None in layers:
                layers.remove(None) #  type: ignore
            return nn.Sequential(*layers)        

        self.fc_z: nn.Linear = nn.Linear(self.h_dim + self.input_size, self.h_dim)
        self.fc_r: nn.Linear = nn.Linear(self.h_dim + self.input_size, self.h_dim)
        self.fc_n: nn.Linear = nn.Linear(self.h_dim + self.input_size, self.h_dim * 2)

        self.fc_obs: nn.Linear = nn.Linear(self.h_dim + self.input_size, 1)
        self.hid_obs: nn.Sequential = mlp([self.h_dim, 24, 2], nn.ReLU)
        self.hnn_dropout: nn.Dropout = nn.Dropout(p=0)

    def forward(
        self, input_: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """One step forward for PFGRU

        Arguments:
            input_ {tensor} -- the input tensor
            hx {Tuple} -- previous hidden state (particles, weights)

        Returns:
            Tuple -- new tensor
        """
        h0, p0 = hx
        obs_in = input_.repeat(h0.shape[0], 1)
        obs_cat = torch.cat((h0, obs_in), dim=1)

        z = torch.sigmoid(self.fc_z(obs_cat))
        r = torch.sigmoid(self.fc_r(obs_cat))
        n_1 = self.fc_n(torch.cat((r * h0, obs_in), dim=1))

        mu_n, var_n = torch.split(n_1, split_size_or_sections=self.h_dim, dim=1)
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

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        initializer: Callable[[int, int], torch.Tensor] = (
            torch.rand if self.initialize == "rand" else torch.zeros
        )
        h0 = initializer(batch_size * self.num_particles, self.h_dim)
        p0: torch.Tensor = torch.ones(batch_size * self.num_particles, 1) * np.log(
            1 / self.num_particles
        )
        hidden = (h0, p0)
        return hidden


@dataclass
class CCNBase:
    '''
    This is the base class for the Actor-Critic (A2C) Convolutional Neural Network (CNN) architecture. The Actor subclass is an approximator for an Agent's policy.
    The Critic subclass is an approximator for the value function (for more information, see Barto and Sutton's "Reinforcement Learning"). 
    When an observation is fed to this base class, it is transformed into a series of stackable observation heatmaps maps (stored as matrices/tensors). As these maps are fed 
    through the subclass networks, Convolutional and pooling layers train a series of filters that operate on the data and extract features from it. 
    
    An adjustable resolution accuracy variable is computed to indicate the level of accuracy desired. Note: Higher accuracy increases training time.
    
    :param id: (int) Unique identifier key that is used to identify own observations from observation object during map conversions.
    
    :param action_space: (int) Also called action-dimensions. From the environment, get the total number of actions an agent can take. This is used to configure the last 
        linear layer for action-selection in the Actor class.
        
    :param observation_space: (int) Also called state-space or state-dimensions. The dimensions of the observation returned from the environment.For rad-search this will 
        be 11, for the 11 elements of the observation array. This is used for the PFGRU. Future work: make observation-to-map function accomodate differently sized state-spaces.
        
    :param steps_per_epoch: (int) Number of steps of interaction (state-action pairs) for the agent and the environment in each epoch before updating the neural network modules.
        Used for determining stackable map buffer max size.
        
    :param steps_per_episode: (int) Number of steps of interaction (state-action pairs) for the agent and the environment in each episode before resetting the environment. Used
        for resolution accuracy calculation and during normalization of visits-counts map and a multiplier for the log base.  
        
    :param number_of_agents: (int) Number of agents. Used during normalization of visits-counts map and a multiplier for the log base.
    
    :param detector_step_size: (int) Distance an agent can travel in one step (centimeters). Used for inflating scaled coordinates.
    
    :param environment_scale: (int) Value that is being used to normalize grid coodinates for agent. This is later used to reinflate coordinates for increased accuracy, though
        increased computation time, for convolutional networks.
        
    :param bounds_offset: (tuple[float, float]) The difference between the search area and the observation area in the environemnt. This is used to ensure agents can search the 
        entire grid when boundaries are being enforced, not just the obstruction/spawning area. For the CNN, this expands the size of the network to accomodate these extra grid 
        coordinates. This is optional, but for this implementation, to remove this would require adjusting environment to not let agents through to that area when grid boundaries
        are being enforced. Removing this also makes renders look very odd, as agent will not be able to travel to bottom coordinates.
        
    :param grid_bounds: (tuple[float, float]) The grid bounds for the state returned by the environment. This represents the max x and the max y for the scaled coordinates 
        in the rad-search environment (usually (1, 1)). This is used for scaling in the map buffer by the resolution variable.
        
    :param enforce_boundaries: Indicates whether or not agents can walk out of the gridworld. If they can, CNNs must be expanded to include the maximum step count so that all
        coordinates can be encompased in a matrix element.
        
    :param Critic: [Optional] For future work, this allows for the inclusion of a pointer to a global critic
    
    **Important variables that are initialized elsewhere:**
    
    '''
        
    id: int    
    action_space: int
    observation_space: int
    steps_per_epoch: int
    steps_per_episode: int
    number_of_agents: int
    detector_step_size: int  # No default to ensure changes to step size in environment are propogated to this function
    environment_scale: int
    bounds_offset: tuple  # No default to ensure changes to environment are propogated to this function  
    enforce_boundaries: bool  # No default due to the increased computation needs for non-enforced boundaries. Ensures this was done intentionally.
    grid_bounds: Tuple[int, int] = field(default_factory= lambda: (1, 1))
            
    # Initialized elsewhere
    #: Critic/Value network. Allows for critic to be accepted as an argument for global-critic situations
    critic: Union[Critic] = field(default=None)  # type: ignore
    #: Policy/Actor network
    pi: Actor = field(init=False)
    #: Particle Filter Gated Recurrent Unit (PFGRU) for guessing the location of the radiation. This is named model for backwards compatibility reasons.
    model: PFGRUCell = field(init=False)
    #: Buffer that holds map-stacks and converts observations to maps
    maps: MapsBuffer = field(init=False)
    #: Mean Squared Error for loss for critic network
    mseLoss: nn.MSELoss = field(init=False)
    #: How much unscaling to do to reinflate agent coordinates to full representation.
    scaled_offset: float = field(init=False)
    #: An adjustable resolution accuracy variable is computed to indicate the level of accuracy desired. Higher accuracy increases training time.
    resolution_accuracy: float = field(init=False)
    #: Ensures heatmap renders to not overwrite eachother
    render_counter: int = field(init=False)    

    def __post_init__(self):
        # How much unscaling to do. Current environment returnes scaled coordinates for each agent. A resolution_accuracy value of 1 here 
        #  means no unscaling, so all agents will fit within 1x1 grid. To make it less accurate but less memory intensive, reduce the 
        #  number being multiplied by the 1/env_scale. To return to full inflation, change multipier to 1
        multiplier = 0.01
        self.resolution_accuracy = multiplier * 1/self.environment_scale
        if self.enforce_boundaries:
            self.scaled_offset = self.environment_scale * max(self.bounds_offset)                    
        else:
            self.scaled_offset = self.environment_scale * (max(self.bounds_offset) + (self.steps_per_episode * self.detector_step_size))           
        
        # For render
        self.render_counter = 0
        # Initialize buffers and neural networks
        self.maps = MapsBuffer(
                observation_dimension = self.observation_space,
                max_size=self.steps_per_epoch,
                grid_bounds=self.grid_bounds, 
                resolution_accuracy=self.resolution_accuracy,
                offset=self.scaled_offset,
                steps_per_episode=self.steps_per_episode,
                number_of_agents = self.number_of_agents
            )
        
        self.pi = Actor(map_dim=self.maps.map_dimensions, observation_space=self.observation_space, action_dim=self.action_space)#.to(self.maps.buffer.device)
        
        if not self.critic:
            self.critic = Critic(map_dim=self.maps.map_dimensions, observation_space=self.observation_space, action_dim=self.action_space)#.to(self.maps.buffer.device) # TODO these are really slow
            
        self.mseLoss = nn.MSELoss()
        
        # TODO rename this (this is the PFGRU module); naming this "model" for compatibility reasons (one refactor at a time!), but the true model is the maps buffer
        self.model = PFGRUCell(input_size=self.observation_space - 8, obs_size=self.observation_space - 8, use_resampling=True, activation="relu")             
        
    def select_action(self, state_observation: Dict[int, list], id: int, save_map=True) -> ActionChoice:
        ''' Takes a multi-agent observation and converts it to maps and store to a buffer. Also logs the reading at this location
            to resample from in order to estimate a more accurate radiation reading. Then uses the actor network to select an 
            action (and returns action logprobabilities) and the critic network to calculate state-value. 
        '''
        # TODO also currently storing the map to a buffer for later use in PPO; consider moving this to the PPO buffer and PPO class
        
        # If a new observation to be added to maps and buffer, else pull from buffer to avoid overwriting visits count and resampling stale intensity observation.
        with torch.no_grad():        
            if save_map:     
                # Add intensity readings to a list if reading has not been seen before at that location. 
                for observation in state_observation.values():
                    key: Tuple[float, float] = (observation[1], observation[2])
                    intensity: np.floating[Any] = observation[0]
                    if key in self.maps.buffer.readings:
                        if intensity not in self.maps.buffer.readings[key]: # type: ignore 
                            self.maps.buffer.readings[key].append(intensity) # type: ignore 
                    else:
                        self.maps.buffer.readings[key] = [intensity]
                    assert intensity in self.maps.buffer.readings[key], "Observation not recorded into readings buffer" # type: ignore 

                (
                    location_map,
                    others_locations_map,
                    readings_map,
                    visit_counts_map,
                    obstacles_map
                ) = self.maps.observation_to_map(state_observation, id)
                
                # Convert to tensor
                map_stack: torch.Tensor = torch.stack([torch.tensor(location_map), torch.tensor(others_locations_map), torch.tensor(readings_map), torch.tensor(visit_counts_map),  torch.tensor(obstacles_map)]) # Convert to tensor
                
                # Add to mapstack buffer to eventually be converted into tensor with minibatches
                #self.maps.buffer.mapstacks.append(map_stack)  # TODO if we're tracking this, do we need to track the observations?
                self.maps.buffer.coordinate_buffer.append({})
                for i, observation in state_observation.items():
                    self.maps.buffer.coordinate_buffer[-1][i] = Point((observation[1], observation[2]))
                
                #print(state_observation[self.id])
                #observation_key = hash(state_observation[self.id].flatten().tolist)
                self.maps.observation_buffer.append([state_observation[self.id], map_stack]) # TODO Needs better way of matching observation to map_stack
            else:
                with torch.no_grad():
                    map_stack = self.maps.observation_buffer[-1][1]
                
            # Add single batch tensor dimension for action selection
            batched_map_stack: torch.Tensor = torch.unsqueeze(map_stack, dim=0) 
            
            # Get actions and values                          
            action, action_logprob  = self.pi.act(batched_map_stack) # Choose action
            state_value: torch.Tensor = self.critic.act(batched_map_stack)  # size(1)

        return ActionChoice(id=id, action=action.numpy(), action_logprob=action_logprob.numpy(), state_value=state_value.numpy())
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path) # Actor-critic
   
    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)) # Actor-critic
        
    def render(self, savepath: str=getcwd(), save_map: bool=True, add_value_text: bool=False, interpolation_method: str='nearest', epoch_count: int=0):
        ''' Renders heatmaps from maps buffer '''
        if save_map:
            if not path.isdir(str(savepath) + "/heatmaps/"):
                mkdir(str(savepath) + "/heatmaps/")
        else:
            plt.show()                
     
        loc_transposed: npt.NDArray = self.maps.location_map.T # TODO this seems expensive
        other_transposed: npt.NDArray  = self.maps.others_locations_map.T 
        readings_transposed: npt.NDArray  = self.maps.readings_map.T
        visits_transposed: npt.NDArray  = self.maps.visit_counts_map.T
        obstacles_transposed: npt.NDArray  = self.maps.obstacles_map.T
     
        fig, (loc_ax, other_ax, intensity_ax, visit_ax, obs_ax) = plt.subplots(nrows=1, ncols=5, figsize=(30, 10), tight_layout=True)
        
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
        obs_ax.set_title('Obstacles Detected') 
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
                        intensity_ax.text(j, i, readings_transposed[i, j].astype(float).round(2), ha="center", va="center", color="black", size=6)
                    if visits_transposed[i, j] > 0:
                        visit_ax.text(j, i, visits_transposed[i, j].astype(float).round(2), ha="center", va="center", color="black", size=6)
                    if obstacles_transposed[i, j] > 0:
                        obs_ax.text(j, i, obstacles_transposed[i, j].astype(float).round(2), ha="center", va="center", color="black", size=6)                        
        
        fig.savefig(f'{str(savepath)}/heatmaps/heatmap_agent{self.id}_epoch_{epoch_count}-{self.render_counter}.png', format='png')
        fig.savefig(f'{str(savepath)}/heatmaps/heatmap_agent{self.id}_epoch_{epoch_count}-{self.render_counter}.eps', format='eps')

        
        self.render_counter += 1
        plt.close(fig)  # TODO figs arent closing, causes memory issues during large training

    def reset(self):
        ''' Reset entire CNN '''
        self.maps.clear()
        
    def clear_maps(self):
        ''' Just clear maps and buffer for new episode'''
        self.maps.clear_maps()        
