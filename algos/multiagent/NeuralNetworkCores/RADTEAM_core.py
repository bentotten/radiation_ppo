from os import stat, path, mkdir, getcwd
import sys
from math import sqrt, log
from statistics import median

from dataclasses import dataclass, field, asdict
from typing import Any, List, Tuple, Union, NewType, Optional, TypedDict, cast, get_args, Dict, Callable, overload, Union, List, Dict, NamedTuple

import numpy as np
import numpy.typing as npt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt # type: ignore

import warnings
import json

# Maps
#: [New Type] Array indicies to access a GridSquare (x, y). Type: Tuple[float, float]
Point = NewType("Point", Tuple[Union[float, int], Union[float, int]])
#: [New Type] Heatmap - a two dimensional array that holds heat values for each gridsquare. Note: the number of gridsquares is scaled with a resolution accuracy variable. Type: numpy.NDArray[np.float32]
Map = NewType("Map", npt.NDArray[np.float32])
#: [New Type] Mapstack - a Tuple of all existing maps.
MapStack = NewType("MapStack", Tuple[Map, Map, Map, Map, Map, Map])
#: [New Type] Tracks last known coordinates of all agents in order to update them on the current-location and others-locations heatmaps. Type: Dict[str, Dict[int, Point]]
CoordinateStorage = NewType("CoordinateStorage", Dict[int, Point])

# Helpers
#: [Type Alias] Used in multi-layer perceptron in the prediction module (PFGRU). Type: int | Tuple[int, ...]
Shape = Union[int, Tuple[int, ...]]

#: [Global] Detector-obstruction range measurement threshold in cm for inflating step size for obstruction heatmap. Type: float
DIST_TH = 110.0
#: [Global] Toggle for simple value/max normalization vs stdbuffer for radiation intensity map and log-based for visit-counts map. Type: bool
SIMPLE_NORMALIZATION = False
NORMALIZE_RADIATION = False


def calculate_map_dimensions(grid_bounds: Tuple, resolution_accuracy: float, offset: float):
    return (
            int(grid_bounds[0] * resolution_accuracy) + int(offset  * resolution_accuracy),
            int(grid_bounds[1] * resolution_accuracy) + int(offset  * resolution_accuracy)
        )


def calculate_resolution_accuracy(resolution_multiplier: float, scale: float):
    return resolution_multiplier * 1/scale


class ActionChoice(NamedTuple):
    ''' Named Tuple - Standardized response/return template from Actor-Critic for action selection '''
    #: An Agent's unique identifier that serves as a hash key. 
    id: int 
    #: A single integer that represents an agent action in the environment. Stored in a single-element numpy array for processing convinience.
    action: Union[float, int] # size (1)
    #: The log of the policy distribution. Taking the gradient of the log probability is more stable than using the actual density.
    action_logprob: float # size (1)
    #: The estimated value of being in this state. Note: Using GAE for advantage, this is the state-value, not the q-value
    state_value: Union[float, None] # size(1)
    #: Coordinates predicted by the location prediction model (PFGRU). TODO: Implement for CNN
    loc_pred: Union[torch.Tensor, None] = None

    # For compatibility with RAD-PPO
    #: Hidden state (for compatibility with RAD-PPO)
    hiddens: Union[torch.Tensor, None] = None
    
class HeatMaps(NamedTuple):
    ''' Named Tuple - Stores actor and critic heatmaps for a step'''
    actor: torch.Tensor
    critic: torch.Tensor


@dataclass
class IntensityEstimator():
    ''' 
        Hash table that stores radiation intensity levels as seen at each unscaled coordinate into a buffer. Because radiation intensity readings are drawn from a poisson distribution, 
        the more samples that are available, the more accurate the reading. This can be used before standardizing the input for processing in order to get the most accurate radiation reading possible. 
        
        Future Work: Incorporate radionuclide identification module in conjunction with this (Carson et al.)
    '''
    #: Hash table containing explored coordinates (keys) and radiation readings detected there (list of values)
    readings: Dict[Tuple[int, int], List[float]] = field(default_factory=lambda: dict())
    
    # Private
    _min: float = field(default=0.0)  # Minimum radiation reading estimate
    _max: float = field(default=0.0)  # Maximum radiation reading estimate. This is used for normalization in simple normalization mode.
    
    def update(self, key: Tuple[int, int], value: float)-> None:
        ''' 
            Method to add value to radiation hashtable. If key does not exist, creates key and new buffer with value. Also updates running max/min estimate, if applicable.
            Note that the max/min is the ESTIMATE of the true value, not the observed value.
            :param value: (float) Sampled radiation intensity value
            :param key: (Tuple[int, int]) Inflated coordinates where radiation intensity (value) was sampled
        '''
        if self.check_key(key=key):
            self.readings[key].append(value)
        else:
            self.readings[key] = [value]
        
        estimate = self.get_estimate(key)
        if estimate > self._max or self._max == 0: self._set_max(estimate)
        if estimate < self._min or self._min == 0: self._set_min(estimate)

    def _set_max(self, value: float)-> None:
        ''' Method to set the maximum radiation reading estimated thus far.'''
        self._max = value
        
    def _set_min(self, value: float)-> None:
        ''' Method to set the minimum radiation reading estimated thus far.'''
        self._min = value
            
    def get_buffer(self, key: Tuple[int, int])-> List:
        ''' 
            Method to return existing buffer for key. Raises exception if key does not exist.
            :param value: (float) Sampled radiation intensity value
            :param key: (Point) Coordinates where radiation intensity (value) was sampled
        '''      
        if not self.check_key(key=key):
            raise ValueError("Key does not exist")
        return self.readings[key]
        
    def get_estimate(self, key: Tuple[int, int])-> float:
        ''' 
            Method to returns radiation estimate for current coordinates. Raises exception if key does not exist.
            :param key: (Point) Coordinates for desired radiation intensity estimate
        '''      
        if not self.check_key(key=key):
            raise ValueError("Key does not exist")        
        return median(self.readings[key])

    def get_max(self)-> float:
        ''' 
            Method to return the maximum radiation reading estimated thus far. This can be used for normalization in simple normalization mode.
            NOTE: the max/min is the ESTIMATE of the true value, not the observed value.
        '''
        return self._max
        
    def get_min(self)-> float:
        ''' 
            Method to return the minimum radiation reading estimated thus far. 
            NOTE: the max/min is the ESTIMATE of the true value, not the observed value.
        '''
        return self._min

    def check_key(self, key: Tuple[int, int]):
        ''' Method to check if coordinates (key) exist in hashtable '''
        return True if key in self.readings else False
        
    def reset(self):
        ''' Method to reset class members to defaults '''
        self.__init__()  # TODO after attributes have settled, write a proper reset function   


@dataclass
class StatisticStandardization:
    ''' 
    Statistics buffer for standardizing intensity readings from environment (B. Welford, "Note on a method for calculating corrected sums of squares and products").
    Because an Agent collects observations online and does not know the intensity values it will encounter beforehand, it uses this estimated running sample mean and variance instead. 
    '''
    #: Running mean of entire dataset, represented by mu
    mean: float = 0.0
    #: Aggregated squared distance from the mean, represented by M_2
    square_dist_mean: float = 0.0 
    #: Sample variance, represented by s^2. This is used instead of the running variance to reduce bias.
    sample_variance: float = 0.0 
    #: Standard-deviation, represented by sigma
    std: float = 1.0     
    
    #: Count of how many samples have been seen so far
    count: int = 0
            
    # Private
    _max: float = field(default=0.0)  # Maximum radiation reading estimate. This is used for normalization in simple normalization mode.
    _min: float = field(default=0.0)  # Minimum radiation reading estimate. This is used for shifting normalization data in the case of a negative.
    
    def update(self, reading: float) -> None:
        ''' Method to update estimate running mean and sample variance for standardizing radiation intensity readings. Also updates max standardized value 
            for normalization, if applicable.
            
            #. The existing mean is subtracted from the new reading to get the initial delta. 
            #. This delta is then divided by the number of samples seen so far and added to the existing mean to create a new mean.
            #. This new mean is then subtracted from the reading to get new delta.
            #. This new delta is multiplied by the old delta and added to the existing squared distance from the mean. 
            #. To get the sample variance, the new existing squared distance from the mean is divided by the number of samples seen so far minus 1. 
            #. To get the sample standard deviation, the square root of this value is taken.
            
            Thank you to `Wiki - Algorithms for calculating variance <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#cite_ref-5>`_ 
            and `NZMaths - Sample Variance <https://nzmaths.co.nz/category/glossary/sample-variance>`_
            Original: B. Welford, "Note on a method for calculating corrected sums of squares and products"
            
            :param reading: (float) radiation intensity reading
        '''
        assert reading >= 0
        
        self.count += 1
        if self.count == 1:
            self.mean = reading  # For first reading, mean is equal to that reading        
        else:
            mean_new = self.mean + (reading - self.mean) / (self.count) 
            square_dist_mean_new = self.square_dist_mean + (reading - self.mean) * (reading - mean_new)
            self.mean = mean_new
            self.square_dist_mean = square_dist_mean_new
            self.sample_variance = square_dist_mean_new / (self.count - 1)
            self.std = max(sqrt(self.sample_variance), 1)
            
        new_standard = self.standardize(reading=reading)
        if new_standard > self._max: self._max = new_standard
        if new_standard < self._min: self._min = new_standard
             
    def standardize(self, reading: float) -> float:
        ''' 
            Method to standardize input data using the Z-score method by by subtracting the mean and dividing by the standard deviation. 
            Standardizing input data increases training stability and speed. Once the standardization is done, all the features will have a mean of zero and a standard deviation of one, 
            and thus, the same scale. NOTE: Because the first reading will always be standardized to zero, it is important to standardize after a reset before the first step to ensure 
            first steps reading is not wasted.
            
            :param reading: (float) radiation intensity reading
            :return: (float) Standardized radiation reading (z-score) where all existing samples have a std of 1
        '''
        assert reading >= 0
                
        return (reading - self.mean) / self.std

    def get_max(self)-> float:
        ''' Method to return the current maximum standardized sample (updated during update function)'''
        return self._max
    
    def get_min(self)-> float:
        ''' Method to return the current minimum standardized sample (updated during update function)'''
        return self._min

    def reset(self) -> None:
        ''' Method to reset class members to defaults '''
        self.__init__()  # TODO after attributes have settled, write a proper reset function


@dataclass
class Normalizer():
    ''' Normalization methods '''
    _base_check: Any = field(default=None)
    _increment_check: Any = field(default=None)
    
    def normalize(self, current_value: Any, max: Any, min: Union[float, None] = None)-> float:
        ''' 
            Method to do min-max normalization to the range [0,1]. If min is below zero, the data will be shifted by the absolute value of the minimum
            :param current_value: (Any) value to be normalized
            :param max: (Any) Maximum possible
        '''
        # Check for edge cases and invalid inputs
        if current_value == 0:
            return 0.0
        assert max >= current_value, "Value error - Current value is less than max"
        
        # Process min (if current is negative, that is ok as it will be offset by min's absolute value)
        offset: Union[float, int]
        if min:
            assert current_value >= min, "Value error - current is more than max."
            offset = abs(min) if min < 0 else 0
        # If no min provided, assume min == 0 and adjust negative current values
        elif not min:
            min = 0.0
            # If current value is less than 0, it will always be offset to equal 0
            if current_value < 0:
                return 0
            offset =  0

        assert max + offset > 0, "Value error - Max is 0 but current value is not."
        
        result = ((current_value + offset) - (min + offset))  / ((max + offset) - (min + offset))
        assert result >= 0 and result <= 1, "Normalization error"
        return result


    def normalize_incremental_logscale(self, current_value: Any, base: Any, increment_value: int = 2)-> float:
        ''' 
            Method to normalize on a logarithmic scale. This is specifically for a value that increases incrementally every time.
            For TEAM-RAD, every time an agent accesses a grid coordinate, a visits count shadow table is incremented by 1.
            That value is multiplied by the increment_value (here using 2 due to log(1) == 0) and the log is taken. This value
            is then multiplied by 1/ the increment value multiplied by the base in order to put it between 0 and 1. The base is 
            the maximum number of possible steps in an episode multiplied by the number of agents.
            
            :param current_value: (Any) value to be normalized
            :param base: (Any) Maximum possible value (steps per episode multiplied by the number of agents)
            :param increment_value (int): Value from shadow table is expected to increment by this amount every time  
        '''
        assert current_value >= 0 and base > 0 and increment_value > 0, "Value error - input was negative that should not be"
        
        # Warnings for different scales
        if not self._base_check:
            self._base_check = base
        elif self._base_check != base:
            warnings.warn("Base mismatch from first use of normalize_incremental_logscale function! Ensure this was intentional! ")
        if not self._increment_check:
            self._increment_check = increment_value
        if self._increment_check != increment_value:
            warnings.warn("Increment mismatch from first use of normalize_incremental_logscale function! Ensure this was intentional")
            
        result = (log(increment_value + current_value, base)) * 1/log(increment_value * base, base) # Put in range [0, 1]
        assert result >= 0 and result <= 1, f"Normalization error for Result: {result}, Increment_value: {increment_value}, Current value: {current_value}, Base: {base}"
        
        return result

        
@dataclass
class ConversionTools:
    '''
        Stores class objects that assist the conversion from an observation from the environment to a heatmap for processing by the neural networks.
    '''
    #: Stores last coordinates for all agents. This is used to update current-locations heatmaps.
    last_coords: Dict[int, Tuple[int, int]] = field(init=False, default_factory=lambda: dict())
    #: An intensity estimator class that samples every reading and estimates what the true intensity value is
    readings: IntensityEstimator = field(init=False, default_factory=lambda: IntensityEstimator())
    #: Statistics class for standardizing intensity readings from samples from the environment
    standardizer: StatisticStandardization = field(init=False, default_factory=lambda: StatisticStandardization())
    #: Normalization class for adjusting data to be between 0 and 1
    normalizer: Normalizer = field(init=False, default_factory=lambda: Normalizer())
    
    # Reset flag for unit testing
    reset_flag: int = field(init=False, default=0) # TODO switch out for pytest mock

    def reset(self)-> None:
        ''' Method to reset and clear all members '''
        self.last_coords = dict()
        self.readings.reset()
        self.standardizer.reset()
        self.reset_flag += 1 if self.reset_flag < 100 else 1


@dataclass()
class MapsBuffer:        
    ''' Handles all maps operations. Holds the locations maps, readings map, visit counts maps, and obstacles map. 
        Additionally holds toolbox to convert observations into normalized/standardized values and updates maps with these values.
        
        6 maps: 
        
        * Location Map: a 2D matrix showing the individual agent's location.
        
        * Map of Other Locations: a grid showing the number of agents located in each grid element (excluding current agent).
        
        * Readings map: a grid of the last reading collected in each grid square - unvisited squares are given a reading of 0.
        
        * Visit Counts Map: a grid of the number of visits to each grid square from all agents combined.
        
        * Obstacles Map: a grid of how far from an obstacle each agent was when they detected it
        
        :param observation_dimension: (int) Shape of state space. This is how many elements are in the observation array that is returned from the environment. For Rad-Search, this should be 11.
        :param steps_per_episode: (int) Maximum steps per episode. This is used for the base calculation for the log_normalizing for visit counts in visits map.
        :param number_of_agents: (int) Total number of agents. This is used for the base calculation for the log_normalizing for visit counts in visits map.
        
        :param grid_bounds: (tuple) Initial grid boundaries for the scaled x and y coordinates observed from the environment. For Rad-Search, these are scaled to the range [0, 1], 
            so the default bounds are 1x1. Defaults to (1, 1).
            
        :param resolution_accuracy: This is the value to multiply grid bounds and agent coordinates by to inflate them to a more useful size. This is calculated by the CNNBase class
            and indicates the level of accuracy desired. For this class, its function is to inflate grid coordinates to the appropriate size in order to convert an observation into a
            map stack. Defaults to 22, where the graph is inflated to a 22x22 grid + offset.
            
        :param offset: Scaled offset for when boundaries are different than "search area". This parameter increases the number of nodes around the "search area" to accomodate possible 
            detector positions in the bounding-box area that are not in the search area. Further clarification: In the Rad-Search environment, the bounding box indicates the rendered
            grid area, however the search area is where agents, sources, and obstacles spawn. Due to limits with the visilibity library and obstacle generation, there needed to be two 
            grids to contain them. Default is 0.22727272727272727 to increase the grid size to 27.
            
        :param obstacle_state_offset: Number of initial elements in state return that do not indicate there is an obstacle. First element is intensity, second two are x and y coords. 
            Defaults to 3, with the 4th element indicating the beginning of obstacle detections. This defaults to 3.       
            
        :param resolution_multiplier: The multiplier used to create the resolution accuracy. Used to indicate how maps should be reset, for efficiency.   
    '''
    #TODO change env return to a named tuple instead.   
    # Inputs
    observation_dimension: int  
    steps_per_episode: int # 
    number_of_agents: int # Used for normalizing visists count in visits map
            
    # Option Parameters
    grid_bounds: Tuple = field(default_factory= lambda: (1,1)) 
    resolution_accuracy: float = field(default=22.0) 
    resolution_multiplier: float = field(default=0.01)   # If wrong, may just be slow for map resets 
    offset: float = field(default=0.22727272727272727) 
    obstacle_state_offset: int = field(default=3)     
    
    # Initialized elsewhere
    #: Maximum x bound in observation maps, used to fill observation maps with zeros during initialization. This defaults to 27.
    x_limit_scaled: int = field(init=False) 
    #: Maximum y bound in observation maps, used to fill observation maps with zeros during initialization. This defaults to 27. 
    y_limit_scaled: int = field(init=False)
    #: Actual dimensions of each map. These need to match the convolutional setup in order to not cause errors when being processed by the linear layer during action selection (and state-value estimating).
    #:  this defaults to (27, 27).
    map_dimensions: Tuple[int, int] = field(init=False)
    #: The base for log() for visit count map normalization. This is equivilant to the maximum number of steps possible in one location, T*n where T is the maximum number of steps per episode and n is the 
    #:  number of agents.
    base: int = field(init=False) 
        
    # Blank Maps
    #: Number of maps
    map_count: int = field(init=False, default=6)
    # Combined Location Map: a 2D matrix showing all agent locations. Used for the critic
    combined_location_map: Map = field(init=False) 
    #: Location Map: a 2D matrix showing the individual agent's location.
    location_map: Map = field(init=False) 
    #: Map of Other Locations: a grid showing the number of agents located in each grid element (excluding current agent).    
    others_locations_map: Map = field(init=False)
    #: Readings map: a grid of the last estimated reading in each grid square - unvisited squares are given a reading of 0.
    readings_map: Map = field(init=False)  
    #: Obstacles Map: a grid of how far from an obstacle each agent was when they detected it
    obstacles_map: Map = field(init=False) 
    #: Visit Counts Map: a grid of the number of visits to each grid square from all agents combined.
    visit_counts_map: Map = field(init=False)  
    
    #: Shadow hashtable for visits counts map, increments a counter every time that location is visited. This is used during logrithmic normalization to reduce computational complexity and python floating 
    #:  point precision errors, it is "cheaper" to calculate the log on the fly with a second sparce matrix than to inflate a log'd number. Stores tuples (x, y, 2(i)) where i increments every hit.
    visit_counts_shadow: Dict = field(default_factory=lambda: dict())     
    
    #: Locations matrix for all tracked locations. Used for fast map clearing after reset
    locations_matrix: List = field(default_factory=lambda: list())
    
    # Buffers
    #: Data preprocessing tools for standardization, normalization, and estimating values in order to input into a observation map.
    tools: ConversionTools = field(default_factory=lambda: ConversionTools())
    #: This needs to be moved to PPO, but currently holds a set of stackable maps in an experience buffer to later be used for updates.
    observation_buffer: List = field(default_factory=lambda: list())  # TODO move to PPO buffer

    # Reset flag for unit testing
    reset_flag: int = field(init=False, default=0) # TODO switch out for pytest mock

    def __post_init__(self)-> None:
        # Set logrithmic base for visits counts normalization
        self.base = ((self.steps_per_episode+1) * self.number_of_agents)  # Extra observation is for the "last step" where the next state value is used to bootstrap rewards
        
        # Calculate map x and y bounds for observation maps
        self.map_dimensions = calculate_map_dimensions(grid_bounds=self.grid_bounds, resolution_accuracy=self.resolution_accuracy, offset= self.offset)
        self.x_limit_scaled: int = self.map_dimensions[0]
        self.y_limit_scaled: int = self.map_dimensions[1]
        
        # Initialize maps and buffer
        self._reset_maps() # Initialize maps
        self.reset()         
        
    def reset(self)-> None:
        ''' Method to clear maps and reset matrices. If seeing errors in maps, try a full reset with full_reset() '''
        
        self._clear_maps()
        
        self.locations_matrix.clear()
        self.visit_counts_shadow.clear()        
        self.tools.reset() 
        self.reset_flag += 1 if self.reset_flag < 100 else 1
        
        
    def full_reset(self)-> None:
        ''' Obsolete method to reinitialize maps and reset matrices. Slower than reset()'''
        
        self._reset_maps()
        
        self.locations_matrix.clear()
        self.visit_counts_shadow.clear()        
        self.tools.reset()         

    def observation_to_map(self, observation: Dict[int, np.ndarray], id: int) -> MapStack:  
        '''
        Method to process observation data into observation maps from a dictionary with agent ids holding their individual 11-element observation. Also updates tools.
        
        :param observation: (dict) Dictionary of agent IDs and their individual observations from the environment.
        :param id: (int) Current Agent's ID, used to differentiate between the agent location map and the other agents map.
        :return: Returns a tuple of five 2d map arrays.
        '''
        
        # Add intensity readings to readings buffer for estimates
        for obs in observation.values():
            key: Tuple[int, int] = self._inflate_coordinates(obs)
            intensity: np.floating[Any] = obs[0]
            self.tools.readings.update(key=key, value=float(intensity))        
                    
        for agent_id in observation:
            # Fetch scaled coordinates
            inflated_agent_coordinates: Tuple[int, int] = self._inflate_coordinates(single_observation=observation[agent_id])
            #inflated_last_coordinates: Union[Tuple[int, int], None] = self._inflate_coordinates(single_observation=self.tools.last_coords[agent_id]) if agent_id in self.tools.last_coords.keys() else None
            last_coordinates: Union[Tuple[int, int], None] = self.tools.last_coords[agent_id] if agent_id in self.tools.last_coords.keys() else None
            self.locations_matrix.append(inflated_agent_coordinates)
            
            # Update Locations maps
            if id == agent_id:
                self._update_current_agent_location_map(current_coordinates=inflated_agent_coordinates, last_coordinates=last_coordinates)
                self._update_combined_agent_locations_map(current_coordinates=inflated_agent_coordinates, last_coordinates=last_coordinates)
            else:   
                self._update_other_agent_locations_map(current_coordinates=inflated_agent_coordinates, last_coordinates=last_coordinates)                
                self._update_combined_agent_locations_map(current_coordinates=inflated_agent_coordinates, last_coordinates=last_coordinates)                
                
            # Readings and Visits counts maps
            self._update_readings_map(coordinates=inflated_agent_coordinates)
            self._update_visits_count_map(coordinates=inflated_agent_coordinates)
            
            # Detected Obstacles map
            if np.count_nonzero(observation[agent_id][self.obstacle_state_offset:]) > 0:
                self._update_obstacle_map(coordinates=inflated_agent_coordinates, single_observation=observation[agent_id])
                
            # Update last coordinates
            #self.tools.last_coords[agent_id] = Point((observation[agent_id][1], observation[agent_id][2]))          
            self.tools.last_coords[agent_id] = inflated_agent_coordinates
        
        return MapStack((self.location_map, self.others_locations_map, self.readings_map, self.visit_counts_map, self.obstacles_map, self.combined_location_map))
            
    def _clear_maps(self)-> None:
        ''' Clear values stored in maps from coordinates stored in sparse matrices '''
        
        if self.resolution_multiplier < 0.1:
            # For sparse matrices, reset via saved coordinates
            for coords in self.tools.last_coords.values():
                #inflated_last_coordinates = self._inflate_coordinates(single_observation=coords)
                inflated_last_coordinates = coords
                self.combined_location_map[inflated_last_coordinates] = 0 
                self.location_map[inflated_last_coordinates] = 0 
                self.others_locations_map[inflated_last_coordinates] = 0                
            
            # Reinitialize non-sparse matrices 
            self.readings_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))                      
            self.visit_counts_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))
            self.obstacles_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32)) 
            
        else:
            for k in self.locations_matrix:
                self.combined_location_map[k] = 0 
                self.location_map[k] = 0 
                self.others_locations_map[k] = 0    
                self.readings_map[k] = 0    
                self.obstacles_map[k] = 0
                self.visit_counts_map[k] = 0                 
            
        assert (self.obstacles_map.max() == 0 and self.obstacles_map.min() == 0)                
        assert (self.readings_map.max() == 0 and self.readings_map.min() == 0)                
        assert (self.others_locations_map.max() == 0 and self.others_locations_map.min() == 0)                  
        assert (self.location_map.max() == 0 and self.location_map.min() == 0)            
        assert (self.combined_location_map.max() == 0 and self.combined_location_map.min() == 0)              
        assert (self.visit_counts_map.max() == 0 and self.visit_counts_map.min() == 0) 
        
    def _reset_maps(self)-> None:
        ''' Fully reinstatiate maps'''
        self.combined_location_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))         
        self.location_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32)) 
        self.others_locations_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  
        self.readings_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))  
        self.obstacles_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32)) 
        self.visit_counts_map: Map = Map(np.zeros(shape=(self.x_limit_scaled, self.y_limit_scaled), dtype=np.float32))
        self.visit_counts_shadow.clear()
                              
    def _inflate_coordinates(self, single_observation: Union[np.ndarray, Point])-> Tuple[int, int]:
        ''' 
            Method to take a single observation state, extracts the coordinates, then inflates them to the resolution accuracy specified during initialization. Also works with tuple of deflated coordinates.
            
            :param singe_observation: (np.ndarray, tuple) single observation state from a single agent observation OR single pair of deflated coordinates
            :return: (Tuple[int, int]) Inflated coordinates
        '''
        # Calculate current agent inflated location        
        result: Tuple[int, int]
        if isinstance(single_observation, np.ndarray):
            result = (int(single_observation[1] * self.resolution_accuracy), int(single_observation[2] * self.resolution_accuracy))
        elif type(single_observation) == tuple:
            result = (int(single_observation[0] * self.resolution_accuracy), int(single_observation[1] * self.resolution_accuracy))
        else:
            raise ValueError("Unsupported type for observation parameter")
        return result        
        
    def _deflate_coordinates(self, single_observation: Union[np.ndarray, Tuple[int, int]])-> Point:
        ''' 
            Method to take a single observation state that has already been adjusted to the resolution accuracy specified during initialization, then extracts the coordinates, then deflates them back to their
            normalized inital values. Also works with tuple of inflated coordinates.
            
            :param singe_observation: (np.ndarray, tuple) single observation state from a single agent observation OR single pair of inflated coordinates
            :return: (Tuple[int, int]) deflated coordinates
        '''
        # Calculate current agent inflated location        
        result: Point
        if isinstance(single_observation, np.ndarray):
            result = Point((float(single_observation[1] / self.resolution_accuracy), float(single_observation[2] / self.resolution_accuracy)))
        elif type(single_observation) == tuple:
            result = Point((float(single_observation[0] / self.resolution_accuracy), float(single_observation[1] / self.resolution_accuracy)))
        else:
            raise ValueError("Unsupported type for observation parameter")
        return result         

    def _update_combined_agent_locations_map(self, current_coordinates: Tuple[int, int], last_coordinates: Union[Tuple[int, int], None])-> None:
        ''' 
            Method to update the other-agent locations observation map. If prior location exists, this is reset to zero. Note: updates one location at a time, not in a batch
            
            :param id: (int) ID of current agent being processed
            :param current_coordinates: (Tuple[int, int]) Inflated current location of agent to be processed
            :param last_coordindates: (Tuple[int, int]) Inflated previous location of agent to be processed. Note: These must be ints.
            :return: None
        '''           
        if last_coordinates:
            self.combined_location_map[last_coordinates[0]][last_coordinates[1]] -= 1 
            # In case agents are at same location, usually the start-point, just ensure was not negative.
            assert self.combined_location_map[last_coordinates[0]][last_coordinates[1]] > -1, "Location map grid coordinate reset where agent was not present"
        else:
            assert self.combined_location_map.max() < self.number_of_agents, "Location exists on map however no last coordinates buffer passed for processing."              
            
        self.combined_location_map[current_coordinates[0]][current_coordinates[1]] += 1                  
        
    def _update_current_agent_location_map(self, current_coordinates: Tuple[int, int], last_coordinates: Union[Tuple[int, int], None])-> None:
        ''' 
            Method to update the current agents location observation map. If prior location exists, this is reset to zero.
            
            :param current_coordinates: (Tuple[int, int]) Inflated current location of agent
            :param last_coordindates: (Tuple[int, int]) Inflated previous location of agent. Note: These must be ints.
            :return: None
        '''
        if last_coordinates:
            self.location_map[last_coordinates[0]][last_coordinates[1]] -= 1 
            assert self.location_map[last_coordinates[0]][last_coordinates[1]] > -1, "location_map grid coordinate reset where agent was not present. The map location that was reset was already at 0."  # type: ignore # Type will already be a float
        else:
            assert self.location_map.max() == 0.0, "Location exists on map however no last coordinates buffer passed for processing."        
            
        self.location_map[current_coordinates[0]][current_coordinates[1]] = 1
        assert self.location_map.max() == 1, "Location was updated twice for single agent"

    def _update_other_agent_locations_map(self, current_coordinates: Tuple[int, int], last_coordinates: Union[Tuple[int, int], None])-> None:
        ''' 
            Method to update the other-agent locations observation map. If prior location exists, this is reset to zero. Note: updates one location at a time, not in a batch
            
            :param id: (int) ID of current agent being processed
            :param current_coordinates: (Tuple[int, int]) Inflated current location of agent to be processed
            :param last_coordindates: (Tuple[int, int]) Inflated previous location of agent to be processed. Note: These must be ints.
            :return: None
        '''     
        if last_coordinates:
            self.others_locations_map[last_coordinates[0]][last_coordinates[1]] -= 1 
            # In case agents are at same location, usually the start-point, just ensure was not negative.
            assert self.others_locations_map[last_coordinates[0]][last_coordinates[1]] > -1, "Location map grid coordinate reset where agent was not present"
        else:
            assert self.others_locations_map.max() < self.number_of_agents, "Location exists on map however no last coordinates buffer passed for processing."     
                    
        self.others_locations_map[current_coordinates[0]][current_coordinates[1]] += 1  # Initial agents begin at same location             

    def _update_readings_map(self, coordinates: Tuple[int, int], key: Union[Tuple[int, int], None] = None)-> None:
        ''' 
            Method to update the radiation intensity observation map with single observation. If prior location exists, this is overwritten with the latest estimation.
            
            :param id: (int) ID of current agent being processed
            :param coordinates: (Tuple[int, int]) Inflated current location of agent to be processed
            :param key: (Point) Deflated current location to be used as a key for the readings hashtable
            :return: None
        '''      
        # Estimate true radiation reading
        estimate: float        
        if key:
            estimate = self.tools.readings.get_estimate(key=key)
        else:
            estimate = self.tools.readings.get_estimate(key=coordinates)
        
        # Standardize radiation reading
        self.tools.standardizer.update(estimate)                
        reading = self.tools.standardizer.standardize(estimate)
        
        # Normalize radiation reading and save to map
        if NORMALIZE_RADIATION:
            reading = self.tools.normalizer.normalize(current_value=reading, max=self.tools.standardizer.get_max(), min=self.tools.standardizer.get_min())
        
        # Save to map
        self.readings_map[coordinates[0]][coordinates[1]] = reading        

    def _update_visits_count_map(self, coordinates: Tuple[int, int])-> None:
        ''' 
            Method to update the visits count observation map. Increments in a logarithmic fashion.
            
            :param id: (int) ID of current agent being processed
            :param coordinates: (Tuple[int, int]) Inflated current location of agent to be processed
            :return: None
        ''' 
        # If visited before, fetch counter from shadow table, else create shadow table entry
        if coordinates in self.visit_counts_shadow.keys():
            current = self.visit_counts_shadow[coordinates]
            self.visit_counts_shadow[coordinates] += 2
        else:
            current = 0
            self.visit_counts_shadow[coordinates] = 2

        if SIMPLE_NORMALIZATION:
            normalized_value = self.tools.normalizer.normalize(current_value=current, max=self.base)
        else: 
            normalized_value = self.tools.normalizer.normalize_incremental_logscale(current_value=current, base=self.base, increment_value=2)
        
        # Save to  matrix
        self.visit_counts_map[coordinates[0]][coordinates[1]] = normalized_value
        
        # Sanity warning
        if self.visit_counts_map[coordinates[0]][coordinates[1]] == 1.0: warnings.warn("Visit count is normalized to 1; either all Agents did not move entire episode or there is a normalization error")
        if self.visit_counts_map.max() > 1.0: raise Exception(f"Visit count max is normalized greater than 1: {self.visit_counts_map.max()}")
        
    def _update_obstacle_map(self, coordinates: Tuple[int, int], single_observation: np.ndarray)-> None:
        ''' 
            Method to update the obstacle detection observation map. Renders headmap of Agent distance from obstacle. Using two parameters removes unnecessary reinflation step.
            
            :param id: (int) ID of current agent being processed
            :param singe_observation: (np.ndarray, tuple) single observation state from a single agent observation OR single pair of inflated coordinates
            :return: None
        '''
        # Fill in map
        for detection in single_observation[self.obstacle_state_offset::]:
            if detection != 0:
                self.obstacles_map[coordinates[0]][coordinates[1]] = detection


class Actor(nn.Module):
    ''' 
        In deep reinforcement learning, an Actor is a neural network architecture that represents an agent's control policy. Each agent is outfit with their own Actor. 
        Learning aims to optimize a policy gradient. For RAD-TEAM, the Actor consists of a convolutional neural Network Architecture that includes two convolution layers,
        a maxpool layer, and three fully connected layers to distill the previous layers into an action probability distribution. Following Alagha et al.'s multi-agent CNN 
        architecture, each of the nodes are activated with ReLU (to dodge vanishing gradient problem) and the probability distribution is calculated using the softmax function.

        The Actor takes a stack of observation maps (numerical matrices/tensors) and processes them with the neural network architecture. Convolutional and pooling layers 
        train a series of filters that operate on the data and extract features from it. These features are then distilled through linear layers to produce an array that 
        contains probabilities, where each element cooresponds to an action.
        
        This Actor expects the input tensor shape: (batch size, number of channels, height of grid, width of grid) where
        *. batch size: How many mapstacks (default 1)
        *. number of channels: Number of input maps in each stack (default 5)
        *. Height: Map height (from map_dim)
        *. Width: Map width (from map_dim)
        
        The network for RAD-TEAM expects mapstacks in tensor form, where the shape is [b, n, x, y]. Here b is the number of batches, n is the number of maps, x is the map 
        width and y is the map height.
        
        :param map_dim: (Tuple[int, int]) Map dimensions (discrete). This is the scaled height and width that each observation map will be. NOTE: dimensions must be equal to each other,
            discrete, and real.
        :param batches: (int) Number of observation mapstacks to be processed - each step in the environment yields one mapstack. Defaults to 1.
        :param map_count: (int) Number of observation maps in a single mapstack. Defaults to 5. 
        :param action_dim: (int) Number of actions to choose from. Defaults to 8.         
    '''
    def __init__(self, map_dim: Tuple[int, int], batches: int=1, map_count: int=5, action_dim: int=8)-> None:
        super(Actor, self).__init__()

        assert map_dim[0] > 0 and map_dim[0] == map_dim[1], 'Map dimensions mismatched. Must have equal x and y bounds.'
        
        # Set for later fetching for global critic
        self.batches = batches
        
        channels: int = map_count
        pool_output: int = int(((map_dim[0]-2) / 2) + 1) # Get maxpool output height/width and floor it
        
        # Actor network
        self.step1 = nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, stride=1, padding=1)  
        self.relu = nn.ReLU()
        self.step2 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.step3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1) 
        self.step4 = nn.Flatten(start_dim=0, end_dim= -1) 
        self.step5 = nn.Linear(in_features=16 * batches * pool_output * pool_output, out_features=32) 
        self.step6 = nn.Linear(in_features=32, out_features=16) 
        self.step7 = nn.Linear(in_features=16, out_features=action_dim) 
        self.softmax = nn.Softmax(dim=0)  # Put in range [0,1] 

        self.actor = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, stride=1, padding=1),  # output tensor with shape (5, 8, Height, Width)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output tensor with shape (4, 8, 2, 2). Output height and width is floor(((Width - Size)/ Stride) +1)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1),  # output tensor with shape (4, 16, 2, 2)
            nn.ReLU(),
            nn.Flatten(start_dim=0, end_dim= -1),  # output tensor with shape (1, x)
            nn.Linear(in_features=16 * batches * pool_output * pool_output, out_features=32), # output tensor with shape (32)
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16), # output tensor with shape (16)
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=action_dim), # output tensor with shape (8)
            nn.Softmax(dim=0)  # Put in range [0,1]
        )
        
    def _test(self, state_map_stack)-> None:
        ''' Deconstructed Actor layers to assist with debugging '''
        
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
   
    def act(self, observation_map_stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
            Method that selects action from action probabilities returned by actor network.
            
            :param observation_map_stack: (Tensor) Contains five stacked observation maps. Should be in shape [batch_size, number of maps, map width, map height]. 
            :return: (Tensor, Tensor) Returns the action selected (tensor(1)) and the log-probability for that action (tensor(1)).
        '''
        #: Raw action probabilities for each available action for this particular observation
        action_probs: torch.Tensor = self.actor(observation_map_stack)
        #: Convert raw action probabilities into a probability distribution that sums to 1.
        dist = Categorical(action_probs)
        #: Sample an action from the action probability distribution
        action: torch.Tensor = dist.sample()
        #: Take the log probability of the action. This can be used to compute the policy gradient used during updates; 
        #:  Taking the gradient of the log probability is more stable than using the actual density
        action_logprob: torch.Tensor = dist.log_prob(action)
        
        return action, action_logprob

    def forward(self, observation_map_stack: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        ''' 
            Method that takes the observation and returns all action probabilities. 
            :param observation_map_stack: (Tensor) Contains five stacked observation maps. Should be in shape [batch_size, number of maps, map width, map height]. 
            :return: (Tensor, Tensor) Returns probability distribution for all actions (tensor(action_space)) and the entropy for the action distribution (tensor(1)).
        '''
        #: Raw action probabilities for each available action for this particular observation        
        action_probs: torch.Tensor = self.actor(observation_map_stack)
        #: Convert raw action probabilities into a probability distribution that sums to 1.        
        dist = Categorical(action_probs)
        #: Degree of randomness in distribution        
        dist_entropy: torch.Tensor  = dist.entropy()
        return dist, dist_entropy
    
    def get_action_information(self, state_map_stack: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' Method that gets the action logprobabilities for an observation mapstack and calculates a particular actions entropy.
        
            :param observation_map_stack: (Tensor) Contains five stacked observation maps. Should be in shape [batch_size, number of maps, map width, map height]. 
            :param action: (Tensor) The the action taken (tensor(1))
            :return: (Tensor, Tensor) Returns the log-probability for the passed-in action (tensor(1)) and the entropy for the action distribution (tensor(1)).
        '''
        #: Raw action probabilities for each available action for this particular observation        
        action_probs: torch.Tensor = self.actor(state_map_stack)
        #: Convert raw action probabilities into a probability distribution that sums to 1.        
        dist = Categorical(action_probs)
        #: Take the log probability of the action. This is used to compute loss for the policy gradient update; 
        #:  Taking the gradient of the log probability is more stable than using the actual density        
        action_logprobs: torch.Tensor  = dist.log_prob(action)
        #: Degree of randomness in distribution
        dist_entropy: torch.Tensor  = dist.entropy()
            
        return action_logprobs, dist_entropy        

    def put_in_training_mode(self)-> None:
        ''' Method to put actor in train mode. This adds dropout, batch normalization, and gradients.'''
        self.actor.train()
        
    def put_in_evaluation_mode(self)-> None:
        ''' Method to put actor in eval mode. This disables dropout, batch normalization, and gradients.'''    
        self.actor.eval()  

    def reset_output_layers(self)-> None:
        ''' Method to only reset weights and biases in output layers. This removes the learning needed to pick a correct action for a prior episode. '''
        for layer in self.actor:
            if isinstance(layer, nn.Linear):            
                layer.reset_parameters()                
        
    def reset_all_hidden(self)-> None:
        ''' Method to completely reset all weights and biases in all hidden layers ''' 
        for layer in self.actor:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):            
                layer.reset_parameters()
                
    def save_model(self, checkpoint_path:str)-> None:
        ''' Save model to a file'''
        torch.save(self.state_dict(), f"{checkpoint_path}/actor.pt")   

    def load_model(self, checkpoint_path:str)-> None:
        assert path.isfile(f"{checkpoint_path}/actor.pt"), "Model does not exist"
        self.load_state_dict(torch.load(f"{checkpoint_path}/actor.pt", map_location=lambda storage, loc: storage)) 
                

class Critic(nn.Module):
    ''' 
        In deep reinforcement learning, a Critic is a neural network architecture that approximates the state-value V^pi(s) for the policy pi. This indicates how "good it is" to be
        in any state. For RAD-TEAM, we use the Generalized Advantage Estimator (GAE) that does not require the Q-value, only the state-value. RAD-Team is set up to work with both a 
        per-agent critic and a global critic. A global critic can still work, in spite of the independent agent policies, due to the team-based reward. Learning aims to minimize
        The mean-squared error. For RAD-TEAM, the Critic consists of a convolutional neural Network Architecture that includes two convolution layers, a maxpool layer, and three 
        fully connected layers to distill the previous layers into a single state-value. Following Alagha et al.'s multi-agent CNN architecture, each of the nodes are activated with
        ReLU (to dodge vanishing gradient problems).

        The Critic, like the actor takes a stack of observation maps (numerical matrices/tensors) and processes them with the neural network architecture. Convolutional and pooling 
        layers train a series of filters that operate on the data and extract features from it. These features are then distilled through linear layers to produce a single state-value.
        Note: Critic is estimating State-Value here, not Q-Value.
        
        This Critic expects the input tensor shape: (batch size, number of channels, height of grid, width of grid) where
        *. batch size: How many mapstacks (default 1)
        *. number of channels: Number of input maps in each stack (default 4)
        *. Height: Map height (from map_dim)
        *. Width: Map width (from map_dim)
        
        The network for RAD-TEAM expects mapstacks in tensor form, where the shape is [b, n, x, y]. Here b is the number of batches, n is the number of maps, x is the map 
        width and y is the map height.
        
        :param map_dim: (Tuple[int, int]) Map dimensions (discrete). This is the scaled height and width that each observation map will be. NOTE: dimensions must be equal to each other,
            discrete, and real.
        :param batches: (int) Number of observation mapstacks to be processed - each step in the environment yields one mapstack. Defaults to 1.
        :param map_count: (int) Number of observation maps in a single mapstack. Defaults to 4.
    '''    
    def __init__(self, map_dim, batches: int=1, map_count: int=4):
        super(Critic, self).__init__()    
            
        assert map_dim[0] > 0 and map_dim[0] == map_dim[1], 'Map dimensions mismatched. Must have equal x and y bounds.'
        
        self.map_count = map_count
        channels: int = map_count
        pool_output: int = int(((map_dim[0]-2) / 2) + 1) # Get maxpool output height/width and floor it

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
                )

    def _test(self, state_map_stack)-> None: 
        ''' Method to test individual critic layers and examine their sizes '''
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
        print(x)
        pass
    
    def forward(self, observation_map_stack: torch.Tensor)-> torch.Tensor:
        ''' 
            Get the state-value for a given state-observation from the environment 
            :param observation_map_stack: (Tensor) Contains five stacked observation maps. Should be in shape [batch_size, number of maps, map width, map height]. 
            :return: (Tensor) Returns state-value for the given observation (tensor(1)).        
        '''
        return self.critic(observation_map_stack)

    def act(self, observation_map_stack: torch.Tensor) -> torch.Tensor: 
        ''' 
            Alias for "Forward()". Get the state-value for a given state-observation from the environment 
            :param observation_map_stack: (Tensor) Contains five stacked observation maps. Should be in shape [batch_size, number of maps, map width, map height]. 
            :return: (Tensor) Returns state-value for the given observation (tensor(1)).                    
        '''
        return self.forward(observation_map_stack=observation_map_stack) 

    def put_in_training_mode(self)-> None:
        self.critic.train()
        
    def put_in_evaluation_mode(self)-> None:
        self.critic.eval()        

    def reset_output_layers(self)-> None:
        ''' Method to only reset weights and biases in output layers. This removes the learning needed to pick a correct action for a prior episode. '''
        for layer in self.critic:
            if isinstance(layer, nn.Linear):            
                layer.reset_parameters()                
        
    def reset_all_hidden(self)-> None:
        ''' Method to completely reset all weights and biases in all hidden layers ''' 
        for layer in self.critic:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):            
                layer.reset_parameters()
                
    def save_model(self, checkpoint_path:str)-> None:
        ''' Save model to a file'''
        torch.save(self.state_dict(), f"{checkpoint_path}/critic.pt")

    def load_model(self, checkpoint_path:str)-> None:
        assert path.isfile(f"{checkpoint_path}/critic.pt"), "Model does not exist"        
        self.load_state_dict(torch.load(f"{checkpoint_path}/critic.pt", map_location=lambda storage, loc: storage))

    def is_mock_critic(self)-> bool:
        return False

class EmptyCritic():
    ''' 
        This is an empty critic object that simulates a critic for compatibility during evaluation runs in order to avoid the volume of conditional statements required otherwise.
    '''
    
    def __init__(self, map_dim: Union[int, None] = None, batches: int=1, map_count: int=4):
        super(EmptyCritic, self).__init__()
        self.training: bool = False
        self.map_count = map_count
        return 
        
    def _test(self, state_map_stack)-> None: 
        return
    
    def forward(self, observation_map_stack: torch.Tensor)-> None:
        return

    def act(self, observation_map_stack: torch.Tensor)-> None:
        raise ValueError("Attempting act on empty critic object!")

    def put_in_training_mode(self)-> None:
        return
        
    def put_in_evaluation_mode(self)-> None:
        return

    def reset_output_layers(self)-> None:
        return              
        
    def reset_all_hidden(self)-> None:
        return  
    
    def save_model(self, checkpoint_path:str)-> None:
        return
    
    def load_model(self, checkpoint_path:str)-> None:
        return    
    
    def eval(self)-> None:
        return   
    
    def train(self)-> None:
        return
    
    def is_mock_critic(self)-> bool:
        return True    

#TODO Add to new map
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

    def reparameterize(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Implements the reparameterization trick introduced in [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

        Arguments:
            mean {tensor} -- learned mean
            var {tensor} -- learned variance

        Returns:
            tensor -- sample
        """
        std: torch.Tensor = F.softplus(var)
        eps: torch.Tensor = torch.FloatTensor(std.shape).normal_()
        return mean + eps * std

#TODO Add to new map
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

    def save_model(self, checkpoint_path: str)-> None:
        torch.save(self.state_dict(), f"{checkpoint_path}/predictor.pt")       
        
    def load_model(self, checkpoint_path:str)-> None:
        assert path.isfile(f"{checkpoint_path}/predictor.pt"), "Model does not exist"                
        self.load_state_dict(torch.load(f"{checkpoint_path}/predictor.pt", map_location=lambda storage, loc: storage))        

@dataclass
class CNNBase:
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
    :resolution_multiplier: Multiplier used to indicate how accurate scaling should be for heatmaps. A value of 1 will represent the original accuracy presented by the environment.
        By default, this is downsized to 0.01 in order to reduce the number of trainable parameters in the heatmaps, leading to faster convergence. Only for very small search spaces
        is it recommended to use full accuracy - heatmaps indicate "area of interest trends", the values themselves are less important.
    :param enforce_boundaries: Indicates whether or not agents can walk out of the gridworld. If they can, CNNs must be expanded to include the maximum step count so that all
        coordinates can be encompased in a matrix element.
    :param GlobalCritc: (Critic) Actual global critic object
    :param no_critic: (bool) Used to indicate if A2C should be instatiated with a empty, stand-in critic object. This is used during evaluation.
    
    **Important variables that are initialized elsewhere:**
    
    '''
        
    id: int    
    action_space: int
    observation_space: int
    steps_per_episode: int
    number_of_agents: int
    detector_step_size: int  # No default to ensure changes to step size in environment are propogated to this function
    environment_scale: int
    bounds_offset: tuple  # No default to ensure changes to environment are propogated to this function  
    enforce_boundaries: bool  # No default due to the increased computation needs for non-enforced boundaries. Ensures this was done intentionally.
    grid_bounds: Tuple[int, int] = field(default_factory= lambda: (1, 1))
    resolution_multiplier: float = field(default=0.01)
    GlobalCritic: Union[Critic, None] = field(default=None)
    no_critic: bool = field(default=False)
    save_path: str = field(default='.')
   
    # Initialized elsewhere
    #: Policy/Actor network
    pi: Actor = field(init=False)
    #: Critic/Value network
    critic: Union[Critic, EmptyCritic] = field(default=None)  # type: ignore             
    #: Particle Filter Gated Recurrent Unit (PFGRU) for guessing the location of the radiation. This is named model for backwards compatibility reasons.
    model: PFGRUCell = field(init=False)
    #: Buffer that holds map-stacks and converts observations to maps
    maps: MapsBuffer = field(init=False)
    #: Mean Squared Error for loss for critic network
    mseLoss: nn.MSELoss = field(init=False)
    #: How much unscaling to do to reinflate agent coordinates to full representation. If the boundary is being enforced, this will be set to the grid boundaries; if not, it will set
    #:  it to the maximum possible steps an agent can take outside of a boundary. 
    scaled_offset: float = field(init=False)
    #: An adjustable resolution accuracy variable is computed to indicate the level of accuracy desired. Higher accuracy increases training time. Current environment returnes 
    #:  scaled coordinates for each agent. A resolution_accuracy value of 1 here means no unscaling, so all agents will fit within 1x1 grid. To make it less accurate but less 
    #:  memory intensive, reduce the resolution multiplier. To return to full inflation and full accuracy, change the multipier to 1. 
    resolution_accuracy: float = field(init=False)
    #: Ensures heatmap renders to not overwrite eachother when saving to a file.
    render_counter: int = field(init=False)  
    
    # Reset flag for unit testing
    reset_flag: int = field(init=False, default=0) # TODO switch out for pytest mock   

    def __post_init__(self)-> None:
        # Put agent number on save_path
        self.save_path_test = self.save_path
        self.save_path = f"{self.save_path[0]}/{self.id}_agent_{self.save_path[1]}"
        # Set resolution accuracy
        self.resolution_accuracy = calculate_resolution_accuracy(resolution_multiplier=self.resolution_multiplier, scale=self.environment_scale)
            
        # Set map dimension offset for boundaries
        if self.enforce_boundaries:
            self.scaled_offset = self.environment_scale * max(self.bounds_offset)                    
        else:
            self.scaled_offset = self.environment_scale * (max(self.bounds_offset) + (self.steps_per_episode * self.detector_step_size))           
        
        # For render
        self.render_counter = 0

        # Initialize buffers and neural networks
        self.maps = MapsBuffer(
                observation_dimension = self.observation_space,
                grid_bounds=self.grid_bounds, 
                resolution_accuracy=self.resolution_accuracy,
                offset=self.scaled_offset,
                steps_per_episode=self.steps_per_episode,
                number_of_agents = self.number_of_agents,
                resolution_multiplier = self.resolution_multiplier
            )
        
        # Set up actor and critic
        self.pi = Actor(map_dim=self.maps.map_dimensions, action_dim=self.action_space)
        if self.GlobalCritic:
            self.critic = self.GlobalCritic       
        elif self.no_critic:
            self.critic = EmptyCritic()
        else:
            self.critic = Critic(map_dim=self.maps.map_dimensions)

        self.mseLoss = nn.MSELoss()
        
        # TODO rename this (this is the PFGRU module); naming this "model" for compatibility reasons (one refactor at a time!), but the true model is the maps buffer
        # TODO Finish integrating this 
        self.model = PFGRUCell(input_size=self.observation_space - 8, obs_size=self.observation_space - 8, use_resampling=True, activation="relu")  
                        
    def set_mode(self, mode: str) -> None:
        ''' 
            Set mode for network. 
            :param mode: (string) Set agent training mode. Options include 'train' or 'eval'. 
        '''
        if mode == 'train':
            self.pi.put_in_training_mode()
            self.critic.put_in_training_mode()
            #self.model.put_in_training_mode() # TODO add
        elif mode == 'eval':
            self.pi.put_in_evaluation_mode
            self.critic.put_in_evaluation_mode
            #self.model.put_in_training_mode() # TODO add
            
        else:
            raise Warning('Invalid mode set for Agent. Agent remains in their original training mode')

    def get_map_stack(self, state_observation: Dict[int, npt.NDArray], id: int):
        with torch.no_grad():
            # TODO Maps are not matching between agents, needs check 
            (
                location_map,
                others_locations_map,
                readings_map,
                visit_counts_map,
                obstacles_map,
                combo_location_map                
            ) = self.maps.observation_to_map(state_observation, id)
            
            # Convert map to tensor
            actor_map_stack: torch.Tensor = torch.stack(
                [torch.tensor(location_map), torch.tensor(others_locations_map), torch.tensor(readings_map), torch.tensor(visit_counts_map),  torch.tensor(obstacles_map)]
            )
            
            critic_map_stack: torch.Tensor = torch.stack(
                [torch.tensor(combo_location_map), torch.tensor(readings_map), torch.tensor(visit_counts_map),  torch.tensor(obstacles_map)]
            )            
            
        # Add single batch tensor dimension for action selection
        batched_actor_mapstack: torch.Tensor = torch.unsqueeze(actor_map_stack, dim=0)      
        batched_critic_mapstack: torch.Tensor = torch.unsqueeze(critic_map_stack, dim=0)   
        
        return batched_actor_mapstack, batched_critic_mapstack
                     
    def select_action(self, state_observation: Dict[int, npt.NDArray], id: int) -> Tuple[ActionChoice, HeatMaps]:
        ''' 
            Method to take a multi-agent observation and convert it to maps and store to a buffer. Then uses the actor network to select an 
            action (and returns action logprobabilities) and the critic network to calculate state-value. 
            
            :param state_observation: (Dict[int, npt.NDArray]) Dictionary with each agent's observation. The agent id is the key.
            :param id: (int) ID of the agent who's observation is being processed. This allows any agent to recreate mapbuffers for any other agent
        '''
        #try:
        with torch.no_grad():
            batched_actor_mapstack, batched_critic_mapstack = self.get_map_stack(state_observation=state_observation, id=id)          
                
            # Get actions and values                          
            action, action_logprob  = self.pi.act(batched_actor_mapstack) # Choose action
                        
            state_value: Union[torch.Tensor, None] = self.critic.forward(batched_critic_mapstack)  # size(1)
            
        # except Exception as err:
        #     ''' If exception, save current model, dump the local variables to a file, and print exception'''
        #     print("Exception encountered, saving model...")
        #     if path.exists(self.save_path):
        #         self.save(checkpoint_path=self.save_path)
        #     else:
        #         self.save(checkpoint_path='./rad_team_error_log.log')
                        
        #     print(repr(err))

        state_value_item: Union[float, None]
        if state_value:
            state_value_item = state_value.item()            
        else:
            state_value_item = None
        return ActionChoice(id=id, action=action.item(), action_logprob=action_logprob.item(), state_value=state_value_item), HeatMaps(batched_actor_mapstack, batched_critic_mapstack)

    def get_map_dimensions(self)-> Tuple[int, int]:
        return self.maps.map_dimensions
    
    def get_critic_map_count(self)-> int:
        ''' total maps. NOTE: Actor takes one less than this and critic takes two less. '''
        return self.critic.map_count
    
    def get_batch_size(self)-> int:
        return self.pi.batches

    def save(self, checkpoint_path: str)-> None:
        '''
            Save the actor, critic, and predictor neural network models.
            
            :param checkpoint_path: (str) Path to save neural network models to.
        '''
        
        # Save original modes
        pi_train_mode: bool = self.pi.training
        critic_train_mode: bool = self.critic.training
        predictor_train_mode: bool = self.model.training 
        self.set_mode(mode='eval')
        
        self.pi.save_model(checkpoint_path=checkpoint_path)
        self.critic.save_model(checkpoint_path=checkpoint_path)
        self.model.save_model(checkpoint_path=checkpoint_path)
        
        # Restore original modes
        if pi_train_mode: self.pi.train() 
        else: self.pi.eval()
        if critic_train_mode: self.critic.train() 
        else: self.critic.eval()
        if predictor_train_mode: self.model.train() 
        else: self.model.eval()           
   
    def load(self, checkpoint_path)-> None:
        '''
            Load a saved actor, critic, and predictor neural network model.
            
            :param checkpoint_path: (str) Path to read neural network models from.
        '''
        self.pi.load_model(checkpoint_path=checkpoint_path)
        self.critic.load_model(checkpoint_path=checkpoint_path)
        self.model.load_model(checkpoint_path=checkpoint_path)  
            
    def reset(self)-> None:
        ''' Reset entire maps buffer '''
        self.maps.reset()
        self.reset_flag += 1 if self.reset_flag < 100 else 1

    def step(self, state_observation: Dict[int, npt.NDArray], hidden: Union[torch.Tensor, None] = None) -> Tuple[ActionChoice, HeatMaps]:
        ''' Alias for select_action '''
        return self.select_action(state_observation=state_observation, id=self.id)

    def reset_hidden(self)-> None:
        ''' For compatibility - returns nothing '''
        return
    def render(self, savepath: str=getcwd(), save_map: bool=True, add_value_text: bool=False, interpolation_method: str='nearest', epoch_count: int=0, episode_count: int=0)-> None:
        ''' 
            Renders heatmaps from maps buffer
            
            :param savepath: (str) Path to save heatmaps to.
            :param save_map: (bool) Whether to save or immediately render heatmaps.
            :param add_value_text: (bool) Whether to add font with values to heatmap.
            :param interpolation_method: (str) Interpolation method for "heat". Defaults to "nearest".
        '''
        if save_map:
            if not path.isdir(str(savepath) + "/heatmaps/"):
                mkdir(str(savepath) + "/heatmaps/")
        else:
            plt.show()
     
        loc_transposed: npt.NDArray = self.maps.location_map.T
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
        
        fig.savefig(f'{str(savepath)}/heatmaps/heatmap_agent{self.id}_epoch_{epoch_count}-{episode_count}({self.render_counter}).png', format='png')
        fig.savefig(f'{str(savepath)}/heatmaps/heatmap_agent{self.id}_epoch_{epoch_count}-{episode_count}({self.render_counter}).eps', format='eps')
        
        self.render_counter += 1
        plt.close(fig)