from dataclasses import dataclass, field, asdict
from typing import Any, List, Tuple, Union, Literal, NewType, Optional, TypedDict, cast, get_args, Dict, Callable, overload, NamedTuple
from typing_extensions import TypeAlias
from gym_rad_search.envs import StepResult # type: ignore
from numpy import dtype
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical
from torchinfo import summary

Action: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]  # 8 is idle step
ACTION_MAPPING: dict = {"left": 0,"up left": 1, "up": 2, "up right": 3, "right": 4,"down right": 5,"down": 6, "down left": 7, "idle": 8}
STATES: dict = {
    "search": 0, 
    "wall left": 1, 
    "wall right": 2, 
    "corner upper left": 3, 
    "corner upper right": 4, 
    "corner lower left": 5, 
    "corner lower right": 6
    }

@dataclass()
class ActionChoice():
    id: int 
    action: npt.NDArray # size (1)
    action_logprob: npt.NDArray # size (1)
    state_value: npt.NDArray # size(1)

    # For compatibility with RAD-PPO
    hiddens: Union[torch.Tensor, None] = field(default=None)
    loc_pred: Union[npt.NDArray, None] = field(default=None)
    

class NestedFiller(NamedTuple):
    shape: Tuple[int, int]


class Filler():
    ''' filler for compatability '''
    
    def eval(self):
        print("eval() filler")
    
    def parameters(self):
        return NestedFiller()

    def state_dict(self):
        return {0: None}

    
@dataclass
class Core:
    id: int    
    state_dim: int
    action_dim: int
    grid_bounds: Tuple[float, float]
    resolution_accuracy: float
    steps_per_epoch: int
    scaled_offset: float = field(default=0)    
    random_seed: Union[None, int] = field(default=None)
    critic: Any = field(default=None)  # Eventually allows for a global critic
    render_counter: int = field(init=False)
    '''
    Skeleton for testing purposes. A non-learning uniform search with many filler elements for compatability.
    '''
    def __post_init__(self)-> None:
        # Scaled maps
        self.map_dimensions = (
            int(self.grid_bounds[0] * self.resolution_accuracy) + int(self.scaled_offset  * self.resolution_accuracy),
            int(self.grid_bounds[1] * self.resolution_accuracy) + int(self.scaled_offset  * self.resolution_accuracy)
        )
        self.x_limit_scaled: int = self.map_dimensions[0]
        self.y_limit_scaled: int = self.map_dimensions[1]
        
        # Set initial travel direction
        self.search_direction = ACTION_MAPPING['left']
        self.search_up = True # Inidicates if should search upwards or downwards
        self.state = STATES['search']  # indicates what mode agent is in. Used for corner and wall navigation
        
        # Filler for compatibility
        self.model = Filler()
        self.pi = Filler()
        self.critic = Filler()
                
    def select_action(self, observation: Dict[int, list], message: Dict[int, Any], id: int, save_map=True) -> ActionChoice:         
        # Inflate current coordinates
        scaled_coordinates = (int(observation[id][1] * self.resolution_accuracy), int(observation[id][2] * self.resolution_accuracy))
        action = self.search_direction
        
        # Handle wall
        if message[id]['out_of_bounds']:
            action = self.handle_boundary(scaled_coordinates)
        
        # TODO Finish U-Turns and reset state marker
        
        return action
            
            
    def handle_boundary(self, scaled_coordinates: tuple):
        ''' Negotiates walls and corners '''
        
        # def handle_wall(self, wall: str):
        #     match wall: # type: ignore
        #         case 'left':
        #             self.state = STATES['wall left']
        #             return ACTION_MAPPING['up'] if self.search_up else ACTION_MAPPING['down']

        #         case 'right':
        #             self.state = STATES['wall right']
        #             return ACTION_MAPPING['up'] if self.search_up else ACTION_MAPPING['down']
                    
        #         case 'up':
        #             self.state = STATES['wall up']
        #             # TODO handle corner
                    
        #         case 'down':
        #             self.state = STATES['wall down']
        #             # TODO handle corner
                    
        #         case _:
        #             raise ValueError(f"Invalid wall direction: {wall}")
        
        def handle_corner(Self):
            pass
        
        # if self.state == STATES['search']:
        #     ''' Find which boundary was violated '''
        #     if scaled_coordinates[0] == 0:
        #         action: int = handle_wall(self, 'left')
        #     elif scaled_coordinates[0] == self.x_limit_scaled:
        #         action: int = handle_wall(self, 'right')
        #     elif scaled_coordinates[1] == 0:
        #         action: int = handle_wall(self, 'down')
        #     elif scaled_coordinates[1] == self.y_limit_scaled:
        #         action: int = handle_wall(self, 'up')
        # else:
        #     # TODO Handle corner
        #     pass   
        action = None
        return action
    
    def update(self):
        print("Filler for update()")     
        return 0, 0            
         
    def save(self, checkpoint_path):
        print("Filler for save()")     
   
    def load(self, checkpoint_path):
        print("Filler for load()")     
        
    def render(self, savepath: Union[None, str] = None, save_map: bool=True, add_value_text: bool=False, interpolation_method: str='nearest', epoch_count: int=0):
        ''' Renders heatmaps from maps buffer '''
        print("Filler for render()")     
           
    def reset(self):
        ''' Reset entire CNN '''
        print("Filler for reset()")     
        
    def clear_maps(self):
        ''' Just clear maps and buffer for new episode'''
        print("Filler for clear_maps()")     

