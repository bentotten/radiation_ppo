'''
Evaluate agents and update neural networks using simulation environment.
'''

import os
import sys
import time
import random
from datetime import datetime
import math

import torch
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import numpy.random as npr
import numpy.typing as npt

from typing import Any, List, Literal, NewType, Optional, TypedDict, cast, get_args, Dict, NamedTuple, Type, Union, Tuple
from typing_extensions import TypeAlias
from dataclasses import dataclass, field

import json
import joblib # type: ignore
import ray

# Simulation Environment
import gym  # type: ignore
from gym_rad_search.envs import rad_search_env # type: ignore
from gym_rad_search.envs.rad_search_env import RadSearch, StepResult  # type: ignore
from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore

# PPO and logger
try:
    from ppo import OptimizationStorage, PPOBuffer, AgentPPO  # type: ignore
    from ppo import BpArgs  # type: ignore

except ModuleNotFoundError:
    from algos.multiagent.ppo import AgentPPO  # type: ignore
    from algos.multiagent.ppo import BpArgs  # type: ignore    
    
except: 
    raise Exception

# Neural Networks
try:
    import NeuralNetworkCores.FF_core as RADFF_core # type: ignore
    import NeuralNetworkCores.RADTEAM_core as RADCNN_core # type: ignore
    import NeuralNetworkCores.RADA2C_core as RADA2C_core # type: ignore
    from NeuralNetworkCores.RADTEAM_core import StatisticStandardization # type: ignore
except ModuleNotFoundError:
    import algos.multiagent.NeuralNetworkCores.FF_core as RADFF_core # type: ignore
    import algos.multiagent.NeuralNetworkCores.RADTEAM_core as RADCNN_core # type: ignore
    import algos.multiagent.NeuralNetworkCores.RADA2C_core as RADA2C_core # type: ignore
    from algos.multiagent.NeuralNetworkCores.RADTEAM_core import StatisticStandardization # type: ignore


# TODO Delete me after working
@ray.remote
def sampling_task(num_samples: int, task_id: int,
                  progress_actor: ray.actor.ActorHandle) -> int:
    num_inside = 0
    for i in range(num_samples):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        if math.hypot(x, y) <= 1:
            num_inside += 1

        # Report progress every 1 million samples.
        if (i + 1) % 1_000_000 == 0:
            # This is async.
            progress_actor.report_progress.remote(task_id, i + 1)

    # Report the final progress.
    progress_actor.report_progress.remote(task_id, num_samples)
    return num_inside

# TODO Delete me after working
@ray.remote
class ProgressActor:
    def __init__(self, total_num_samples: int):
        self.total_num_samples = total_num_samples
        self.num_samples_completed_per_task: dict = {}

    def report_progress(self, task_id: int, num_samples_completed: int) -> None:
        self.num_samples_completed_per_task[task_id] = num_samples_completed

    def get_progress(self) -> float:
        return (
            sum(self.num_samples_completed_per_task.values()) / self.total_num_samples
        )

# Uncomment when ready to run with Ray
#@ray.remote 
@dataclass
class EpisodeRunner:
    '''
        Remote function to execute requested number of episodes for requested number of monte carlo runs each episode.
        
        Process from RAD-A2C:
        - 100 episodes classes:
            - [done] create environment
            - [done] refresh environment with test env
            - create agent
            - Get initial environment observation
            - Do monte-carlo runs
                - Get action
                - Take step in env
                - Save return and increment steps-in-episode
                - If terminal or timeout: 
                    - Save det_sto from environment (why?)
                    - If first monte-carlo:
                        - If terminal, save intensity/background intenity into "done" list
                        - If not terminal, save intensity/background intenity into "not done" list
                    - If Terminal, increment done counter, add episode length to "done" list, and add episode return to "done" list
                    - If not Terminal, add episode length to "not done" list, and add episode return to "not done" list
                    - Render if desired
                    - Refresh environment and reset episode tracking
                    - ? #Reset model in action selection fcn. get_action(0)
                    - ? #Get initial location prediction
            - Render
            - Save stats/results and return:
                mc_stats['dEpLen'] = d_ep_len
                mc_stats['ndEpLen'] = nd_ep_len
                mc_stats['dEpRet'] = d_ep_ret
                mc_stats['ndEpRet'] = nd_ep_ret
                mc_stats['dIntDist'] = done_dist_int
                mc_stats['ndIntDist'] = not_done_dist_int
                mc_stats['dBkgDist'] = done_dist_bkg
                mc_stats['ndBkgDist'] = not_done_dist_bkg
                mc_stats['DoneCount'] = np.array([done_count])
                mc_stats['TotEpLen'] = tot_ep_len
                mc_stats['LocEstErr'] = loc_est_err
                results = [loc_est_ls, FIM_bound, J_score_ls, det_ls]
                print(f'Finished episode {n}!, completed count: {done_count}')
                return (results,mc_stats)            
        
    ''' 
    id: int
    env_name: str
    env_kwargs: Dict
    env_sets: Dict
    steps_per_episode: int
    
    resolution_multiplier: float
    
    model_path: str
    test_env_path: str = field(default='./evaluation/test_environments')
    save_path: str = field(default='.')
    save_gif: bool = field(default=True)
    save_gif_freq: int = field(default=100)      
    seed: Union[int, None] = field(default=9389090)
    
    obstruction_count: int = field(default=0) 
    enforce_boundaries: bool = field(default=False)
    actor_critic_architecture: str = field(default="cnn")    
    number_of_agents: int = field(default=1)
    episodes: int = field(default=100)
    montecarlo_runs: int = field(default=100)
    snr: str = field(default='high')
  
    
    # Initialized elsewhere
    #: Object that holds agents    
    agents: Dict[int, RADCNN_core.CNNBase] = field(default_factory=lambda:dict())
    
    
    def __post_init__(self)-> None:
        # Create own instatiation of environment
        self.env = self.create_environment()
        
        # Get agent model paths and general agent parameters
        agent_models = {}
        for child in os.scandir(self.model_path):
            if child.is_dir() and 'agent' in child.name:
                agent_models[int(child.name[0])] = child.path  # Read in model path by id number. NOTE: Important that ID number is the first element of file name 
            if child.is_dir() and 'general' in child.name:
                general_config_path = child.path  
        original_configs = list(json.load(open(f"{general_config_path}/config.json"))['self'].values())[0]['ppo_kwargs']['actor_critic_args']
        
        # Setup Agent arguments
        # Bootstrap particle filter args for the PFGRU, from Particle Filter Recurrent Neural Networks by Ma et al. 2020.
        bp_args = BpArgs(
            bp_decay=0.1,
            l2_weight=1.0,
            l1_weight=0.0,
            elbo_weight=1.0,
            area_scale=self.env.search_area[2][1]
        )
        # Set up static A2C args.      
        actor_critic_args=dict(
            action_space=self.env.detectable_directions,
            observation_space=self.env.observation_space.shape[0], # Also known as state dimensions: The dimensions of the observation returned from the environment
            steps_per_episode=self.steps_per_episode,
            number_of_agents=self.number_of_agents,
            detector_step_size=self.env.step_size,
            environment_scale=self.env.scale,
            bounds_offset=self.env.observation_area,
            enforce_boundaries=self.enforce_boundaries,
            grid_bounds=self.env.scaled_grid_max,
            resolution_multiplier=self.resolution_multiplier,
            GlobalCritic=None,
            no_critic=True
        )
        
        for arg in actor_critic_args:
            if arg != 'no_critic' and arg != 'GlobalCritic':
                print(arg)
                print(type(original_configs[arg]))
                if type(original_configs[arg]) == int or type(original_configs[arg]) == float or type(original_configs[arg]) == bool:
                    assert actor_critic_args[arg] == original_configs[arg], f"CNN Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                elif type(original_configs[arg]) is str:
                    to_list = original_configs[arg].strip('][').split(' ')
                    config = np.array([float(x) for x in to_list], dtype=np.float32)
                    assert np.array_equal(config, actor_critic_args[arg]), f"CNN Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                elif type(original_configs[arg]) is list:
                    for a,b in zip(original_configs[arg], actor_critic_args[arg]):
                        assert a == b, f"CNN Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                else:
                    assert actor_critic_args[arg] == original_configs[arg], f"CNN Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"             
        
        # Initialize agents and load agent models
        for i in range(self.number_of_agents):
            self.agents[i] = RADCNN_core.CNNBase(id=i, **actor_critic_args)  # NOTE: No updates, do not need PPO
            # try: 
            self.agents[i].load(checkpoint_path=agent_models[i])
            # except:
            #     raise Exception("Model does not exist. Be sure to doublecheck path and number of agents are correct")
            
            # Sanity check
            assert self.agents[i].agent.critic.is_mock_critic()
        
        self.run()
    
    def run(self)-> None:
        # Refresh environment with test env parameters
        self.env.refresh_environment(env_dict=self.env_sets, id=0, num_obs=self.obstruction_count)
        self.env.render(path='.', just_env=True)
        
        # - create agent
        # - Get initial environment observation
        pass
    
    def create_environment(self) -> RadSearch:
        return gym.make(self.env_name, **self.env_kwargs) 
    
    


@dataclass
class evaluate_PPO:
    '''
        Test existing model across random episodes for a set number of monte carlo runs per episode.

    '''

    eval_kwargs: Dict
    
    # Initialized elsewhere
    #: Directory containing test environments. Each test environment file contains 1000 environment configurations.
    test_env_dir: str = field(init=False)
    #: Full path to file containing chosen test environment. Each test environment file contains 1000 environment configurations.
    test_env_path: str = field(init=False)    
    #: Sets of environments for specifications. Comes in sets of 1000.
    environment_sets: Dict = field(init=False)

    def __post_init__(self)-> None:
        self.test_env_dir = self.eval_kwargs['test_env_path']
        self.test_env_path = self.test_env_dir + f"/test_env_dict_obs{self.eval_kwargs['obstruction_count']}_{self.eval_kwargs['snr']}_v4"
                
        # Load test environments
        self.environment_sets = joblib.load(self.test_env_path)
        
        # Uncomment when ready to run with Ray                
        # Initialize ray
        # try:
        #     ray.init(address='auto')
        # except:
        #     print("Ray failed to initialize. Running on single server.")

    def evaluate(self):
        ''' Driver '''       
        # Uncomment when ready to run with Ray        
        # runners = {i: EpisodeRunner.remote(
        #         id=i, 
        #         env_name=self.env_name, 
        #         env_kwargs=self.env_kwargs, 
        #         env_sets=self.environment_sets, 
        #         number_of_obstructions=self.obstruction_count
        #     ) for i in range(self.episodes)} 
        
        EpisodeRunner(id=0, env_sets=self.environment_sets, **self.eval_kwargs)
        
    def _test_remote(self):
        # https://docs.ray.io/en/latest/ray-core/examples/monte_carlo_pi.html
        
        # Change this to match your cluster scale.
        NUM_SAMPLING_TASKS = 10
        NUM_SAMPLES_PER_TASK = 10_000_000
        TOTAL_NUM_SAMPLES = NUM_SAMPLING_TASKS * NUM_SAMPLES_PER_TASK

        # Create the progress actor.
        progress_actor = ProgressActor.remote(TOTAL_NUM_SAMPLES)
        
        # Create and execute all sampling tasks in parallel.
        results = [
            sampling_task.remote(NUM_SAMPLES_PER_TASK, i, progress_actor)
            for i in range(NUM_SAMPLING_TASKS)
        ]        
            
        # Query progress periodically.
        while True: 
            progress = ray.get(progress_actor.get_progress.remote())
            print(f"Progress: {int(progress * 100)}%")

            if progress == 1:
                break

            time.sleep(1)            
        
        # Get all the sampling tasks results.
        total_num_inside = sum(ray.get(results))
        pi = (total_num_inside * 4) / TOTAL_NUM_SAMPLES
        print(f"Estimated value of π is: {pi}")
        pass