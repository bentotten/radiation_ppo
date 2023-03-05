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
    from epoch_logger import EpochLogger, EpochLoggerKwargs, setup_logger_kwargs, convert_json  # type: ignore
except ModuleNotFoundError:
    from algos.multiagent.ppo import OptimizationStorage, PPOBuffer, AgentPPO  # type: ignore
    from algos.multiagent.epoch_logger import EpochLogger, EpochLoggerKwargs, setup_logger_kwargs, convert_json
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
            - refresh environment with test env
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
        
        :param env_name: (str) Name of environment to be loaded with GymAI.
        :param env_kwargs: (Dict) Arguments to create Rad-Search environment.
        :param env_sets: (Dict) Dictionary of test environments
        
    ''' 
    id: int
    env_name: str
    env_kwargs: Dict
    env_sets: Dict
    number_of_obstructions: int
    
    def __post_init__(self)-> None:
        # Create own instatiation of environment
        self.env = self.create_environment()
        
        self.run()
    
    def run(self)-> None:
        # Refresh environment with test env parameters
        self.env.refresh_environment(env_dict=self.env_sets, id=0, num_obs=self.number_of_obstructions)
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
        
        :param env_name: (str) Name of environment to be loaded with GymAI.
        :param env_kwargs: (Dict) Arguments to create Rad-Search environment. Needs to be the arguments so multiple environments can be used in parallel.
        :param ac_kwargs: (dict) Arguments for A2C neural networks for agent.
        
        :param model_path: (str) Directory containing trained models.
        :param test_env_path: (str) Directory containing test environments. Each test environment file contains 1000 environments
        :param save_path: (str) Directory to save results to. Defaults to '.'
        :param seed: (Union[int, None]) Random seed control for reproducability. Defaults to 9389090.
        
        :param number_of_agents: (int) Number of agents. Defaults to 1.        
        
        :param episodes: (int) Number of episodes to test on [1 - 1000]. Defaults to 100.
        :param montecarlo_runs: (int) Number of Monte Carlo runs per episode (How many times to run/sample each episode setup). Defaults to 100. 
        :param snr: (str) Signal to noise ratio [None, low, medium, high] of background radiation and gamma radiation in environment. Defaults to high.
        :param obstruction_count: (int) Number of obstructions in the environment [0 - 7]. Defaults to zero.
    
        :param actor_critic_architecture: (string) Short-version indication for what neural network core to use for actor-critic agent
        
        :param save_gif: (bool) Save gif of episodes or not. Defaults to True.
        :param save_gif_freq: (int) How many epsiodes should be saved (including monte-carlo repeats). Defaults to 100.
    
    '''
    env_name: str
    
    env_kwargs: Dict
    
    model_path: str
    test_env_path: str = field(default='./evaluation/test_environments')
    save_path: str = field(default='.')
    seed: Union[int, None] = field(default=9389090)
    number_of_agents: int = field(default=1)
    episodes: int = field(default=100)
    montecarlo_runs: int = field(default=100)
    snr: str = field(default='high')
    obstruction_count: int = field(default=0) 
    
    actor_critic_architecture: str = field(default='cnn')
    
    save_gif: bool = field(default=True)
    save_gif_freq: int = field(default=100)
    
    # Initialized elsewhere
    episode_counter_buffer: npt.NDArray = field(init=False) # TODO is this necessary with ray?
    #: Sets of environments for specifications. Comes in sets of 1000.
    environment_sets: Dict = field(init=False)

    def __post_init__(self)-> None:
        
        self.episode_counter = np.arange(start=0, stop=100, step=1) # TODO is this necessary with ray?
        
        # Load test environments
        self.environment_sets = joblib.load(self.test_env_path + f"/test_env_dict_obs{self.obstruction_count}_{self.snr}_v4")
        
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
        
        EpisodeRunner(
                id=0, 
                env_name=self.env_name, 
                env_kwargs=self.env_kwargs, 
                env_sets=self.environment_sets, 
                number_of_obstructions=self.obstruction_count
            )
        
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