'''
Evaluate agents and update neural networks using simulation environment.
'''

import os
import sys
import glob
import time
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
except: 
    raise Exception

@dataclass
class evaluate_PPO:
    '''
        Test existing model across random episodes for a set number of monte carlo runs per episode.
        
        :param env_name: (str) Name of environment to be loaded with GymAI.
        :param env_kwargs: (Dict) Arguments to create Rad-Search environment. Needs to be the arguments so multiple environments can be used in parallel.
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
    episode_counter_buffer: npt.NDArray = field(init=False)

    def __post_init__(self)-> None:
        
        self.episode_counter = np.arange(start=0, stop=100, step=1) # TODO is this necessary with ray?
        
        # Load test environments
        full_test_env_path = self.test_env_path + f"/test_env_dict_obs{self.obstruction_count}_{self.snr}_v4"
        environment_sets = joblib.load(full_test_env_path)
        
        self.refresh_environment(environment_sets=environment_sets, counter_number=0)
        pass

    def create_environment(self) -> RadSearch:
        return gym.make(self.env_name, **self.env_kwargs)
        

    def refresh_environment(self, environment_sets: Dict, counter_number: int) -> None:
        """
            Load saved test environment parameters from dictionary into the current instantiation of environment
        """
        
        # TODO switch out when done
        env_dict = environment_sets
        num_obs = self.obstruction_count
        ###############################
        
        key = 'env_'+str(n)
        env.src_coords    = env_dict[key][0]
        env.det_coords    = env_dict[key][1].copy()
        env.intensity     = env_dict[key][2]
        env.bkg_intensity = env_dict[key][3]
        env.source        = set_vis_coord(env.source,env.src_coords)
        env.detector      = set_vis_coord(env.detector,env.det_coords) # TODO Make compatible with multi-agent env
        
        if num_obs > 0:
            env.obs_coord = env_dict[key][4]
            env.num_obs = len(env_dict[key][4])
            env.poly = []
            env.line_segs = []
            for obs in env.obs_coord:
                geom = [vis.Point(float(obs[0][jj][0]),float(obs[0][jj][1])) for jj in range(len(obs[0]))]
                poly = vis.Polygon(geom)
                env.poly.append(poly)
                env.line_segs.append([vis.Line_Segment(geom[0],geom[1]),vis.Line_Segment(geom[0],geom[3]),
                vis.Line_Segment(geom[2],geom[1]),vis.Line_Segment(geom[2],geom[3])]) 
            
            env.env_ls = [solid for solid in env.poly]
            env.env_ls.insert(0,env.walls)
            env.world = vis.Environment(env.env_ls)
            # Check if the environment is valid
            assert env.world.is_valid(EPSILON), "Environment is not valid"
            env.vis_graph = vis.Visibility_Graph(env.world, EPSILON)

        o, _, _, _        = env.step(-1)
        env.det_sto       = [env_dict[key][1].copy()]  # TODO Make compatible with multi-agent env
        env.src_sto       = [env_dict[key][0].copy()]  # TODO Make compatible with multi-agent env
        env.meas_sto      = [o[0].copy()]  # TODO Make compatible with multi-agent env
        env.prev_det_dist = env.world.shortest_path(env.source,env.detector,env.vis_graph,EPSILON).length() # TODO Make compatible with multi-agent env
        env.iter_count    = 1
        return o, env

    # robust_seed = _int_list_from_bigint(hash_seed(seed))[0]
    # rng = np.random.default_rng(robust_seed)
    # params = np.arange(0,args.episodes,1)

    # #Model parameters, must match the model being loaded # TODO load dynamically from saved configs instead
    # ac_kwargs = {'batch_s': 1, 'hidden': [24], 'hidden_sizes_pol': [32], 'hidden_sizes_rec': [24], 
    #              'hidden_sizes_val': [32], 'net_type': 'rnn', 'pad_dim': 2, 'seed': robust_seed}

    # # #Bootstrap particle filter parameters for RID-FIM controller and BPF-A2C
    # # bp_kwargs = {'nParticles':int(6e3), 'noise_params':[15.,1.,1],'thresh':1,'s_size':3,
    # #             'rng':rng, 'L': 1,'k':0.0, 'alpha':0.6, 'fim_thresh':0.36,'interval':[75,75]}
    
    # # #Gradient search parameters
    # # grad_kwargs = {'q':0.0042,'env':env}

    # #Create partial func. for use with multiprocessing
    # # TODO use Ray instead
    # func = partial(run_policy, env, env_set, args.render,args.save_gif, 
    #                args.fpath, args.mc_runs, args.control,args.fisher,
    #                ac_kwargs,bp_kwargs,grad_kwargs, args.episodes)
    # mc_results = []
    # print(f'Number of cpus available: {os.cpu_count()}')
    # print('Starting pool')
    # p = Pool(processes=args.num_cpu)
    # mc_results.append(p.map(func,params)) 
    # stats, len_freq = calc_stats(mc_results,mc=args.mc_runs,plot=False,snr=args.snr,control=args.control,obs=args.num_obs)
    
    # if args.save_results:
    #     print('Saving results..')
    #     joblib.dump(stats,'results/raw/n_'+str(args.episodes)+'_mc'+str(args.mc_runs)+'_'+args.control+'_'+'stats_'+args.snr+'_v4.pkl')
    #     joblib.dump(len_freq,'results/raw/n_'+str(args.episodes)+'_mc'+str(args.mc_runs)+'_'+args.control+'_'+'freq_stats_'+args.snr+'_v4.pkl')
    #     joblib.dump(mc_results,'results/raw/n_'+str(args.episodes)+'_mc'+str(args.mc_runs)+'_'+args.control+'_'+'full_dump_'+args.snr+'_v4.pkl')
      