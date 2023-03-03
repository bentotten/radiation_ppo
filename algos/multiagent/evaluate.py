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
        
        :param env: An environment satisfying the OpenAI Gym API.    
        :param model_path: (str) Directory containing trained models.
        :param save_path: (str) Directory to save results to. Defaults to '.'
        :param seed: (Union[int, None]) Random seed control for reproducability. Defaults to 9389090.
        :param episodes: (int) Number of episodes to test on [1 - 1000]. Defaults to 100.
        :param montecarlo_runs: (int) Number of Monte Carlo runs per episode (How many times to run/sample each episode setup). Defaults to 100. 
        :param snr: (str) Signal to noise ratio [None, low, medium, high] of background radiation and gamma radiation in environment. Defaults to high.
        :param obstruction_count: (int) Number of obstructions in the environment [0 - 7]. Defaults to zero.
        
        :param save_gif: (bool) Save gif of episodes or not. Defaults to True.
        :param save_gif_freq: (int) How many epsiodes should be saved (including monte-carlo repeats). Defaults to 100.
    
    '''
    env: RadSearch    
    model_path: str
    save_path: str = field(default='.')
    seed: Union[int, None] = field(default=9389090)
    episodes: int = field(default=100)
    montecarlo_runs: int = field(default=100)
    snr: str = field(default='high')
    obstruction_count: int = field(default=0) 
    save_gif: bool = field(default=True)
    save_gif_freq: int = field(default=100)

if __name__ == '__main__':
    # TODO port into current main and call evalute.py instead of train.py
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--fpath', type=str,default='../models/pre_train/rad_a2c/loc24_hid24_pol32_val32_alpha01_tkl07_val01_lam09_npart40_lr3e-4_proc10_obs-1_iter40_blr5e-3_2_tanh_ep3000_steps4800_s1',
    # help='Specify model directory, Ex: ../models/train/bpf/model_dir')
    # parser.add_argument('--episodes', '-n', type=int, default=100,help='Number of episodes to test on, option: [1-1000]') 
    # parser.add_argument('--render', '-r',type=bool, default=False,help='Produce gif of agent in environment, last episode of n episodes. Num_cpu should be 1')
    # parser.add_argument('--save_gif', type=bool,default=False, help='Save gif of the agent in model folder, render must be true')
    # #parser.add_argument('--control', type=str, default='rad-a2c',help='Control algorithm, options: [rad-a2c,bpf-a2c,gs,rid-fim]')
    # parser.add_argument('--snr', type=str, default='high',help='Signal to Noise ratio (SNR) of environment, options: [low, med, high]')
    # parser.add_argument('--num_obs', type=int, default=0,help='Number of obstructions in environment, options:[0, 1, 2, 3, 4, 5, 6, 7]')
    # parser.add_argument('--mc_runs', type=int, default=100,help='Number of Monte Carlo runs per episode (How many times to run/sample each episode setup)')
    # #parser.add_argument('--num_cpu', '-ncpu', type=int, default=10,help='Number of cpus to run episodes across')
    # #parser.add_argument('--fisher',type=bool, default=False,help='Calculate the posterior Cramer-Rao Bound for BPF based methods')
    # parser.add_argument('--save_results', type=bool, default=False, help='Save list of results across episodes and runs')
    # args = parser.parse_args()
    
    #Path for the test environments
    env_fpath = 'test_envs/snr/test_env_dict_obs'
    robust_seed = _int_list_from_bigint(hash_seed(seed))[0]
    rng = np.random.default_rng(robust_seed)
    params = np.arange(0,args.episodes,1)

    #Load set of test envs 
    env_path = env_fpath + str(args.num_obs) if args.snr is None else env_fpath + str(args.num_obs)+'_'+args.snr+'_v4'

    try:
        env_set = joblib.load(env_path)
    except:
        env_set = joblib.load(f'eval/{env_path}')

    #Model parameters, must match the model being loaded # TODO load dynamically from saved configs instead
    ac_kwargs = {'batch_s': 1, 'hidden': [24], 'hidden_sizes_pol': [32], 'hidden_sizes_rec': [24], 
                 'hidden_sizes_val': [32], 'net_type': 'rnn', 'pad_dim': 2, 'seed': robust_seed}

    # #Bootstrap particle filter parameters for RID-FIM controller and BPF-A2C
    # bp_kwargs = {'nParticles':int(6e3), 'noise_params':[15.,1.,1],'thresh':1,'s_size':3,
    #             'rng':rng, 'L': 1,'k':0.0, 'alpha':0.6, 'fim_thresh':0.36,'interval':[75,75]}
    
    # #Gradient search parameters
    # grad_kwargs = {'q':0.0042,'env':env}

    #Create partial func. for use with multiprocessing
    # TODO use Ray instead
    func = partial(run_policy, env, env_set, args.render,args.save_gif, 
                   args.fpath, args.mc_runs, args.control,args.fisher,
                   ac_kwargs,bp_kwargs,grad_kwargs, args.episodes)
    mc_results = []
    print(f'Number of cpus available: {os.cpu_count()}')
    print('Starting pool')
    p = Pool(processes=args.num_cpu)
    mc_results.append(p.map(func,params)) 
    stats, len_freq = calc_stats(mc_results,mc=args.mc_runs,plot=False,snr=args.snr,control=args.control,obs=args.num_obs)
    
    if args.save_results:
        print('Saving results..')
        joblib.dump(stats,'results/raw/n_'+str(args.episodes)+'_mc'+str(args.mc_runs)+'_'+args.control+'_'+'stats_'+args.snr+'_v4.pkl')
        joblib.dump(len_freq,'results/raw/n_'+str(args.episodes)+'_mc'+str(args.mc_runs)+'_'+args.control+'_'+'freq_stats_'+args.snr+'_v4.pkl')
        joblib.dump(mc_results,'results/raw/n_'+str(args.episodes)+'_mc'+str(args.mc_runs)+'_'+args.control+'_'+'full_dump_'+args.snr+'_v4.pkl')
      