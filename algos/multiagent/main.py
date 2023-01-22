import argparse
from dataclasses import dataclass
from typing import Literal
from datetime import datetime

import numpy as np
import numpy.random as npr

import gym
from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore

import core
from epoch_logger import setup_logger_kwargs, EpochLogger
import train
from gym_rad_search.envs import RadSearch  # type: ignore

''' Parameters passed in through the command line '''
@dataclass
class CliArgs:
    hid_gru: int
    hid_pol: int
    hid_val: int
    hid_rec: int
    l_pol: int
    l_val: int
    gamma: float
    seed: int
    steps_per_epoch: int
    epochs: int
    exp_name: str
    dims: tuple[int, int]
    area_obs: tuple[int, int]
    obstruct: Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7]
    net_type: str
    alpha: float
    render: bool
    agent_count: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    enforce_grid_boundaries: bool
    minibatches: int
    env_name: str
    save_freq: int
    save_gif_freq: int

''' Function to parge command line arguments '''
def parse_args(parser: argparse.ArgumentParser) -> CliArgs:
    args = parser.parse_args()
    return CliArgs(
        hid_gru=args.hid_gru,
        hid_pol=args.hid_pol,
        hid_val=args.hid_val,
        hid_rec=args.hid_rec,
        l_pol=args.l_pol,
        l_val=args.l_val,
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        exp_name=args.exp_name,
        dims=args.dims,
        area_obs=args.area_obs,
        obstruct=args.obstruct,
        net_type=args.net_type,
        alpha=args.alpha,
        render=args.render,
        agent_count=args.agent_count,
        enforce_grid_boundaries=args.enforce_grid_boundaries,
        minibatches=args.minibatches,
        env_name=args.env_name,
        save_freq=args.save_freq,
        save_gif_freq=args.save_gif_freq
    )

''' Function to generate argument parser '''
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    
    # General parameters
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=480,
        help="Number of timesteps per epoch (before updating agent networks)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3000, help="Number of total epochs to train the agent"
    )
    parser.add_argument("--seed", type=int, default=2, help="Random seed control")
    parser.add_argument(
        "--exp-name",
        type=str,
        default="test",
        help="Name of experiment for saving",
    )
    parser.add_argument(
        "--agent_count", type=int, # Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        default=1, 
        help="Number of agents"
    )   
    parser.add_argument(
        "--render", type=bool, default=False, help="Save gif"
    )          
    parser.add_argument(
        "--save_gif_freq", type=float, default=float('inf'), help="If render is true, save gif after this many epochs."
    )     
    parser.add_argument(
        "--save_freq", type=int, default=500, help="How often to save the model."
    )        
    
    # Environment Parameters
    parser.add_argument('--env-name', type=str, default='gym_rad_search:RadSearchMulti-v1', help="Environment name registered with Gym")
    parser.add_argument(
        "--dims",
        type=float,
        nargs=2,
        default=[2700.0, 2700.0],
        metavar=("dim_length", "dim_height"),
        help="Dimensions of radiation source search area in cm, decreased by area_obs param. to ensure visilibity graph setup is valid. Length by height.",
    )
    parser.add_argument(
        "--area-obs",
        type=float,
        nargs=2,
        default=[200.0, 500.0],
        metavar=("area_obs_min", "area_obs_max"),
        help="Interval for each obstruction area in cm. This is how much to remove from bounds to make the 'visible bounds'",
    )
    parser.add_argument(
        "--obstruct",
        type= int, #Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7],
        default=-1,
        help="Number of obstructions present in each episode, options: -1 -> random sampling from [1,5], 0 -> no obstructions, [1-7] -> 1 to 7 obstructions",
    )  
    parser.add_argument(
        "--enforce_grid_boundaries", type=bool, default=False, help="Indicate whether or not agents can travel outside of the search area"
    )   
              
    # Hyperparameters and PPO parameters
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Reward attribution for advantage estimator for PPO updates",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Entropy reward term scaling"
    )
    parser.add_argument(
        "--minibatches", type=int, default=1, help="Batches to sample data during actor policy update (k_epochs)"
    )    
    
    # Parameters for Neural Networks
    parser.add_argument(
        "--net-type",
        type=str,
        default="rnn",
        help="Choose between recurrent neural network or MLP Actor-Critic (A2C), option: rnn, mlp",
    )    
    parser.add_argument(
        "--hid-pol", type=int, default=32, help="Actor linear layer size (Policy Hidden Layer Size)"
    )
    parser.add_argument(
        "--hid-val", type=int, default=32, help="Critic linear layer size (State-Value Hidden Layer Size)"
    )
    parser.add_argument(
        "--hid-rec", type=int, default=24, help="PFGRU hidden state size (Localization Network)"
    )
    parser.add_argument(
        "--hid-gru", type=int, default=24, help="Actor-Critic GRU hidden state size (Embedding Layers)"
    )    
    parser.add_argument(
        "--l-pol", type=int, default=1, help="Number of layers for Actor MLP (Policy Multi-layer Perceptron)"
    )
    parser.add_argument(
        "--l-val", type=int, default=1, help="Number of layers for Critic MLP (State-Value Multi-layer Perceptron)"
    )
    
    return parser


if __name__ == "__main__":
    args = parse_args(create_parser())

    # Save directory and experiment name
    save_dir_name: str = args.exp_name  # Stands for bootstrap particle filter, one of the neat resampling methods used
    exp_name: str = (
        args.exp_name        
        + "_"
        "agents"
        + str(args.agent_count)
        # + "_loc"
        # + str(args.hid_rec)
        # + "_hid"
        # + str(args.hid_gru)
        # + "_pol"
        # + str(args.hid_pol)
        # + "_val"
        # + str(args.hid_val)
        # + f"_epochs{args.epochs}"
        # + f"_steps{args.steps_per_epoch}"
    )

    # Generate a large random seed and random generator object for reproducibility
    robust_seed = _int_list_from_bigint(hash_seed(args.seed))[0]
    exp_name = datetime.now().replace(microsecond=0).strftime('%Y-%m-%d-%H:%M:%S') + "_" + exp_name
    rng = npr.default_rng(robust_seed)

    # Set up logger args 
    logger_kwargs = {'exp_name': exp_name, 'seed': args.seed, 'data_dir': "../../models/train", 'env_name': save_dir_name}   

    # Set up Radiation environment
    dim_length, dim_height = args.dims
    intial_parameters = {'bbox': np.array(  # type: ignore
            [[0.0, 0.0], [dim_length, 0.0], [dim_length, dim_height], [0.0, dim_height]]
        ),
        'observation_area': np.array(args.area_obs),  # type: ignore
        'obstruction_count': args.obstruct,
        'np_random': rng,
        'number_agents': args.agent_count,
        'save_gif': args.render,
    }

    env: RadSearch = gym.make(args.env_name,**intial_parameters)
    
    # Uncommenting this will make the environment without Gym's oversight (useful for debugging)
    # env: RadSearch = RadSearch(
    #     bbox=np.array(  # type: ignore
    #         [[0.0, 0.0], [dim_length, 0.0], [dim_length, dim_height], [0.0, dim_height]]
    #     ),
    #     observation_area=np.array(args.area_obs),  # type: ignore
    #     obstruction_count=args.obstruct,
    #     np_random=rng,
    #     number_agents = args.agent_count
    # )    

    # Run ppo training function
    ppo = train.PPO(
        env=env,
        actor_critic=core.RNNModelActorCritic,
        logger_kwargs=logger_kwargs,
        ac_kwargs=dict(
            hidden_sizes_pol=[[args.hid_pol]] * args.l_pol,
            hidden_sizes_val=[[args.hid_val]] * args.l_val,
            hidden_sizes_rec=[args.hid_rec],
            hidden=[[args.hid_gru]],
            net_type=args.net_type,
            batch_s=args.minibatches,
        ),
        gamma=args.gamma,
        alpha=args.alpha,
        seed=robust_seed,
        steps_per_epoch=args.steps_per_epoch,
        total_epochs=args.epochs,
        number_of_agents=args.agent_count,
        render=args.render,
        save_gif=args.render, # TODO combine into just render
        save_freq=args.save_freq,
        save_gif_freq=args.save_gif_freq
    )
    
    ppo.train()
    #ppo.train_old()
