import argparse
from dataclasses import dataclass
from typing import Literal
from datetime import datetime

import numpy as np
import numpy.random as npr

import gym
from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore
from gym_rad_search.envs import RadSearch  # type: ignore

try:
    import NeuralNetworkCores.RADA2C_core as RADA2C_core
    from epoch_logger import setup_logger_kwargs, EpochLogger
    import train as train
except:
    import algos.multiagent.NeuralNetworkCores.RADA2C_core as RADA2C_core
    from algos.multiagent.epoch_logger import setup_logger_kwargs, EpochLogger
    import algos.multiagent.train as train


@dataclass
class CliArgs:
    ''' Parameters passed in through the command line 
    General parameters:
        --DEBUG, type=bool, default=False, 
            help="Enable DEBUG mode - contains extra logging and set minimal setups"
        --steps-per-episode, type=int, default=120, 
            help="Number of timesteps per episode (before resetting the environment)"
        --steps-per-epoch, type=int, default=480,
            help="Number of timesteps per epoch (before updating agent networks)"
        --epochs, type=int, default=3000, 
            help="Number of total epochs to train the agent"
        --seed, type=int, default=2, 
            help="Random seed control" 
        --exp-name, type=str, default="test", 
            help="Name of experiment for saving"
        --agent-count, type=int, # Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], default=1, 
            help="Number of agents"
        --render, type=bool, default=False, 
            help="Save gif"
        --save-gif-freq, type=int, default=float('inf'),
            help="If render is true, save gif after this many epochs."
        --save-freq, type=int, default=500, 
            help="How often to save the model."
             
    Environment Parameters:
        --env-name, type=str, default='gym_rad_search:RadSearchMulti-v1', 
            help="Environment name registered with Gym" 
        --dims, type=float, nargs=2, default=[2700.0, 2700.0], metavar=("dim_length", "dim_height"),
            help="Dimensions of radiation source search area in cm, decreased by area_obs param. to ensure visilibity graph setup is valid. Length by height.",
        --area-obs, type=float, nargs=2, default=[200.0, 500.0], metavar=("area_obs_min", "area_obs_max"),
            help="Interval for each obstruction area in cm. This is how much to remove from bounds to make the 'visible bounds'",
        --obstruct, type= int, #Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7], default=-1,
            help="Number of obstructions present in each episode, options: -1 -> random sampling from [1,5], 0 -> no obstructions, [1-7] -> 1 to 7 obstructions",
        --enforce-boundaries, type=bool, default=False,
            help="Indicate whether or not agents can travel outside of the search area"
  
    Hyperparameters and PPO parameters:
        --gamma, type=float, default=0.99,
            help="Reward attribution for advantage estimator for PPO updates",
        --alpha, type=float, default=0.1, 
            help="Entropy reward term scaling"
        --minibatches, type=int, default=1, 
            help="Batches to sample data during actor policy update (k_epochs)"

    Parameters for Neural Networks:
        --net-type, type=str, default="rnn",
            help="Choose between convolutional neural network, recurrent neural network, MLP Actor-Critic (A2C , feed forward, or uniform option: cnn, rnn, mlp, ff, uniform",
    
    Parameters for RAD-A2C:
        --hid-pol, type=int, default=32, 
            help="Actor linear layer size (Policy Hidden Layer Size "
        --hid-val, type=int, default=32, 
            help="Critic linear layer size (State-Value Hidden Layer Size "
        --hid-rec, type=int, default=24, 
            help="PFGRU hidden state size (Localization Network "
        --hid-gru, type=int, default=24, 
            help="Actor-Critic GRU hidden state size (Embedding Layers "
        --l-pol, type=int, default=1, 
            help="Number of layers for Actor MLP (Policy Multi-layer Perceptron "
        --l-val, type=int, default=1, 
            help="Number of layers for Critic MLP (State-Value Multi-layer Perceptron "
    '''
    hid_gru: int 
    hid_pol: int # test 
    hid_val: int
    hid_rec: int
    l_pol: int
    l_val: int
    gamma: float
    seed: int
    steps_per_epoch: int
    steps_per_episode: int
    epochs: int
    exp_name: str
    dims: tuple[int, int]
    area_obs: tuple[int, int]
    obstruct: Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7]
    net_type: str
    alpha: float
    render: bool
    agent_count: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    enforce_boundaries: bool
    minibatches: int
    env_name: str
    save_freq: int
    save_gif_freq: int
    DEBUG: bool

def parse_args(parser: argparse.ArgumentParser) -> CliArgs:
    ''' Function to parge command line arguments

        Args: The parser from argparse module with read-in arguments
        
        Return: Command line argument class-object containing read-in arguments
    '''
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
        steps_per_episode=args.steps_per_episode,
        epochs=args.epochs,
        exp_name=args.exp_name,
        dims=args.dims,
        area_obs=args.area_obs,
        obstruct=args.obstruct,
        net_type=args.net_type,
        alpha=args.alpha,
        render=args.render,
        agent_count=args.agent_count,
        enforce_boundaries=args.enforce_boundaries,
        minibatches=args.minibatches,
        env_name=args.env_name,
        save_freq=args.save_freq,
        save_gif_freq=args.save_gif_freq,
        DEBUG=args.DEBUG
    )


def create_parser() -> argparse.ArgumentParser:
    ''' 
        Function to generate argument parser
        Returns: argument parser with command line arguments
    '''
    parser = argparse.ArgumentParser()
    
    # General parameters
    parser.add_argument(
        "--DEBUG",
        type=bool,
        default=False,
        help="Enable DEBUG mode - contains extra logging and set minimal setups",
    )        
    parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=120,
        help="Number of timesteps per episode (before resetting the environment)",
    )    
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
        "--agent-count", type=int, # Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        default=1, 
        help="Number of agents"
    )   
    parser.add_argument(
        "--render", type=bool, default=False, help="Save gif"
    )          
    parser.add_argument(
        "--save-gif-freq", type=int, default=float('inf'), help="If render is true, save gif after this many epochs."
    )     
    parser.add_argument(
        "--save-freq", type=int, default=500, help="How often to save the model."
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
        "--enforce-boundaries", type=bool, default=False, help="Indicate whether or not agents can travel outside of the search area"
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
        help="Choose between convolutional neural network, recurrent neural network, MLP Actor-Critic (A2C), feed forward, or uniform option: cnn, rnn, mlp, ff, uniform",
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

def ping():
    return 'Pong!'

def main():
    ''' Read arguments from command line'''
    args = parse_args(create_parser())

    ''' Save directory and experiment name ''' 
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
    rng = npr.default_rng(robust_seed)

    # Set up logger args 
    timestamp = datetime.now().replace(microsecond=0).strftime('%Y-%m-%d-%H:%M:%S')
    exp_name = timestamp + "_" + exp_name
    save_dir_name = save_dir_name + '/' + timestamp
    logger_kwargs = {'exp_name': exp_name, 'seed': args.seed, 'data_dir': "../../models/train", 'env_name': save_dir_name}   

    # Set up Radiation environment
    dim_length, dim_height = args.dims
    intial_parameters = {
        'bbox': np.array(  # type: ignore
            [[0.0, 0.0], [dim_length, 0.0], [dim_length, dim_height], [0.0, dim_height]]
        ),
        'observation_area': np.array(args.area_obs),  # type: ignore
        'obstruction_count': args.obstruct,
        'np_random': rng,
        'number_agents': args.agent_count,
        'save_gif': args.render,
        'enforce_grid_boundaries': args.enforce_boundaries,
        'DEBUG': args.DEBUG
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
    simulation = train.train_PPO(
        env=env,
        logger_kwargs=logger_kwargs,
        ac_kwargs=dict(
            hidden_sizes_pol=[[args.hid_pol]] * args.l_pol,
            hidden_sizes_val=[[args.hid_val]] * args.l_val,
            hidden_sizes_rec=[args.hid_rec],
            hidden=[[args.hid_gru]],
            net_type=args.net_type,
            batch_s=args.minibatches,
            enforce_boundaries=args.enforce_boundaries
        ),
        gamma=args.gamma,
        alpha=args.alpha,
        seed=robust_seed,
        steps_per_epoch=args.steps_per_epoch,
        steps_per_episode=args.steps_per_episode,
        total_epochs=args.epochs,
        number_of_agents=args.agent_count,
        render=args.render,
        save_gif=args.render, # TODO combine into just render
        save_freq=args.save_freq,
        save_gif_freq=args.save_gif_freq,
        actor_critic_architecture=args.net_type,
        enforce_boundaries=args.enforce_boundaries
    )
    
    simulation.train()

if __name__ == "__main__":
    main()