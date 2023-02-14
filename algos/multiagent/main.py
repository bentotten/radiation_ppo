''' 
Start Rad-Team simulation 
'''
import argparse
from dataclasses import dataclass
from typing import Literal
from datetime import datetime

import numpy as np
import numpy.random as npr
from typing import NamedTuple

import gym
from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore
from gym_rad_search.envs import RadSearch  # type: ignore
from gym_rad_search.envs import rad_search_env

try:
    import NeuralNetworkCores.RADA2C_core as RADA2C_core
    from epoch_logger import setup_logger_kwargs, EpochLogger
    import train as train
    from ppo import BpArgs
except:
    import algos.multiagent.NeuralNetworkCores.RADA2C_core as RADA2C_core
    from algos.multiagent.epoch_logger import setup_logger_kwargs, EpochLogger
    import algos.multiagent.train as train
    from algos.multiagent.ppo import BpArgs    


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
            help="Interval for each obstruction area in cm. This is how much to remove from dimensions of radiation source search area to spawn starting location and obstacles in",
        --obstruct, type= int, #Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7], default=-1,
            help="Number of obstructions present in each episode, options: -1 -> random sampling from [1,5], 0 -> no obstructions, [1-7] -> 1 to 7 obstructions",
        --enforce-boundaries, type=bool, default=False,
            help="Indicate whether or not agents can travel outside of the search area"
  
    Hyperparameters and PPO parameters:
        "--lam", type=float, default=0.9, 
            help="Lamda - Smoothing parameter for GAE-Lambda advantage estimator calculations (Always between 0 and 1, close to 1.)"    
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
    lam: float
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
    clip_ratio: float
    target_kl: float
    render: bool
    agent_count: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    enforce_boundaries: bool
    minibatches: int
    env_name: str
    save_freq: int
    save_gif_freq: int
    actor_learning_rate: float
    critic_learning_rate: float
    pfgru_learning_rate: float
    train_pi_iters: float
    train_v_iters: float
    train_pfgru_iters: float
    reduce_pfgru_iters: float
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
        lam=args.lam,
        clip_ratio=args.clip_ratio,
        target_kl=args.target_kl,
        render=args.render,
        agent_count=args.agent_count,
        enforce_boundaries=args.enforce_boundaries,
        minibatches=args.minibatches,
        env_name=args.env_name,
        save_freq=args.save_freq,
        save_gif_freq=args.save_gif_freq,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        pfgru_learning_rate=args.pfgru_learning_rate,
        train_pi_iters=args.train_pi_iters,
        train_v_iters=args.train_v_iters,
        train_pfgru_iters=args.train_pfgru_iters,
        reduce_pfgru_iters=args.reduce_pfgru_iters,        
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
        "--lam", type=float, default=0.9, help="Lamda - Smoothing parameter for GAE-Lambda advantage estimator calculations (Always between 0 and 1, close to 1.)"
    )    
    parser.add_argument(
        "--clip_ratio", type=float, default=0.2, help="Usually seen as Epsilon Hyperparameter for clipping in the policy objective. Roughly: how far can the new policy go from the old policy while \
            still profiting (improving the objective function)? The new policy can still go farther than the clip_ratio says, but it doesn't help on the objective anymore. \
            (Usually small, 0.1 to 0.3.).Basically if the policy wants to perform too large an update, it goes with a clipped value instead."
    )
    parser.add_argument(
        "--target_kl", type=float, default=0.07, help="Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.) "
    )   
    parser.add_argument(
        "--minibatches", type=int, default=1, help="Batches to sample data during actor policy update (k_epochs)"
    )
    parser.add_argument(
        "--actor_learning_rate", type=float, default=3e-4, help="Learning rate for Actor/policy optimizer."
    )
    parser.add_argument(
        "--critic_learning_rate", type=float, default=1e-3, help="Learning rate for Critic/value optimizer."
    )
    parser.add_argument(
        "--pfgru_learning_rate", type=float, default=5e-3, help="Learning rate for location prediction neural network module."
    )
    parser.add_argument(
        "--train_pi_iters", type=int, default=40, help="Maximum number of gradient descent steps to take on actor policy loss per epoch. NOTE: Early stopping may cause optimizer to take fewer than this."
    )
    parser.add_argument(
        "--train_v_iters", type=int, default=40, help="Maximum number of gradient descent steps to take on critic state-value function per epoch."
    )    
    parser.add_argument(
        "--train_pfgru_iters", type=int, default=15, help="Maximum number of gradient descent steps to take for source localization neural network (the PFGRU unit)."
    )        
    parser.add_argument(
        "--reduce_pfgru_iters", type=bool, default=True, help="Reduce PFGRU training after a certain number of iterations when further along to speed up training."
    )        
    
    # Parameters for RAD-A2C
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
    ''' Check Function call '''
    return 'Pong!'


def main():
    ''' Set up experiment and create simulation environment. '''
    args = parse_args(create_parser())

    save_dir_name: str = args.exp_name  # Stands for bootstrap particle filter, one of the neat resampling methods used
    exp_name: str = (args.exp_name + "_agents" + str(args.agent_count))

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
    
    # Bootstrap particle filter args for the PFGRU, from Particle Filter Recurrent Neural Networks by Ma et al. 2020.
    bp_args = BpArgs(
        bp_decay=0.1,
        l2_weight=1.0,
        l1_weight=0.0,
        elbo_weight=1.0,
        area_scale=env.search_area[2][1]
    )    
    
    # Set up static A2C actor-critic args
    # TODO switch all of these to switch statements
    if args.net_type == 'mlp' or args.net_type =='rnn':
        ac_kwargs=dict(
            hidden_sizes_pol=[[args.hid_pol]] * args.l_pol,
            hidden_sizes_val=[[args.hid_val]] * args.l_val,
            hidden_sizes_rec=[args.hid_rec],
            hidden=[[args.hid_gru]],
            net_type=args.net_type,
            batch_s=args.minibatches,
            enforce_boundaries=args.enforce_boundaries,
            seed=args.seed,
            pad_dim=2                     
        )
    else:
        ac_kwargs=dict(
            number_of_agents=args.agent_count,
            observation_space=env.observation_space.shape[0], # Also known as state dimensions: The dimensions of the observation returned from the environment
            action_space=env.detectable_directions,
            steps_per_episode=args.steps_per_episode,
            steps_per_epoch=args.steps_per_epoch,
            scaled_grid_bounds=(1, 1),        

            net_type=args.net_type,
            batch_s=args.minibatches,
            enforce_boundaries=args.enforce_boundaries,
            seed=args.seed,
            pad_dim=2                     
        )        

    # Set up static PPO args. NOTE: Shared data structure between agents, do not add dynamic data here
    ppo_kwargs=dict(
        observation_space=env.observation_space.shape[0],     # Also known as state dimensions: The dimensions of the observation returned from the environment
        bp_args=bp_args,
        steps_per_epoch=args.steps_per_epoch,
        
        actor_critic_args=ac_kwargs,
                
        bounds_offset=env.observation_area,
        detector_step_size=env.step_size,
        environment_scale=env.scale,
        actor_critic_architecture=args.net_type,        
        seed=args.seed,
        minibatch=args.minibatches,
        enforce_boundaries=args.enforce_boundaries,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        pfgru_learning_rate=args.pfgru_learning_rate,
        train_pi_iters=args.train_pi_iters,
        train_v_iters=args.train_v_iters,
        train_pfgru_iters=args.train_pfgru_iters,
        reduce_pfgru_iters=args.reduce_pfgru_iters,
        clip_ratio=args.clip_ratio,
        alpha=args.alpha,
        target_kl=args.target_kl,
        gamma=args.gamma,        
        lam=args.lam
    )

    # Set up training
    simulation = train.train_PPO(
        env=env,
        logger_kwargs=logger_kwargs,
        ppo_kwargs=ppo_kwargs,
        seed=robust_seed,
        number_of_agents=args.agent_count,
        actor_critic_architecture=args.net_type,            
        steps_per_epoch=args.steps_per_epoch,
        steps_per_episode=args.steps_per_episode,
        total_epochs=args.epochs,
        render=args.render,
        save_gif=args.render, # TODO combine into just render
        save_freq=args.save_freq,
        save_gif_freq=args.save_gif_freq,
        DEBUG=args.DEBUG
    )
    
    # Begin simulation
    simulation.train()

if __name__ == "__main__":
    main()