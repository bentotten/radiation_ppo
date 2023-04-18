'''
Start Rad-Team simulation
'''
import argparse
from dataclasses import dataclass
from typing import Literal
from datetime import datetime

import numpy as np
import numpy.random as npr
from typing import Tuple
import inspect
import traceback
import json
import os

import gym  # type: ignore
from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore
from gym_rad_search.envs import RadSearch  # type: ignore


try:
    import train  # type: ignore
    from ppo import BpArgs  # type: ignore
    import evaluate # type: ignore
    from rl_tools.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads # type: ignore
    from rl_tools.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs # type: ignore
except ModuleNotFoundError:
    import algos.multiagent.train as train  # type: ignore
    import algos.multiagent.train_remote as train_remote  # type: ignore
    from algos.multiagent.ppo import BpArgs  # type: ignore
    import algos.multiagent.evaluate as evaluate  # type: ignore
    from algos.multiagent.rl_tools.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads # type: ignore
    from algos.multiagent.rl_tools.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs # type: ignore
except:
    raise Exception

PROFILE = True

def log_state(error: Exception):
    trace_back = traceback.format_exc()  # Gives error and location
    trace = inspect.trace()
    #vars = json.dumps(trace[-1].frame.f_locals, indent = 4) # Non-serializable

    # vfile = open('RADTEAM_ERROR_STATE.log', 'w')
    # vfile.write(vars)
    # vfile.close()
    tfile = open('RADTEAM_ERROR_TRACEBACK.log', 'w')
    tfile.write(trace_back)
    tfile.close()

# TODO Implement a load from a config file instead of from cli args


@dataclass
class CliArgs:
    ''' Parameters passed in through the command line

    General parameters:
        --DEBUG, type=bool, default=False,
            help="Enable DEBUG mode - contains extra logging and set minimal setups"
        --mode, type=str, default='train',
            help="Running mode - train: train a model. evaluate: run 100 monte carlo simulations and save results",
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
        --net-type, type=str, default="cnn",
            help="Choose between convolutional neural network, recurrent neural network, MLP Actor-Critic (A2C , feed forward, or uniform option: cnn, rnn, mlp, ff, uniform",

    Parameters for RAD-TEAM
        "--resolution-multiplier", type=float, default=0.01, help="Indicate degree of accuracy a heatmap should be downsized to. A value of 1 is full accuracy - not recommended for most training environments (see documentation)"
        "--global-critic", type=bool, default=True, help="Indicate if each agent should have their own critic or a global."

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
    dims: Tuple[int, int]
    area_obs: Tuple[int, int]
    obstruct: Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7]
    net_type: str
    alpha: float
    clip_ratio: float
    target_kl: float
    render: bool
    agent_count: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    enforce_boundaries: bool
    resolution_multiplier: float
    global_critic: bool
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
    mode: str


''' Function to parge command line arguments '''
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
        resolution_multiplier=args.resolution_multiplier,
        global_critic=args.global_critic,
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
        DEBUG=args.DEBUG,
        mode=args.mode
    )

''' Function to generate argument parser '''
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
        "--mode",
        type=str,
        default='train',
        help="Running mode - train: train a model. evaluate: run 100 monte carlo simulations and save results",
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
        default="RADTEAM",
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
        "--enforce-boundaries", type=bool, default=True, help="Indicate whether or not agents can travel outside of the search area"
    )
    parser.add_argument(
        "--resolution-multiplier", type=float, default=0.01, help="Indicate degree of accuracy a heatmap should be downsized to. A value of 1 is full accuracy - not recommended for most training environments (see documentation)"
    )
    parser.add_argument(
        "--global-critic", action=argparse.BooleanOptionalAction, default=True, help="Indicate if each agent should have their own critic or a global."
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
        default="cnn",
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


def main() -> None:
    ''' Set up experiment and create simulation environment. '''
    args = parse_args(create_parser())

    # Setup MPI

    # Steps in batch must be greater than the max epoch steps times num. of cpu
    if args.DEBUG:
        cpus = 1
    else:
        cpus = 1
        #cpus = int(os.cpu_count()/2)
    if cpus > 1:
        tot_epoch_steps = cpus * args.steps_per_epoch if cpus else args.steps_per_epoch
        tot_epoch_steps = tot_epoch_steps if tot_epoch_steps > args.steps_per_epoch else args.steps_per_epoch
        print(f'Sys cpus {cpus}; Total Steps set to {tot_epoch_steps}')
        mpi_fork(cpus)  # run parallel code with mpi

        args.steps_per_epoch = int(tot_epoch_steps / num_procs())

    save_dir_name: str = args.exp_name
    exp_name: str = (args.exp_name + "_agents" + str(args.agent_count))

    # Generate a large random seed and random generator object for reproducibility
    #Generate a large random seed and random generator object for reproducibility
    robust_seed = _int_list_from_bigint(hash_seed((1+proc_id())*args.seed))[0]
    rng = npr.default_rng(robust_seed)

    # Set up logger args
    timestamp = datetime.now().replace(microsecond=0).strftime('%Y-%m-%d-%H:%M:%S')
    exp_name = timestamp + "_" + exp_name
    save_dir_name = save_dir_name + '/' + timestamp
    logger_kwargs = {
        'exp_name': exp_name,
        'seed': args.seed,
        'data_dir': "../../models/train",
        'env_name': save_dir_name
        }

    save_path = [f"../../models/train/{save_dir_name}", f"{exp_name}_s{args.seed}"] # TODO turn into a parameter

    # Set up Radiation environment
    dim_length, dim_height = args.dims
    env_kwargs = {
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

    if PROFILE:
        import cProfile, pstats
        profiler = cProfile.Profile()
        profiler.enable()

    # Set up training
    if args.mode == 'train':
        env: RadSearch = gym.make(args.env_name,**env_kwargs)
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
        if args.net_type == 'mlp' or args.net_type =='rnn':
            ac_kwargs=dict(
                obs_dim=env.observation_space.shape[0],
                act_dim=env.detectable_directions,
                hidden_sizes_pol=[[args.hid_pol]] * args.l_pol,
                hidden_sizes_val=[[args.hid_val]] * args.l_val,
                hidden_sizes_rec=[args.hid_rec],
                hidden=[[args.hid_gru]],
                net_type=args.net_type,
                batch_s=args.minibatches,
                seed=robust_seed,
                pad_dim=2
            )
        else:
            ac_kwargs=dict(
                action_space=env.detectable_directions, # Usually 8
                observation_space=env.observation_space.shape[0], # Also known as state dimensions: The dimensions of the observation returned from the environment. Usually 11
                steps_per_episode=args.steps_per_episode,
                number_of_agents=args.agent_count,
                detector_step_size=env.step_size, # Usually 100 cm
                environment_scale=env.scale,
                bounds_offset=env.observation_area,
                enforce_boundaries=args.enforce_boundaries,
                grid_bounds=env.scaled_grid_max,
                resolution_multiplier=args.resolution_multiplier,
                GlobalCritic=None,
                save_path = save_path
            )

        # Set up static PPO args. NOTE: Shared data structure between agents, do not add dynamic data here
        ppo_kwargs=dict(
            observation_space=env.observation_space.shape[0],
            bp_args=bp_args,
            steps_per_epoch=args.steps_per_epoch,
            steps_per_episode=args.steps_per_episode,
            number_of_agents=args.agent_count,
            env_height=env.search_area[2][1],
            actor_critic_args=ac_kwargs,
            actor_critic_architecture=args.net_type,
            minibatch=args.minibatches,
            train_pi_iters=args.train_pi_iters,
            train_v_iters=args.train_v_iters,
            train_pfgru_iters=args.train_pfgru_iters,
            reduce_pfgru_iters=args.reduce_pfgru_iters,
            actor_learning_rate=args.actor_learning_rate,
            critic_learning_rate=args.critic_learning_rate,
            pfgru_learning_rate=args.pfgru_learning_rate,
            gamma=args.gamma,
            alpha=args.alpha,
            clip_ratio=args.clip_ratio,
            target_kl=args.target_kl,
            lam=args.lam,
            GlobalCriticOptimizer=None
        )

        simulation = train.train_PPO(
            env=env,
            logger_kwargs=logger_kwargs,
            ppo_kwargs=ppo_kwargs,
            seed=robust_seed,
            number_of_agents=args.agent_count,
            actor_critic_architecture=args.net_type,
            global_critic_flag = args.global_critic,
            steps_per_epoch=args.steps_per_epoch,
            steps_per_episode=args.steps_per_episode,
            total_epochs=args.epochs,
            render=args.render,
            save_path=save_path,
            save_gif=args.render, # TODO combine into just render
            save_freq=args.save_freq,
            save_gif_freq=args.save_gif_freq,
            DEBUG=args.DEBUG
        )

        # try:
        #     # Begin simulation
        #     simulation.train()
        # except Exception as err:
        #     log_state(err)
        simulation.train()

    elif args.mode == 'evaluate':
        # Game mode notes:
        #   Cooperative = Team reward, global critic
        #   Collaborative = shared observations, individual nnets
        #   Competative = zero-sum reward framework
        #   Individual = No shared observations, individual nnets. Other agents treated as part of environment
        
        # TODO move to CLI
        eval_kwargs=dict(
            env_name = args.env_name,
            test_env_path = (lambda: os.getcwd() + '/evaluation/test_environments')(),
            env_kwargs=env_kwargs,
            #model_path='./evaluation/saves/2023-03-02-13:39:06', # Specify model directory (fpath) RAD-TEAM old with bad reset
            #model_path='./evaluation/saves/2023-04-16-12:20:31', # Specify model directory (fpath) RAD-A2C
            model_path='./evaluation/saves/2023-04-16-01:52:15',
            episodes=100, # Number of episodes to test on [1 - 1000]
            montecarlo_runs=100, # Number of Monte Carlo runs per episode (How many times to run/sample each episode setup) (mc_runs)
            actor_critic_architecture=args.net_type, # Neural network type (control)
            snr='none', # signal to noise ratio [none, low, medium, high]
            obstruction_count=args.obstruct, # number of obstacles [0 - 7] (num_obs)
            steps_per_episode=args.steps_per_episode,
            number_of_agents=args.agent_count,
            enforce_boundaries=args.enforce_boundaries,
            resolution_multiplier=args.resolution_multiplier,
            team_mode=((lambda flag: 'cooperative' if flag else 'individual')(flag = args.global_critic)), 
            render=args.render,
            save_gif_freq=args.save_gif_freq,
            render_path='.',
            save_path_for_ac=save_path,
            seed=robust_seed
        )

        simulation = evaluate.evaluate_PPO(eval_kwargs=eval_kwargs)

        simulation.evaluate()

    if PROFILE:
        profiler.disable()
        # cumtime is the cumulative time spent in this and all subfunctions
        with open(f"{save_path[0]}/profile_cumtime.txt", 'w') as stream:
            stats = pstats.Stats(profiler,  stream=stream).sort_stats('cumtime')
            stats.print_stats()

        stream.close()

        # tottime is a total of the time spent in the given function
        with open(f"{save_path[0]}/profile_tottime.txt", 'w') as stream:
            stats = pstats.Stats(profiler,  stream=stream).sort_stats('tottime')
            stats.print_stats()

        # print("##### BY CUMTIME #####")
        # stats = pstats.Stats(profiler).sort_stats('cumtime')
        # stats.print_stats()
        # stats.dump_stats(f"{save_path[0]}/profile_cumtime.txt")
        # print("##### BY TOTTIME #####")
        # stats = pstats.Stats(profiler).sort_stats('tottime')
        # #stats.print_stats()
        # stats.dump_stats(f"{save_path[0]}/profile_tottime.txt")

    #     try:
    #         # Begin simulation
    #         simulation.evaluate()
    #     except Exception as err:
    #         log_state(err)
    else:
        raise Exception("Unknown mode specified. Acceptable modes: train, evaluate")

if __name__ == "__main__":
    main()
