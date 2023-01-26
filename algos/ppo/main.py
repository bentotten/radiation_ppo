import argparse
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.random as npr

from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore

import core
from epoch_logger import setup_logger_kwargs, EpochLogger
import ppo
from gym_rad_search.envs import RadSearch  # type: ignore


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
    obstruct: Literal[-1, 0, 1]
    net_type: str
    alpha: float
    render: bool


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
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hid-gru", type=int, default=24, help="A2C GRU hidden state size"
    )
    parser.add_argument(
        "--hid-pol", type=int, default=32, help="Actor linear layer size"
    )
    parser.add_argument(
        "--hid-val", type=int, default=32, help="Critic linear layer size"
    )
    parser.add_argument(
        "--hid-rec", type=int, default=24, help="PFGRU hidden state size"
    )
    parser.add_argument(
        "--l-pol", type=int, default=1, help="Number of layers for Actor MLP"
    )
    parser.add_argument(
        "--l-val", type=int, default=1, help="Number of layers for Critic MLP"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Reward attribution for advantage estimator",
    )
    parser.add_argument("--seed", type=int, default=2, help="Random seed control")
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=480,
        help="Number of timesteps per epoch",
    )
    parser.add_argument(
        "--epochs", type=int, default=3000, help="Number of epochs to train the agent"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="alpha01_tkl07_val01_lam09_npart40_lr3e-4_proc10_obs-1_iter40_blr5e-3_2_tanh",
        help="Name of experiment for saving",
    )
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
        help="Interval for each obstruction area in cm",
    )
    parser.add_argument(
        "--obstruct",
        type=Literal[-1, 0, 1],
        default=-1,
        help="Number of obstructions present in each episode, options: -1 -> random sampling from [1,5], 0 -> no obstructions, [1-7] -> 1 to 7 obstructions",
    )
    parser.add_argument(
        "--net-type",
        type=str,
        default="rnn",
        help="Choose between recurrent neural network A2C or MLP A2C, option: rnn, mlp",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Entropy reward term scaling"
    )
    parser.add_argument(
        "--render", type=bool, default=False, help="Render Gif"
    )
    return parser


if __name__ == "__main__":
    args = parse_args(create_parser())

    # Change mini-batch size, only been tested with size of 1
    batch_s: int = 1

    # Save directory and experiment name
    save_gif = True
    env_name: str = "bpf"
    exp_name: str = (
        "loc"
        + str(args.hid_rec)
        + "_hid"
        + str(args.hid_gru)
        + "_pol"
        + str(args.hid_pol)
        + "_val"
        + str(args.hid_val)
        + "_"
        + args.exp_name
        + f"_ep{args.epochs}"
        + f"_steps{args.steps_per_epoch}"
    )

    # Generate a large random seed and random generator object for reproducibility
    robust_seed = _int_list_from_bigint(hash_seed(args.seed))[0]
    rng = npr.default_rng(robust_seed)

    dim_length, dim_height = args.dims
    logger_kwargs = setup_logger_kwargs(
        exp_name, args.seed, data_dir="../../models/train", env_name=env_name
    )
    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    
    # Number of agents
    number_of_agents = 1

    env: RadSearch = RadSearch(
        bbox=np.array(  # type: ignore
            [[0.0, 0.0], [dim_length, 0.0], [dim_length, dim_height], [0.0, dim_height]]
        ),
        observation_area=np.array(args.area_obs),  # type: ignore
        obstruction_count=args.obstruct,
        np_random=rng,
        number_agents = number_of_agents
    )

    # Run ppo training function
    ppo = ppo.PPO(
        env=env,
        actor_critic=core.RNNModelActorCritic,
        logger=logger,
        ac_kwargs=dict(
            hidden_sizes_pol=[[args.hid_pol]] * args.l_pol,
            hidden_sizes_val=[[args.hid_val]] * args.l_val,
            hidden_sizes_rec=[args.hid_rec],
            hidden=[[args.hid_gru]],
            net_type=args.net_type,
            batch_s=batch_s,
        ),
        gamma=args.gamma,
        alpha=args.alpha,
        seed=robust_seed,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        number_of_agents=number_of_agents,
        render=args.render,
        save_gif=save_gif,
    )
