# type: ignore
"""
    NOTE: This is a duplicate PPO for testing purposes only! 
"""

import torch
from torch.optim import Adam
import torch.nn.functional as F

import numpy as np
import numpy.typing as npt

from dataclasses import dataclass, field
from typing_extensions import TypeAlias  # type: ignore
from typing import Union, cast, Optional, Any, NamedTuple, Tuple, Dict, List, Dict
import scipy.signal  # type: ignore

from rl_tools.mpi_tools import mpi_statistics_scalar  # type: ignore


# If prioritizing memory, only keep observations and reinflate heatmaps when update happens. Reduces memory requirements, but greatly slows down training.
PRIO_MEMORY = False

Shape: TypeAlias = Union[int, Tuple[int], Tuple[int, Any], Tuple[int, int, Any]]


def combined_shape(length: int, shape: Optional[Shape] = None) -> Shape:
    """
    This method combines dimensions. It combines length and existing shape dimension into a new tuple representing dimensions (useful for numpy.zeros() or tensor creation).
    Length is in x position. If shape is a tuple, flatten it and add it to remaining tuple positions. Returns dimensions of new shape.

    Example 1 : Size (steps_per_epoch) - Make a buffer to store advantages for an epoch. Returns (x, )
    Example 2: Size (steps_per_epoch, 2) - Make a buffer for source locations (x, y) for an epoch. Returns Returns (x, 2)
    Example 3: Size (steps_per_epoch, num_agents, observation_dimensions) - Make a buffer for agent observations for an epoch. Returns (x, n, 11)

    :param length: (int) X position of tuple.
    :param shape: (int | Tuple[int, Any]) remaining positions in tuple.
    """
    # See Example 1
    if shape is None:
        return (length,)

    # See Example 2
    elif np.isscalar(shape):
        shape = cast(int, shape)
        return (length, shape)

    # See Example 3
    else:
        shape = cast(Tuple[int, Any], shape)
        return (length, *shape)


def discount_cumsum(
    x: npt.NDArray[np.float64], discount: float
) -> npt.NDArray[np.float64]:
    """
    Function from rllab for computing discounted cumulative sums of vectors.
    See: https://docs.scipy.org/doc/scipy/tutorial/signal.html#difference-equation-filtering

    Input: vector x,
        [x0,
        x1,
        x2]

    Output:
        [x0 + discount * x1 + discount^2 * x2,
        x1 + discount * x2,
        x2]

    :param x: Vector to apply discounts to.
    :param discount: Discounts to be applied to calculations

    :return:

    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class UpdateResult(NamedTuple):
    """Object that contains the return values from the neural network updates"""

    stop_iteration: int
    loss_policy: float
    loss_critic: Union[float, None]
    loss_predictor: float
    kl_divergence: npt.NDArray[np.float32]
    Entropy: npt.NDArray[np.float32]
    ClipFrac: npt.NDArray[np.float32]
    LocLoss: Union[torch.Tensor, None]
    VarExplain: int  # TODO what is this?


class BpArgs(NamedTuple):
    """Object that contains the parameters for bootstrap particle filter"""

    bp_decay: float
    l2_weight: float
    l1_weight: float
    elbo_weight: float
    area_scale: float


@dataclass
class OptimizationStorage:
    """
    Class that stores information related to updating neural network models for each agent. It includes the clip ratio for
    ensuring a destructively large policy update doesn't happen, an entropy parameter for randomness/entropy, and the target kl divergence
    for early stopping.

    :param critic_flag: (bool) Indicate if there is a critic
    :param {*}_optimizer: (torch.optim) Pytorch Optimizer with learning rate decay [Torch]
    :param clip_ratio: (float) Hyperparameter for clipping in the policy objective. Roughly: how far can the new policy go from the old policy
        while still profiting (improving the objective function)? The new policy can still go farther than the clip_ratio says, but it doesn't
        help on the objective anymore. This is usually small, often 0.1 to 0.3, and is typically denoted by :math:`\epsilon`. Basically if the
        policy wants to perform too large an update, it goes with a clipped value instead.
    :param alpha: (float) Entropy reward term scaling used during calculating loss.
    :param target_kl: (float) Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used
        for early stopping. It's usually small, 0.01 or 0.05.
    """

    critic_flag: bool
    pi_optimizer: torch.optim.Optimizer
    critic_optimizer: Union[torch.optim.Optimizer, None]
    model_optimizer: torch.optim.Optimizer

    # Initialized elsewhere
    #: Schedules gradient steps for actor
    pi_scheduler: torch.optim.lr_scheduler.StepLR = field(init=False)
    #: Schedules gradient steps for value function (critic)
    critic_scheduler: Union[torch.optim.lr_scheduler.StepLR, None] = field(init=False)
    #: Schedules gradient steps for PFGRU location predictor module
    pfgru_scheduler: torch.optim.lr_scheduler.StepLR = field(init=False)
    #: Loss calculator utility for Critic
    MSELoss: torch.nn.modules.loss.MSELoss = field(
        default_factory=(lambda: torch.nn.MSELoss(reduction="mean"))
    )

    def __post_init__(self):
        self.pi_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pi_optimizer, step_size=100, gamma=0.99
        )
        self.model_scheduler = torch.optim.lr_scheduler.StepLR(
            self.model_optimizer, step_size=100, gamma=0.99
        )

        if self.critic_flag:
            self.critic_scheduler = torch.optim.lr_scheduler.StepLR(
                self.critic_optimizer, step_size=100, gamma=0.99
            )
        else:
            self.critic_scheduler = None  # RAD-A2C has critic embeded in pi


# Ok now
@dataclass
class PPOBuffer:
    """
    A buffer for storing histories/trajectories experienced by a PPO agent interacting with the environment, and using Generalized Advantage Estimation (GAE-Lambda) for calculating the
    advantages of state-action pairs. This is left outside of the PPO agent so that A2C architectures can be swapped out as desired.

    :param observation_dimension: (int) Dimensions of observation. For RAD-TEAM and RAD-A2C the observation will be a one dimensional array/tuple where the first element is the
        detected radiation intensity, the second and third elements are the x,y coordinates, and the remaining 8 elements are a reading of how close an agent is to an obstacle.
        Obstacle sensor readings and x,y coordinates are normalized.
    :param max_size: Max steps per epoch.
    :param max_episode_length: (int) Maximum steps per episode
    :param number_agents: Number of agents.
    :param gamma: (float) Discount rate for expected return and Generalize Advantage Estimate (GAE) calculations (Always between 0 and 1.)
    :param lam: (float) Exponential weight decay/discount; controls the bias variance trade-off for Generalize Advantage Estimate (GAE) calculations (Always between 0 and 1, close to 1)
    """

    observation_dimension: int
    max_size: int
    max_episode_length: int
    number_agents: int

    ptr: int = field(
        init=False
    )  # For keeping track of location in buffer during update
    path_start_idx: int = field(
        init=False
    )  # For keeping track of starting location in buffer during update

    episode_lengths_buffer: List = field(init=False)  # Stores episode lengths
    # In memory-priority mode, for each timestep, stores every agents observation
    full_observation_buffer: List[
        Dict[Union[int, str], Union[npt.NDArray[np.float32], bool, None]]
    ] = field(init=False)
    heatmap_buffer: Dict[str, List[torch.Tensor]] = field(
        init=False
    )  # When memory is not a concern, for each timestep, stores every steps heatmap stack for both actor and critic

    obs_buf: npt.NDArray[np.float32] = field(
        init=False
    )  # Observation buffer for each agent
    act_buf: npt.NDArray[np.float32] = field(
        init=False
    )  # Action buffer for each step. Note: each agent carries their own PPO buffer, no need to track all agent actions.
    adv_buf: npt.NDArray[np.float32] = field(
        init=False
    )  # Advantages buffer for each step
    rew_buf: npt.NDArray[np.float32] = field(init=False)  # Rewards buffer for each step
    ret_buf: npt.NDArray[np.float32] = field(
        init=False
    )  # Rewards-to-go buffer (Rewards gained from timestep t until terminal state (similar to expected return, but actual))
    val_buf: npt.NDArray[np.float32] = field(
        init=False
    )  # State-value buffer for each step
    source_tar: npt.NDArray[np.float32] = field(
        init=False
    )  # Source location buffer (for moving targets)
    logp_buf: npt.NDArray[np.float32] = field(
        init=False
    )  # action log probabilities buffer
    location_pred_buf: npt.NDArray[np.float32] = field(
        init=False
    )  # Store location predictions

    obs_win: npt.NDArray[np.float32] = field(
        init=False
    )  # For location prediction TODO find out what its doing
    obs_win_std: npt.NDArray[np.float32] = field(
        init=False
    )  # For location prediction TODO find out what its doing

    gamma: float = 0.99  # trajectory discount for Generalize Advantage Estimate (GAE)
    lam: float = 0.90  # exponential mean discount Generalize Advantage Estimate (GAE). Can be thought of like a smoothing parameter.

    def __post_init__(self):
        self.ptr = 0
        self.path_start_idx = 0

        self.episode_lengths_buffer = list()

        if PRIO_MEMORY:
            self.full_observation_buffer = list()
            for i in range(self.max_size):
                self.full_observation_buffer.append({})
                for id in range(self.number_agents):
                    self.full_observation_buffer[i][id] = np.zeros(
                        (self.observation_dimension,)
                    )
                    self.full_observation_buffer[i]["terminal"] = None

            self.heatmap_buffer = None
        else:
            self.heatmap_buffer = dict()
            self.heatmap_buffer["actor"] = list()
            self.heatmap_buffer["critic"] = list()
            for i in range(self.max_size):
                self.heatmap_buffer["actor"].append(None)  # For actor
                self.heatmap_buffer["critic"].append(None)  # For critic
            self.full_observation_buffer = None

        # TODO delete once full_observation_buffer is done
        self.obs_buf = np.zeros(
            combined_shape(self.max_size, self.observation_dimension), dtype=np.float32
        )
        self.act_buf = np.zeros(combined_shape(self.max_size), dtype=np.float32)
        self.adv_buf = np.zeros(self.max_size, dtype=np.float32)
        self.rew_buf = np.zeros(self.max_size, dtype=np.float32)
        self.ret_buf = np.zeros(self.max_size, dtype=np.float32)
        self.val_buf = np.zeros(self.max_size, dtype=np.float32)
        self.source_tar = np.zeros((self.max_size, 2), dtype=np.float32)
        self.logp_buf = np.zeros(self.max_size, dtype=np.float32)
        self.location_pred_buf = np.zeros((self.max_size, 2), dtype=np.float32)

        # TODO artifact - investigate
        self.obs_win = np.zeros(self.observation_dimension, dtype=np.float32)
        self.obs_win_std = np.zeros(self.observation_dimension, dtype=np.float32)

    def quick_reset(self):
        """Resets pointers for existing buffers and creates new epsiode lengths buffer. This avoids having to make a new buffer every time"""
        self.ptr = 0
        self.path_start_idx = 0
        self.episode_lengths_buffer = list()

    def store(
        self,
        obs: npt.NDArray[np.float32],
        act: int,
        rew: float,
        val: float,
        logp: float,
        src: npt.NDArray[np.float32],
        full_observation: Dict[int, npt.NDArray],
        heatmap_stacks: None,
        terminal: bool,
    ) -> None:
        """
        Append one timestep of agent-environment interaction to the buffer.

        :param obs: (npt.ndarray) observation (Usually the one returned from environment for previous step)
        :param act: (int) action taken
        :param rew: (float) reward from environment
        :param val: (float) state-value from critic
        :param logp: (float) log probability from actor
        :param src: (npt.ndarray) source coordinates
        :param full_observation: (dict) all agent observations
        :param terminal: (bool) episode resets next step or not
        """

        assert self.ptr < self.max_size
        self.obs_buf[self.ptr, :] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.source_tar[self.ptr] = src
        self.logp_buf[self.ptr] = logp

        if heatmap_stacks:
            if PRIO_MEMORY:
                for agent_id, agent_obs in full_observation.items():
                    self.full_observation_buffer[self.ptr][agent_id] = agent_obs
                    self.full_observation_buffer[self.ptr]["terminal"] = terminal
            else:
                self.heatmap_buffer["actor"][self.ptr] = heatmap_stacks.actor
                self.heatmap_buffer["critic"][self.ptr] = heatmap_stacks.critic

        self.ptr += 1

    def store_episode_length(self, episode_length: int) -> None:
        """
        Save episode length at the end of an episode for later calculations

        :param episode_length: (int) length of that episode, via either success or timeout. Not stored for partial episodes during epoch cutoff.
        """
        self.episode_lengths_buffer.append(episode_length)

    def GAE_advantage_and_rewardsToGO(self, last_state_value: float = 0.0) -> None:
        """
        Call this at the end of a trajectory when an episode has ended or the max steps per epoch has been reached. This looks back in the buffer to where the history/trajectory started,
        and uses rewards and value estimates from the whole trajectory to compute advantage estimates with GAE-Lambda, as well as compute the rewards-to-go for each state,
        to use as the targets for the value function. The last state value allows us to estimate the next reward and include it.
        Updates the advantage buffer and the return buffer.

        Advantage: roughly how advantageous it is to be in a particular state (see: https://arxiv.org/abs/1506.02438)
        Rewards to go: Instead of the expected return, the sum of the discounted rewards from the time t to the end of the episode.

        :param last_state_value: (float) last state value encountered. Should be 0 if the trajectory ended because the agent reached a terminal state (found source), and otherwise should
            be V(s_T), the value function estimated state-value for the last state.

        Note: Nice description of GAE choices: https://github.com/openai/spinningup/issues/349
        """

        # Choose only relevant section of buffers
        path_slice: slice = slice(self.path_start_idx, self.ptr)
        # size steps + 1. If epoch was 10 steps, this will hold 10 rewards plus the last states state_value (or 0 if terminal)
        rews: npt.NDArray[np.float64] = np.append(
            self.rew_buf[path_slice], last_state_value
        )
        vals: npt.NDArray[np.float64] = np.append(
            self.val_buf[path_slice], last_state_value
        )

        # GAE-Lambda advantage calculation. Gamma determines scale of value function, introduces bias regardless of VF accuracy (similar to discount) and
        # lambda introduces bias when VF is inaccurate
        deltas: npt.NDArray[np.float64] = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        GAE = discount_cumsum(deltas, self.gamma * self.lam)
        self.adv_buf[path_slice] = GAE

        # the next line computes rewards-to-go, to be targets for the value function
        r2g = discount_cumsum(rews, self.gamma)
        self.ret_buf[path_slice] = r2g[:-1]  # Remove last non-step element

        self.path_start_idx = self.ptr  # Update start index

    def get(self) -> Dict[str, object]:
        """
        Call this at the end of an epoch to get all of the data from buffers (except full_observation buffer). Advantages are normalized/shifted to have mean zero and std one).
        Buffer pointers and episode_lengths are reset to start over.
        """
        # Make sure buffers are full
        assert self.ptr == self.max_size

        # Get episode lengths
        episode_lengths: List[int] = self.episode_lengths_buffer
        number_episodes = len(episode_lengths)
        total_episode_length = sum(episode_lengths)

        assert (
            number_episodes > 0
        ), "0 completed episodes, this will cause NaN for advantage. Usually caused by having epochs shorter than a single episode."

        # the next two lines implement the advantage normalization trick
        # adv_mean = self.adv_buf.mean()
        # adv_std = self.adv_buf.std()
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf: npt.NDArray[np.float32] = (self.adv_buf - adv_mean) / adv_std

        # Reset pointers and episode lengths buffer
        self.quick_reset()

        # If they're equal then we don't need to do anything. Otherwise we need to add one to make sure that number_episodes is the correct size.
        # This can happen when an episode is cutoff by an epoch stop, thus meaning the number of complete episodes is short by 1.
        episode_len_Size = number_episodes + int(
            total_episode_length != len(self.obs_buf)
        )

        # Stack all tensors into one tensor
        obs_buf = np.hstack(
            (
                self.obs_buf,
                self.adv_buf[:, None],
                self.ret_buf[:, None],
                self.logp_buf[:, None],
                self.act_buf[:, None],
                self.source_tar,
            )
        )

        # Data is currently stored per-step for the entire epoch. This divides the data into a per-episode form for the model
        episode_form: List[List[torch.Tensor]] = [[] for _ in range(episode_len_Size)]

        # TODO: This is essentially just a sliding window over obs_buf; use a built-in function to do this
        slice_b: int = 0
        slice_f: int = 0
        jj: int = 0
        for ep_i in episode_lengths:
            slice_f += ep_i
            episode_form[jj].append(
                torch.as_tensor(obs_buf[slice_b:slice_f], dtype=torch.float32)
            )  # Take readings from slice_b to slice_f. Slice_f marks the episode offset
            slice_b += ep_i
            jj += 1

        # If epoch was cut off
        if slice_f != len(self.obs_buf):
            episode_form[jj].append(
                torch.as_tensor(obs_buf[slice_f:], dtype=torch.float32)
            )

        # Convert to tensors
        data = dict(
            obs=torch.as_tensor(np.copy(self.obs_buf), dtype=torch.float32),
            act=torch.as_tensor(np.copy(self.act_buf), dtype=torch.float32),
            ret=torch.as_tensor(np.copy(self.ret_buf), dtype=torch.float32),
            adv=torch.as_tensor(np.copy(self.adv_buf), dtype=torch.float32),
            logp=torch.as_tensor(np.copy(self.logp_buf), dtype=torch.float32),
            loc_pred=torch.as_tensor(
                np.copy(self.obs_win_std), dtype=torch.float32
            ),  # TODO artifact - delete? Appears to be used in the location prediction, but is never updated
            ep_len=torch.as_tensor(np.copy(total_episode_length), dtype=torch.float32),
            ep_form=episode_form,
            actor_mapstacks = self.heatmap_buffer['actor'],
            critic_mapstacks = self.heatmap_buffer['critic']
            
        )

        return data

