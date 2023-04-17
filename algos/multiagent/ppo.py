'''
    Original single-agent RAD-A2C Module.
'''

import torch
from torch.optim import Adam
import torch.nn.functional as F

import numpy as np
import numpy.typing as npt

from dataclasses import dataclass, field
from typing_extensions import TypeAlias # type: ignore
from typing import Union, cast, Optional, Any, NamedTuple, Tuple, Dict, List, Dict
import scipy.signal # type: ignore
import ray

try:
    import NeuralNetworkCores.RADTEAM_core as RADCNN_core # type: ignore  
    import NeuralNetworkCores.RADA2C_core as RADA2C_core # type: ignore
    from epoch_logger import EpochLogger # type: ignore
    from rl_tools.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads # type: ignore
    from rl_tools.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs # type: ignore     
except ModuleNotFoundError:
    import algos.multiagent.NeuralNetworkCores.RADTEAM_core as RADCNN_core # type: ignore
    import algos.multiagent.NeuralNetworkCores.RADA2C_core as RADA2C_core # type: ignore
    from algos.multiagent.epoch_logger import EpochLogger 
    from algos.multiagent.rl_tools.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads # type: ignore
    from algos.multiagent.rl_tools.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs # type: ignore      
except: 
    raise Exception

# If prioritizing memory, only keep observations and reinflate heatmaps when update happens. Reduces memory requirements, but greatly slows down training.
PRIO_MEMORY = False

Shape: TypeAlias = Union[int, Tuple[int], Tuple[int, Any], Tuple[int, int, Any]]


def combined_shape(length: int, shape: Optional[Shape] = None) -> Shape:
    '''
        This method combines dimensions. It combines length and existing shape dimension into a new tuple representing dimensions (useful for numpy.zeros() or tensor creation). 
        Length is in x position. If shape is a tuple, flatten it and add it to remaining tuple positions. Returns dimensions of new shape.
        
        Example 1 : Size (steps_per_epoch) - Make a buffer to store advantages for an epoch. Returns (x, )
        Example 2: Size (steps_per_epoch, 2) - Make a buffer for source locations (x, y) for an epoch. Returns Returns (x, 2)
        Example 3: Size (steps_per_epoch, num_agents, observation_dimensions) - Make a buffer for agent observations for an epoch. Returns (x, n, 11)
                
        :param length: (int) X position of tuple.
        :param shape: (int | Tuple[int, Any]) remaining positions in tuple.
    '''
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


def discount_cumsum(x: npt.NDArray[np.float64], discount: float) -> npt.NDArray[np.float64]:
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


# NOTE: Obsolete - use discount cumsum instead. Used for verification purposes
def generalized_advantage_estimate(gamma, lamb, done, rewards, values):
    """
    gamma: trajectory discount (scalar)
    lamda: exponential mean discount (scalar)
    values: value function results for each step
    rewards: rewards for each step
    done: flag for end of episode (ensures advantage only calculated for single epsiode, when multiple episodes are present)
    
    Thank you to https://nn.labml.ai/rl/ppo/gae.html
    """
    batch_size = done.shape[0]

    advantages = np.zeros(batch_size + 1)
    
    last_advantage = 0
    last_value = values[-1]

    for t in reversed(range(batch_size)):
        # Make mask to filter out values by episode
        mask = 1.0 - done[t] # convert bools into variable to multiply by
        
        # Apply terminal mask to values and advantages 
        last_value = last_value * mask
        last_advantage = last_advantage * mask
        
        # Calculate deltas
        delta = rewards[t] + gamma * last_value - values[t]

        # Get last advantage and add to proper element in advantages array
        last_advantage = delta + gamma * lamb * last_advantage                
        advantages[t] = last_advantage
        
        # Get new last value
        last_value = values[t]
        
    return advantages


# NOTE: Obsolete - use discount cumsum instead. Used for verification purposes
def rewards_to_go(batch_rews, gamma):
    ''' 
    Calculate the rewards to go. Gamma is the discount factor.
    Thank you to https://medium.com/swlh/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a
    '''
    # The rewards-to-go (rtg) per episode per batch to return and the shape will be (num timesteps per episode).
    batch_rtgs = [] 
    
    # Iterate through each episode backwards to maintain same order in batch_rtgs
    discounted_reward = 0 # The discounted reward so far
    
    for rew in reversed(batch_rews):
        discounted_reward = rew + discounted_reward * gamma
        batch_rtgs.insert(0, discounted_reward)
            
    return batch_rtgs     


class UpdateResult(NamedTuple):
    ''' Object that contains the return values from the neural network updates '''
    stop_iteration: int
    loss_policy: float
    loss_critic: Union[float, None]
    loss_predictor: float
    kl_divergence: npt.NDArray[np.float32]
    Entropy: npt.NDArray[np.float32]
    ClipFrac: npt.NDArray[np.float32]
    LocLoss: Union[torch.Tensor, None]
    VarExplain: int #TODO what is this?


class BpArgs(NamedTuple):
    ''' Object that contains the parameters for bootstrap particle filter '''
    bp_decay: float
    l2_weight: float
    l1_weight: float
    elbo_weight: float
    area_scale: float


@dataclass
class OptimizationStorage:
    '''     
        Class that stores information related to updating neural network models for each agent. It includes the clip ratio for 
        ensuring a destructively large policy update doesn't happen, an entropy parameter for randomness/entropy, and the target kl divergence 
        for early stopping.
            
        :param train_pi_iters: (int) Maximum number of gradient descent steps to take on actor policy loss per epoch. (Early stopping may cause 
            optimizer to take fewer than this.)
        :param train_v_iters: (int) Number of gradient descent steps to take on critic state-value function per epoch.
        :param train_pfgru_iters: (int) Number of gradient descent steps to take for source localization neural network (the PFGRU unit)
        :param {*}_optimizer: (torch.optim) Pytorch Optimizer with learning rate decay [Torch]
        :param clip_ratio: (float) Hyperparameter for clipping in the policy objective. Roughly: how far can the new policy go from the old policy 
            while still profiting (improving the objective function)? The new policy can still go farther than the clip_ratio says, but it doesn't
            help on the objective anymore. This is usually small, often 0.1 to 0.3, and is typically denoted by :math:`\epsilon`. Basically if the 
            policy wants to perform too large an update, it goes with a clipped value instead.
        :param alpha: (float) Entropy reward term scaling used during calculating loss. 
        :param target_kl: (float) Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used 
            for early stopping. It's usually small, 0.01 or 0.05.
    '''
    train_pi_iters: int
    train_v_iters: Union[int, None]
    train_pfgru_iters: int    
    pi_optimizer: torch.optim.Optimizer
    critic_optimizer: Union[torch.optim.Optimizer, None]
    model_optimizer: torch.optim.Optimizer
    clip_ratio: float
    alpha: float
    target_kl: float
    
    # Initialized elsewhere
    #: Schedules gradient steps for actor
    pi_scheduler: torch.optim.lr_scheduler.StepLR = field(init=False)
    #: Schedules gradient steps for value function (critic)
    critic_scheduler: Union[torch.optim.lr_scheduler.StepLR, None] = field(init=False)
    #: Schedules gradient steps for PFGRU location predictor module    
    pfgru_scheduler: torch.optim.lr_scheduler.StepLR = field(init=False)   
    #: Loss calculator utility for Critic   
    MSELoss: torch.nn.modules.loss.MSELoss = field(default_factory= (lambda: torch.nn.MSELoss(reduction="mean"))) 

    def __post_init__(self):        
        self.pi_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pi_optimizer, step_size=100, gamma=0.99
        )     
        self.pfgru_scheduler = torch.optim.lr_scheduler.StepLR(
            self.model_optimizer, step_size=100, gamma=0.99
        )        
        
        if self.train_v_iters and self.critic_optimizer:
            self.critic_scheduler = torch.optim.lr_scheduler.StepLR(
                self.critic_optimizer, step_size=100, gamma=0.99
            )
        else:
            self.critic_scheduler = None # RAD-A2C has critic embeded in pi


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

    ptr: int = field(init=False)  # For keeping track of location in buffer during update
    path_start_idx: int = field(init=False)  # For keeping track of starting location in buffer during update

    episode_lengths_buffer: List = field(init=False)  # Stores episode lengths
    full_observation_buffer: List[Dict[Union[int, str], Union[npt.NDArray[np.float32], bool, None]]] = field(init=False) # In memory-priority mode, for each timestep, stores every agents observation
    heatmap_buffer: Dict[str, List[torch.Tensor]] = field(init=False) # When memory is not a concern, for each timestep, stores every steps heatmap stack for both actor and critic
    
    obs_buf: npt.NDArray[np.float32] = field(init=False)  # Observation buffer for each agent
    act_buf: npt.NDArray[np.float32] = field(init=False)  # Action buffer for each step. Note: each agent carries their own PPO buffer, no need to track all agent actions.
    adv_buf: npt.NDArray[np.float32] = field(init=False)  # Advantages buffer for each step
    rew_buf: npt.NDArray[np.float32] = field(init=False)  # Rewards buffer for each step
    ret_buf: npt.NDArray[np.float32] = field(init=False)  # Rewards-to-go buffer (Rewards gained from timestep t until terminal state (similar to expected return, but actual))
    val_buf: npt.NDArray[np.float32] = field(init=False)  # State-value buffer for each step
    source_tar: npt.NDArray[np.float32] = field(init=False) # Source location buffer (for moving targets)
    logp_buf: npt.NDArray[np.float32] = field(init=False)  # action log probabilities buffer
        
    obs_win: npt.NDArray[np.float32] = field(init=False) # For location prediction TODO find out what its doing
    obs_win_std: npt.NDArray[np.float32] = field(init=False) # For location prediction TODO find out what its doing

    gamma: float = 0.99 # trajectory discount for Generalize Advantage Estimate (GAE) 
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
                    self.full_observation_buffer[i][id] = np.zeros((self.observation_dimension,))
                    self.full_observation_buffer[i]['terminal'] = None
            
            self.heatmap_buffer = None
        else:
            self.heatmap_buffer = dict()
            self.heatmap_buffer['actor'] = list()
            self.heatmap_buffer['critic'] = list()
            for i in range(self.max_size):
                self.heatmap_buffer['actor'].append(None) # For actor
                self.heatmap_buffer['critic'].append(None)  # For critic
            self.full_observation_buffer = None            

        # TODO delete once full_observation_buffer is done
        self.obs_buf= np.zeros(
            combined_shape(self.max_size, self.observation_dimension), dtype=np.float32
        )
        self.act_buf = np.zeros(
            combined_shape(self.max_size), dtype=np.float32
        )
        self.adv_buf = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.rew_buf = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.ret_buf = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.val_buf = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.source_tar = np.zeros(
            (self.max_size, 2), dtype=np.float32
        )
        self.logp_buf = np.zeros(
            self.max_size, dtype=np.float32
        )

        # TODO artifact - delete? Appears to be used in the location prediction, but is never updated        
        self.obs_win = np.zeros(self.observation_dimension, dtype=np.float32)
        self.obs_win_std = np.zeros(self.observation_dimension, dtype=np.float32)

    def quick_reset(self):
        """ Resets pointers for existing buffers and creates new epsiode lengths buffer. This avoids having to make a new buffer every time """
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
        heatmap_stacks: RADCNN_core.HeatMaps,
        terminal: bool
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
                    self.full_observation_buffer[self.ptr]['terminal'] = terminal
            else:
                self.heatmap_buffer['actor'][self.ptr] = heatmap_stacks.actor
                self.heatmap_buffer['critic'][self.ptr] = heatmap_stacks.critic
        
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
        rews: npt.NDArray[np.float64] = np.append(self.rew_buf[path_slice], last_state_value) # size steps + 1. If epoch was 10 steps, this will hold 10 rewards plus the last states state_value (or 0 if terminal)
        vals: npt.NDArray[np.float64] = np.append(self.val_buf[path_slice], last_state_value) # size steps + 1. If epoch was 10 steps, this will hold 10 values plus the last states state_value (or 0 if terminal)
        
        # GAE-Lambda advantage calculation. Gamma determines scale of value function, introduces bias regardless of VF accuracy (similar to discount) and
        # lambda introduces bias when VF is inaccurate
        deltas: npt.NDArray[np.float64] = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        GAE = discount_cumsum(deltas, self.gamma * self.lam)
        self.adv_buf[path_slice] = GAE

        # the next line computes rewards-to-go, to be targets for the value function
        r2g = discount_cumsum(rews, self.gamma)
        self.ret_buf[path_slice] = r2g[:-1] # Remove last non-step element

    def get(self) -> Dict[str, object]:
        """
        Call this at the end of an epoch to get all of the data from buffers. Advantages are normalized/shifted to have mean zero and std one). 
        Buffer pointers and episode_lengths are reset to start over. NOTE: full observations are not stored here, they need to be converted to mapstacks.
        """
        # Make sure buffers are full
        assert self.ptr == self.max_size 
                
        # Get episode lengths
        episode_lengths: List[int] = self.episode_lengths_buffer
        number_episodes = len(episode_lengths)
        total_episode_length = sum(episode_lengths)
        
        assert number_episodes > 0, "0 completed episodes. Usually caused by having epochs shorter than an episode"
        
        # the next two lines implement the advantage normalization trick
        # adv_mean = self.adv_buf.mean()
        # adv_std = self.adv_buf.std()
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)        
        self.adv_buf: npt.NDArray[np.float32] = (self.adv_buf - adv_mean) / adv_std
        
        # Reset pointers and episode lengths buffer
        self.quick_reset()                

        # If they're equal then we don't need to do anything. Otherwise we need to add one to make sure that number_episodes is the correct size.
        # This can happen when an episode is cutoff by an epoch stop, thus meaning the number of complete episodes is short by 1.
        episode_len_Size = (
            number_episodes
            + int(total_episode_length != len(self.obs_buf))
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
        
        # Save in a giant tensor
        episode_form: List[List[torch.Tensor]] = [[] for _ in range(episode_len_Size)] 
        
        # TODO: This is essentially just a sliding window over obs_buf; use a built-in function to do this        
        slice_b: int = 0
        slice_f: int = 0
        jj: int = 0
        for ep_i in episode_lengths:
            slice_f += ep_i
            episode_form[jj].append(
                torch.as_tensor(obs_buf[slice_b:slice_f], dtype=torch.float32)
            )
            slice_b += ep_i
            jj += 1
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
            loc_pred=torch.as_tensor(np.copy(self.obs_win_std), dtype=torch.float32), # TODO artifact - delete? Appears to be used in the location prediction, but is never updated
            ep_len=torch.as_tensor(np.copy(total_episode_length), dtype=torch.float32),
            ep_form = episode_form
        )               

        return data


@dataclass 
class AgentPPO:
    '''
        This class handles PPO-related functions/calculations, the experience buffer, and holds the agent object. This class also is responsible for calculating advantages after an epoch,
        calling actor/critic update functions located in the agent's individual neural networks, and handling all optimizers and learning cut-offs. Minibatches are sampled
        here. One AgentPPO object per agent. Future work: add functionality for sharing single policy/critic and copying network to new "agents" (see OpenAI 5 by Berner et al. and
        Target Localization by Alagha et al.)
        
        
        :param id: (int) unique identifier for agent that is used as key for identification of correct observations, rewards, etc within shared objects.
        
        :param observation_space: (int) The dimensions of the observation returned from the environment. Also known as state dimensions. For rad-search this will be 11, for the 11 elements of 
            the observation array. This is used for the PPO buffer.
            
        :param bp_args: (BpArgs) Set up bootstrap particle filter args for the PFGRU, from Particle Filter Recurrent Neural Networks by Ma et al. 2020.
        
        :param steps_per_epoch: (int) Number of steps of interaction (state-action pairs) for the agent and the environment in each epoch before updating the neural network modules.
        
        :param env_height: (float) Max y axis bound of search area from environment grid. Note that this is only in spawnable coordinates, so likely is shorter than the full grid.
                
        :param actor_critic_args: (dict) Arguments for A2C neural networks for agent.
        
        :param actor_critic_architecture: (string) Short-version indication for what neural network core to use for actor-critic agent.
                
        :param minibatch: (int) How many observations to sample out of a batch. Used to reduce the impact of fully online learning.
        
        :param pi_learning_rate: (float) Learning rate for Actor/policy optimizer.
        
        :param critic_learning_rate: (float) Learning rate for Critic (value) function optimizer.
        
        :param pfgru_learning_rate: (float) Learning rate for the source prediction module (PFGRU).
        
        :param train_pi_iters: (int) Maximum number of gradient descent steps to take on actor policy loss per epoch (Early stopping may cause optimizer to take fewer than this).
        
        :param train_v_iters: (int) Number of gradient descent steps to take on critic state-value function per epoch.
        
        :param train_pfgru_iters: (int) Number of gradient descent steps to take for source localization neural network (the PFGRU unit).
        
        :param reduce_pfgru_iters: (bool) Reduces PFGRU training iteration when further along to speed up training.
        
        :param GlobalCriticOptimizer: (torch.optim.Optimizer) Optimizer for global critic. Defaults to none.
        
        :param actor_learning_rate: (float) For actor/policy. When updating neural networks, indicates how large of a learning step to take. Larger means a bigger update, and vise versa. This should be
            reduced as the agent's learning progresses.
            
        :param critic_learning_rate: (float) For critic/value function. When updating neural networks, indicates how large of a learning step to take. Larger means a bigger update, and vise versa. This should be
            reduced as the agent's learning progresses.
            
        :param pfgru_learning_rate: (float) For the PFGRU/location predictor module. When updating neural networks, indicates how large of a learning step to take. Larger means a bigger update, and vise versa. This should be
            reduced as the agent's learning progresses.    
                        
        :param alpha: (float) Entropy reward term scaling.
        
        :param clip_ratio: (float) Usually seen as Epsilon Hyperparameter for clipping in the policy objective. Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy can still go farther than the clip_ratio says, but it doesn't help on the objective anymore. 
            (Usually small, 0.1 to 0.3.).Basically if the policy wants to perform too large an update, it goes with a clipped value instead.
            
        :param target_kl: (float) Roughly what KL divergence we think is appropriate between new and old policies after an update; This will get used for early stopping (Usually small, 0.01 or 0.05).  

        :param gamma: (float) Discount rate for expected return and Generalize Advantage Estimate (GAE) calculations (Always between 0 and 1).
         
        :param lam: (float) Exponential weight decay/discount; controls the bias variance trade-off for Generalize Advantage Estimate (GAE) calculations (Always between 0 and 1, close to 1).
        
        :return: None
        
        test test
    '''
    
    id: int
    observation_space: int
    bp_args: BpArgs     # No default due to need for environment height parameter.
    steps_per_epoch: int  # No default value - Critical that it match environment
    steps_per_episode: int 
    number_of_agents: int # Number of agents
    env_height: float
    actor_critic_args: Dict[str, Any]
    actor_critic_architecture: str = field(default="cnn")    
    minibatch: int = field(default=1)    
    train_pi_iters: int = field(default= 40)
    train_v_iters: int = field(default= 40)
    train_pfgru_iters: int = field(default= 15)
    reduce_pfgru_iters: bool = field(default=True)
    GlobalCriticOptimizer: Union[torch.optim.Optimizer, None] = field(default=None)
    actor_learning_rate: float = field(default= 3e-4)
    critic_learning_rate: float = field(default= 1e-3)
    pfgru_learning_rate: float = field(default= 5e-3)
    gamma: float = field(default= 0.99)
    alpha: float = field(default= 0)    
    clip_ratio: float = field(default= 0.2)
    target_kl: float = field(default= 0.07)
    lam: float = field(default= 0.9)
    
    # Initialized elsewhere
    agent: Union[RADCNN_core.CNNBase, RADA2C_core.RNNModelActorCritic] = field(init=False)

    def __post_init__(self):
        ''' Initialize Agent's neural network architecture'''
        
        ################################## set device ##################################
        print("============================================================================================")
        # set device to cpu or cuda
        device = torch.device('cpu')
        if(torch.cuda.is_available()): 
            device = torch.device('cuda:0') 
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set to : cpu")
        print("============================================================================================")        
                  
        # Simple Feed Forward Network
        if self.actor_critic_architecture == 'cnn':
            self.agent = RADCNN_core.CNNBase(id=self.id, **self.actor_critic_args)             

            if not self.GlobalCriticOptimizer:
                CriticOptimizer = Adam(self.agent.critic.parameters(), lr=self.critic_learning_rate)
            else:
                CriticOptimizer = self.GlobalCriticOptimizer
            
            # Initialize learning opitmizers                           
            self.agent_optimizer = OptimizationStorage(
                train_pi_iters = self.train_pi_iters,                
                train_v_iters = self.train_v_iters,
                train_pfgru_iters = self.train_pfgru_iters,              
                pi_optimizer = Adam(self.agent.pi.parameters(), lr=self.actor_learning_rate),
                critic_optimizer = CriticOptimizer,  # Allows for global optimizer
                model_optimizer = Adam(self.agent.model.parameters(), lr=self.pfgru_learning_rate),
                MSELoss = torch.nn.MSELoss(reduction="mean"),
                clip_ratio = self.clip_ratio,
                alpha = self.alpha,
                target_kl = self.target_kl,             
                )                         
        # Gated recurrent architecture for RAD-A2C 
        elif self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
            # Initialize Agents                
            self.agent = RADA2C_core.RNNModelActorCritic(**self.actor_critic_args)
            
            if self.GlobalCriticOptimizer:
                raise Exception("No global critic option for RAD-A2C")        
            
            # Initialize learning opitmizers                           
            self.agent_optimizer = OptimizationStorage(
                train_pi_iters = self.train_pi_iters,                
                train_v_iters = None, # Critic is embeded in policy for RAD-A2C
                train_pfgru_iters = self.train_pfgru_iters,              
                pi_optimizer = Adam(self.agent.pi.parameters(), lr=self.actor_learning_rate),
                critic_optimizer = None, # Critic is embeded in policy for RAD-A2C
                model_optimizer = Adam(self.agent.model.parameters(), lr=self.pfgru_learning_rate),
                MSELoss = torch.nn.MSELoss(reduction="mean"),
                clip_ratio = self.clip_ratio,
                alpha = self.alpha,
                target_kl = self.target_kl,             
                )                
        else:
            raise ValueError('Unsupported Neural Network type requested')
            
        # Inititalize buffer
        if self.steps_per_epoch > 0:
            self.ppo_buffer = PPOBuffer(observation_dimension=self.observation_space, max_size=self.steps_per_epoch, max_episode_length=self.steps_per_episode, gamma=self.gamma, lam=self.lam, number_agents=self.number_of_agents)
        else:
            raise ValueError("Steps per epoch cannot be 0")
        
    def reduce_pfgru_training(self):
        ''' Reduce localization module training iterations after some number of epochs to speed up training '''
        if self.reduce_pfgru_iters:
            self.train_pfgru_iters = 5
            self.reduce_pfgru_iters = False     
    
    def step(self, observations: Dict[int, List[Any]], hiddens: Union[None, List[torch.Tensor]] = None, message: Union[None, Dict] =None) -> RADCNN_core.ActionChoice:
        ''' 
        Wrapper for neural network action selection 
        
        :param observations: (Dict) Observations from all agents.
        :param hiddens: (Dict) Hidden layer values for each agent. Only compatible with RAD-A2C.
        :param message: (Dict) Information from the episode.
        
        '''
        # RAD-A2C compatibility
        if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
            #a, v, logp, hidden, out_pred = self.agent.step(obs=observations[self.id], hidden=hiddens) # type: ignore
            results, heatmaps = self.agent.step(obs=observations[self.id], hidden=hiddens)
            # results = RADCNN_core.ActionChoice(
            #     id= self.id,
            #     action= a,
            #     action_logprob= logp,
            #     state_value= v, 
            #     loc_pred= out_pred,
            #     hiddens= hidden
            #     )
        else:
            results, heatmaps = self.agent.select_action(observations, self.id)  # TODO add in hidden layer shenanagins for PFGRU use
        return results, heatmaps         
    
    def reset_agent(self)-> None:
        ''' 
        Reset the neural networks at the end of an episode or training batch update. For RAD-TEAM, this does not reset the network parameters, it just flushes
        the heatmaps and resets all standardization/normalization tools.
        '''
        self.agent.reset()
        
    def reset_hidden(self)-> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
            return self.agent.reset_hidden() # type: ignore        
        else:
            raise ValueError("Attempting to reset hidden layers on non-RAD-A2C architecture!")
        
    def update_agent(self, logger: EpochLogger = None) -> UpdateResult: #         (env, bp_args, loss_fcn=loss)
        """
        Wrapper function to update individual neural networks. Note: update functions perform multiple updates per call
        
        :param logger: (EpochLogger) Logger used for RAD-A2C updates.
        """   
        
        def sample(self, data, minibatch=None):
            ''' Get sample indexes of episodes to train on'''
            if not minibatch:
                minibatch = self.minibatch
            # Randomize and sample observation batch indexes
            ep_length = data["ep_len"].item()
            indexes = np.arange(0, ep_length, dtype=np.int32)
            number_of_samples = int((ep_length / minibatch))
            return np.random.choice(indexes, size=number_of_samples, replace=False) # Uniform                    
         
        # Get data from buffers. NOTE: this does not get heatmap stacks/full observations.
        data: Dict[str, torch.Tensor] = self.ppo_buffer.get() 
        min_iterations = len(data["ep_form"])
        kk = 0
        term = False        

        # Train RAD-A2C framework
        if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
            # Update function for the PFGRU localization module. Module will be set to train mode, then eval mode within update_model        
            model_loss = self.update_model(data)
                        
            # Reset gradients 
            self.agent_optimizer.pi_optimizer.zero_grad()
            # Train Actor-Critic policy with multiple steps of gradient descent. train_pi_iters == k_epochs
            while not term and kk < self.train_pi_iters:
                # Early stop training if KL divergence above certain threshold
                update_results: Dict[str, Union[torch.Tensor, npt.NDArray[Any], List[Any], bool]] = {}
                (
                    update_results['pi_l'], 
                    update_results['pi_info'], 
                    update_results['term'],  
                    update_results['loc_loss']
                ) = self.update_rada2c(data, min_iterations, logger=logger)  # type: ignore
                kk += 1
                
            # Reduce learning rate
            self.agent_optimizer.pi_scheduler.step()
            self.agent_optimizer.pfgru_scheduler.step()

            # Log changes from update
            return UpdateResult(
                stop_iteration=kk,
                loss_policy=update_results['pi_l'].item(), # type: ignore
                loss_critic=update_results['pi_info']["val_loss"].item(), # type: ignore
                loss_predictor=model_loss.item(),  # TODO if using the regression GRU
                kl_divergence=update_results['pi_info']["kl"], # type: ignore
                Entropy=update_results['pi_info']["ent"], # type: ignore
                ClipFrac=update_results['pi_info']["cf"], # type: ignore
                LocLoss=update_results['loc_loss'], # type: ignore
                VarExplain=0 
            )
        # Train RAD-TEAM framework                 
        else:
            # TODO get PFGRU working with RAD-TEAM
            model_loss = torch.tensor(0)
            
            # Put agents in train mode
            self.agent.set_mode(mode='train')
            
            # Get mapstacks from buffer or inflate from logs, if in max-memory mode
            if PRIO_MEMORY:
                actor_maps_buffer, critic_maps_buffer = self.generate_mapstacks()
            else:
                actor_maps_buffer = self.ppo_buffer.heatmap_buffer['actor']
                critic_maps_buffer = self.ppo_buffer.heatmap_buffer['critic']
                
            # Train Actor policy with multiple steps of gradient descent. train_pi_iters == k_epochs
            for k_epoch in range(self.train_pi_iters):
                
                # Reset gradients 
                self.agent_optimizer.pi_optimizer.zero_grad()
                
                # Get indexes of episodes that will be sampled
                sample_indexes = sample(self, data=data)
                
                #actor_loss_results = self.compute_batched_losses_pi(data=data, map_buffer_maps=map_buffer_maps, sample=sample_indexes)
                actor_loss_results = self.compute_batched_losses_pi(data=data, sample=sample_indexes, mapstacks_buffer=actor_maps_buffer)
                
                # Check Actor KL Divergence
                if actor_loss_results['kl'].item() < 1.5 * self.target_kl:
                    actor_loss_results['pi_loss'].backward()
                    
                    mpi_avg_grads(self.agent.pi) # Average gradients across processes

                    self.agent_optimizer.pi_optimizer.step() 
                else:
                    break  # Skip remaining training               

            # Reduce learning rate
            self.agent_optimizer.pi_scheduler.step()

            # TODO Uncomment after implementing PFGRU
            #self.agent_optimizer.pfgru_scheduler.step() 
            # Reduce pfgru learning rate

            results: UpdateResult

            # If local critic, do Value function learning here
            # For global critic, only first agent performs the update
            if not self.GlobalCriticOptimizer or self.id == 0:       
                for _ in range(self.train_v_iters):
                    self.agent_optimizer.critic_optimizer.zero_grad()
                    critic_loss_results = self.compute_batched_losses_critic(data=data, sample=sample_indexes, map_buffer_maps=critic_maps_buffer)
                    critic_loss_results['critic_loss'].backward()
                    mpi_avg_grads(self.agent.critic) # Average gradients across processes                    
                    self.agent_optimizer.critic_optimizer.step()
                
                # Reduce learning rate
                self.agent_optimizer.critic_scheduler.step()  
                            
                results = UpdateResult(
                    stop_iteration=k_epoch,  
                    loss_policy=actor_loss_results['pi_loss'].item(),
                    loss_critic=critic_loss_results['critic_loss'].item(),
                    loss_predictor=model_loss.item(),  # TODO implement when PFGRU is working for CNN
                    kl_divergence=actor_loss_results["kl"],
                    Entropy=actor_loss_results["entropy"],
                    ClipFrac=actor_loss_results["clip_fraction"],
                    LocLoss= torch.tensor(0), # TODO implement when PFGRU is working for CNN
                    VarExplain=0 
                )                          
            else:
                results = UpdateResult(
                    stop_iteration=k_epoch,  
                    loss_policy=actor_loss_results['pi_loss'].item(),
                    loss_critic=None,
                    loss_predictor=model_loss.item(),  # TODO implement when PFGRU is working for CNN
                    kl_divergence=actor_loss_results["kl"],
                    Entropy=actor_loss_results["entropy"],
                    ClipFrac=actor_loss_results["clip_fraction"],
                    LocLoss= torch.tensor(0), # TODO implement when PFGRU is working for CNN
                    VarExplain=0 
                )                          

            # Take agents out of train mode
            self.agent.set_mode(mode='eval')                           
            
            # Log changes from update
            return results
                    
    def compute_batched_losses_pi(self, sample, data, mapstacks_buffer, minibatch = None):
        ''' Simulates batched processing through CNN. Wrapper for computing single-batch loss for pi'''
        
        # TODO make more concise 
        # Due to linear layer in CNN, this must be run individually
        pi_loss_list = []
        kl_list = []
        entropy_list = []
        clip_fraction_list = []
        
        # Get sampled returns from actor and critic
        for index in sample:
            # Reset existing episode maps
            self.reset_agent()     
            single_pi_l, single_pi_info = self.compute_loss_pi(data=data, index=index, map_stack=mapstacks_buffer[index])
            
            pi_loss_list.append(single_pi_l)
            kl_list.append(single_pi_info['kl'])
            entropy_list.append(single_pi_info['entropy'])
            clip_fraction_list.append(single_pi_info['clip_fraction'])
            
        #take mean of everything for batch update
        results = {
            'pi_loss': torch.stack(pi_loss_list).mean(),
            'kl': np.mean(kl_list),
            'entropy': np.mean(entropy_list),
            'clip_fraction': np.mean(clip_fraction_list),
        }
        return results

    def compute_loss_pi(self, data: Dict[str, Union[torch.Tensor, List]], index: int, map_stack: List[torch.Tensor]):
        ''' 
            Compute loss for actor network. Loss is the difference between the probability of taking the action according to the current policy
            and the probability of taking the action according to the old policy, multiplied by the advantage of the action.
            
            Process:
                #. Calculate how much the policy has changed:  ratio = policy_new / policy_old
                #. Take log form of this:  ratio = [log(policy_new) - log(policy_old)].exp()
                #. Calculate Actor loss as the minimum of two functions: 
                    #. p1 = ratio * advantage
                    #. p2 = clip(ratio, 1-epsilon, 1+epsilon) * advantage
                    #. actor_loss = min(p1, p2)               
            
            :param data: (array) data from PPO buffer. Contains:
                * obs: (tensor) Unused: batch of observations from the PPO buffer. Currently only used to ensure map buffer observations are correct.
                * act: (tensor) batch of actions taken.
                * adv: (tensor) batch of advantages cooresponding to actions. These are the difference between the expected reward for taking that action and the true reward (See: TD Error, GAE).
                * logp: (tensor) batch of action logprobabilities.
                * loc_pred: (tensor) batch of predicted location by PFGRU.
                * ep_len: (tensor[int]) single dimension int of length of episode.
                * ep_form: (tensor) Episode form 
            :param index: (int) If doing a single observation at a time, index for data[]
        '''
        # NOTE: Not using observation tensor, using internal map buffer
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        # Get action probabilities and entropy for an state's mapstack and action, then put the action probabilities on the CPU (if on the GPU)
        if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
            action_logprobs, dist_entropy = self.agent.pi.evaluate(map_stack, act[index])  
        else:
            action_logprobs, dist_entropy = self.agent.pi.get_action_information(map_stack, act[index])              
        action_logprobs = action_logprobs.cpu() 
        
        # Get how much change is about to be made, then clip it if it exceeds our threshold (PPO-CLIP)
        # NOTE: Loss will be averaged in the wrapper function, not here, as this is for a single observation/mapstack
        ratio = torch.exp(action_logprobs - logp_old[index])
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv[index]  # Objective surrogate
        loss_pi = -(torch.min(ratio * adv[index], clip_adv))

        # Useful extra info
        approx_kl = (logp_old[index] - action_logprobs).item()
        ent = dist_entropy.item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).item()
        pi_info = dict(kl=approx_kl, entropy=ent, clip_fraction=clipfrac)

        return loss_pi, pi_info  

    def compute_batched_losses_critic(self, data, map_buffer_maps, sample):
        ''' Simulates batched processing through CNN. Wrapper for single-batch computing critic loss'''    
        
        # TODO make more concise 
        # Due to linear layer in CNN, this must be run fully online (read: every map)
        critic_loss_list = []
        
        # Get sampled returns from actor and critic
        for index in sample:
            # Reset existing episode maps
            self.reset_agent()                           
            critic_loss_list.append(self.compute_loss_critic(data=data, map_stack=map_buffer_maps[index], index=index))

        #take mean of everything for batch update
        results = {'critic_loss': torch.stack(critic_loss_list).mean()}
        return results
            
    def compute_loss_critic(self,  index: int, data: Dict[str, Union[torch.Tensor, List]], map_stack: torch.Tensor):
        ''' Compute loss for state-value approximator (critic network) using MSE. Calculates the MSE of the 
            predicted state value from the critic and the true state value
        
            data (array): data from PPO buffer
                ret (tensor): batch of returns
                
            map_stack (tensor): Either a single observations worth of maps, or a batch of maps
            index (int): If doing a single observation at a time, index for data[]
            
            Adapted from https://github.com/nikhilbarhate99/PPO-PyTorch
            
            Calculate critic loss with MSE between returns and critic value
                critic_loss = (R - V(s))^2            
        '''    
        true_return = data['ret'][index]
        
        # Compare predicted return with true return and use MSE to indicate loss
        predicted_value = self.agent.critic.forward(map_stack)
        critic_loss = self.agent.mseLoss(torch.squeeze(predicted_value), true_return)
        return critic_loss

    def update_model(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        ''' Update a single agent's PFGRU location prediction module (see Ma et al. 2020 for more details) '''      
        # Initial values and compatability
        args: BpArgs = self.bp_args
        ep_form = data["ep_form"]
        source_loc_idx = 15 # src_tar is location estimate
        o_idx = 3
        
        # Put into training mode        
        self.agent.model.train() # PFGRU 
        
        for _ in range(self.train_pfgru_iters):
            model_loss_arr: torch.Tensor = torch.autograd.Variable(
                torch.tensor([], dtype=torch.float32)
            )
            for ep in ep_form:
                sl = len(ep[0])
                hidden = self.reset_hidden()[0] # type: ignore
                #src_tar: npt.NDArray[np.float32] = ep[0][:, source_loc_idx:].clone()
                src_tar: torch.Tensor = ep[0][:, source_loc_idx:].clone()
                src_tar[:, :2] = src_tar[:, :2] / args.area_scale
                obs_t = torch.as_tensor(ep[0][:, :o_idx], dtype=torch.float32)
                loc_pred = torch.empty_like(src_tar) # src_tar is location estimate
                particle_pred = torch.empty(
                    (sl, self.agent.model.num_particles, src_tar.shape[1]) 
                )

                bpdecay_params = np.exp(args.bp_decay * np.arange(sl))
                bpdecay_params = bpdecay_params / np.sum(bpdecay_params)
                for zz, meas in enumerate(obs_t):
                    loc, hidden = self.agent.model(meas, hidden) 
                    particle_pred[zz] = self.agent.model.hid_obs(hidden[0]) 
                    loc_pred[zz, :] = loc

                bpdecay_params = torch.FloatTensor(bpdecay_params)
                bpdecay_params = bpdecay_params.unsqueeze(-1)
                l2_pred_loss = (
                    F.mse_loss(loc_pred.squeeze(), src_tar.squeeze(), reduction="none")
                    * bpdecay_params
                )
                l1_pred_loss = (
                    F.l1_loss(loc_pred.squeeze(), src_tar.squeeze(), reduction="none")
                    * bpdecay_params
                )

                l2_loss = torch.sum(l2_pred_loss)
                l1_loss = 10 * torch.mean(l1_pred_loss)

                pred_loss = args.l2_weight * l2_loss + args.l1_weight * l1_loss

                particle_pred = particle_pred.transpose(0, 1).contiguous()

                particle_gt = src_tar.repeat(self.agent.model.num_particles, 1, 1) 
                l2_particle_loss = (
                    F.mse_loss(particle_pred, particle_gt, reduction="none")
                    * bpdecay_params
                )
                l1_particle_loss = (
                    F.l1_loss(particle_pred, particle_gt, reduction="none")
                    * bpdecay_params
                )

                # p(y_t| \tau_{1:t}, x_{1:t}, \theta) is assumed to be a Gaussian with variance = 1.
                # other more complicated distributions could be used to improve the performance
                y_prob_l2 = torch.exp(-l2_particle_loss).view(
                    self.agent.model.num_particles, -1, sl, 2 
                )
                l2_particle_loss = -y_prob_l2.mean(dim=0).log()

                y_prob_l1 = torch.exp(-l1_particle_loss).view(
                    self.agent.model.num_particles, -1, sl, 2 
                )
                l1_particle_loss = -y_prob_l1.mean(dim=0).log()

                xy_l2_particle_loss = torch.mean(l2_particle_loss)
                l2_particle_loss = xy_l2_particle_loss

                xy_l1_particle_loss = torch.mean(l1_particle_loss)
                l1_particle_loss = 10 * xy_l1_particle_loss

                belief_loss: torch.Tensor = (
                    args.l2_weight * l2_particle_loss
                    + args.l1_weight * l1_particle_loss
                )
                total_loss: torch.Tensor = pred_loss + args.elbo_weight * belief_loss

                model_loss_arr = torch.hstack((model_loss_arr, total_loss.unsqueeze(0)))

            model_loss: torch.Tensor = model_loss_arr.mean()
            self.agent_optimizer.model_optimizer.zero_grad()
            model_loss.backward()
            # Clip gradient TODO should 5 be a variable?
            torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), 5) 

            mpi_avg_grads(self.agent.model) #Average gradients across the processes     # MPI

            self.agent_optimizer.model_optimizer.step()

        self.agent.model.eval() 
        return model_loss

    def update_rada2c(
            self, data: Dict[str, torch.Tensor], min_iterations: int,  logger: EpochLogger, minibatch: Union[int, None] = None
        ) -> Tuple[torch.Tensor, Dict[str, Union[npt.NDArray, list]], bool, torch.Tensor]:
        ''' RAD-A2C Actor and Critic updates'''
        # Start update
        if not minibatch:
            minibatch = self.minibatch
        
        # Set initial variables
        # TODO make a named tuple and pass these that way instead of hardcoded indexes
        observation_idx = 11
        action_idx = 14
        logp_old_idx = 13
        advantage_idx = 11
        return_idx = 12
        source_loc_idx = 15

        ep_form: List[torch.tensor] = data["ep_form"] # type: ignore
        
        # Policy info buffer
        # KL is for KL divergence
        # ent is entropy (randomness)
        # val is state-value from critic
        # val-loss is the loss from the critic model
        pi_info = dict(kl=[], ent=[], cf=[], val=np.array([]), val_loss=[])
        
        # Sample a random tensor
        ep_select = np.random.choice(
            np.arange(0, len(ep_form)), size=int(min_iterations), replace=False
        )
        ep_form = [ep_form[idx] for idx in ep_select]
        
        # Loss storage buffer(s)
        loss_sto: torch.Tensor = torch.tensor([], dtype=torch.float32)
        loss_arr: torch.Tensor = torch.autograd.Variable(
            torch.tensor([], dtype=torch.float32)
        )

        for ep in ep_form:
            # For each set of episodes per process from an epoch, compute loss
            trajectories = ep[0] # type: ignore
            hidden = self.reset_hidden() 
            obs, act, logp_old, adv, ret, src_tar = (
                trajectories[:, :observation_idx],
                trajectories[:, action_idx],
                trajectories[:, logp_old_idx],
                trajectories[:, advantage_idx],
                trajectories[:, return_idx, None],
                trajectories[:, source_loc_idx:].clone(),
            )
            
            # Calculate new action log probabilities

            pi, val, logp, loc = self.agent.grad_step(obs, act, hidden=hidden) # type: ignore
                
            logp_diff: torch.Tensor = logp_old - logp
            ratio = torch.exp(logp - logp_old)

            clip_adv = (torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv)
            clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)

            # Useful extra info
            clipfrac = (
                torch.as_tensor(clipped, dtype=torch.float32).detach().mean().item()
            )
            approx_kl = logp_diff.detach().mean().item()
            ent = pi.entropy().detach().mean().item()
            
            val_loss = self.agent_optimizer.MSELoss(val, ret) # MSE critc loss 

            # TODO: More descriptive name
            new_loss: torch.Tensor = -(
                torch.min(ratio * adv, clip_adv).mean()  # Policy loss
                - 0.01 * val_loss
                + self.alpha * ent
            )
            loss_arr = torch.hstack((loss_arr, new_loss.unsqueeze(0)))

            new_loss_sto: torch.Tensor = torch.tensor(
                [approx_kl, ent, clipfrac, val_loss.detach()]
            )
            loss_sto = torch.hstack((loss_sto, new_loss_sto.unsqueeze(0)))

        mean_loss = loss_arr.mean()
        means = loss_sto.mean(axis=0)  # type: ignore
        loss_pi, approx_kl, ent, clipfrac, loss_val = (
            mean_loss,
            means[0].detach(),
            means[1].detach(),
            means[2].detach(),
            means[3].detach(),
        )
        pi_info["kl"].append(approx_kl)  # type: ignore
        pi_info["ent"].append(ent)  # type: ignore
        pi_info["cf"].append(clipfrac)  # type: ignore
        pi_info["val_loss"].append(loss_val)  # type: ignore

        #kl = pi_info["kl"][-1].mean()  # type: ignore
        #Average KL across processes 
        kl = mpi_avg(pi_info["kl"][-1])             # MPI
                
        if kl.item() < 1.5 * self.target_kl:
            self.agent_optimizer.pi_optimizer.zero_grad()
            loss_pi.backward()
            
            mpi_avg_grads(self.agent.pi) # Average gradients across processes    # MPI
                    
            self.agent_optimizer.pi_optimizer.step()
            term = False
        else:
            term = True

        pi_info["kl"], pi_info["ent"], pi_info["cf"], pi_info["val_loss"] = (
            pi_info["kl"][0].numpy(),  # type: ignore
            pi_info["ent"][0].numpy(),  # type: ignore
            pi_info["cf"][0].numpy(),  # type: ignore
            pi_info["val_loss"][0].numpy(),  # type: ignore
        )
        loss_sum_new = loss_pi
        return (
            loss_sum_new,
            pi_info,
            term,
            (self.env_height * loc - (src_tar)).square().mean().sqrt(),
        )  # type: ignore

    def generate_mapstacks(self):
        ''' Generate a list of inflated maps from buffer '''
        actor_maps_buffer = list()
        critic_maps_buffer = list()
        
        # Clear existing maps
        self.agent.reset()
        
        # Convert observations to maps
        for step in self.ppo_buffer.full_observation_buffer:
            observation = {key: value for key, value in step.items() if key != 'terminal'}
            batched_actor_mapstack, batched_critic_mapstack = self.agent.get_map_stack(state_observation=observation, id=self.id)
            
            actor_maps_buffer.append(batched_actor_mapstack)
            critic_maps_buffer.append(batched_critic_mapstack)
            
            # If next observation is a fresh episode, clear maps
            if step['terminal']:
                self.agent.reset()
    
        return actor_maps_buffer, critic_maps_buffer

    def get_map_dimensions(self):
        return self.agent.get_map_dimensions()
    
    def get_map_count(self):
        return self.agent.get_map_count()
    
    def get_batch_size(self):
        return self.agent.get_batch_size()
    
    def save(self, path: str)-> None:
        ''' Wrapper for network '''
        self.agent.save(checkpoint_path=path)

    def load(self, path: str)-> None:
        ''' Wrapper for network '''
        self.agent.load(checkpoint_path=path)
        
    def render(self, savepath: str='.', save_map: bool=True, add_value_text: bool=False, interpolation_method: str='nearest', epoch_count: int=0, episode_count: int=0):
        print(f"Rendering heatmap for Agent {self.id}")
        self.agent.render(
            savepath=savepath, save_map=save_map, add_value_text=add_value_text, interpolation_method=interpolation_method, epoch_count=epoch_count, episode_count=episode_count
        )
        