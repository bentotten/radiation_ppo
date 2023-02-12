import torch
from torch.optim import Adam
import torch.nn.functional as F

import numpy as np
import numpy.typing as npt

from dataclasses import dataclass, field
from typing import TypeAlias, Union, cast, Optional, Any, NamedTuple
import scipy.signal

try:
    from epoch_logger import EpochLogger
except:
    from algos.multiagent.epoch_logger import EpochLogger

# Neural Networks
try:
    import NeuralNetworkCores.FF_core as RADFF_core
    import NeuralNetworkCores.CNN_core as RADCNN_core
    import NeuralNetworkCores.RADA2C_core as RADA2C_core
except:
    import algos.multiagent.NeuralNetworkCores.FF_core as RADFF_core
    import algos.multiagent.NeuralNetworkCores.CNN_core as RADCNN_core
    import algos.multiagent.NeuralNetworkCores.RADA2C_core as RADA2C_core


Shape: TypeAlias = int | tuple[int, ...]


def combined_shape(length: int, shape: Optional[Shape] = None) -> Shape:
    if shape is None:
        return (length,)
    elif np.isscalar(shape):
        shape = cast(int, shape)
        return (length, shape)
    else:
        shape = cast(tuple[int, ...], shape)
        return (length, *shape)


def discount_cumsum(
    x: npt.NDArray[np.float64], discount: float
) -> npt.NDArray[np.float64]:
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class BpArgs(NamedTuple):
    bp_decay: float
    l2_weight: float
    l1_weight: float
    elbo_weight: float
    area_scale: float


class UpdateResult(NamedTuple):
    StopIter: int
    LossPi: float
    LossV: float
    LossModel: float
    KL: npt.NDArray[np.float32]
    Entropy: npt.NDArray[np.float32]
    ClipFrac: npt.NDArray[np.float32]
    LocLoss: torch.Tensor
    VarExplain: int


@dataclass
class OptimizationStorage:
    train_pi_iters: int
    train_v_iters: int
    train_pfgru_iters: int    
    pi_optimizer: torch.optim
    critic_optimizer: torch.optim
    model_optimizer: torch.optim
    clip_ratio: float
    alpha: float
    target_kl: float
    
    pi_scheduler: torch.optim.lr_scheduler = field(init=False)  # Schedules gradient steps for actor
    critic_scheduler: torch.optim.lr_scheduler = field(init=False)  # Schedules gradient steps for value function (critic)
    pfgru_scheduler: torch.optim.lr_scheduler = field(init=False)   # Schedules gradient steps for PFGRU location predictor module
    loss: torch.nn.modules.loss.MSELoss = field(default_factory= (lambda: torch.nn.MSELoss(reduction="mean"))) # Loss calculator utility NOTE: Actor/PFGRU have other complex loss functions
        
    '''     
    This stores information related to updating neural network models for each agent. It includes the clip ratio for 
    ensuring a destructively large policy update doesn't happen, an entropy parameter for randomness/entropy,
    and the target KL for early stopping.
        
    train_pi_iters (int): Maximum number of gradient descent steps to take on actor policy loss per epoch. 
            (Early stopping may cause optimizer to take fewer than this.)

    train_v_iters (int): Number of gradient descent steps to take on critic state-value function per epoch.
            
    train_pfgru_iters (int): Number of gradient descent steps to take for source localization neural network
        (the PFGRU unit)
    
    {*}_optimizer (torch.optim): Pytorch Optimizer with learning rate decay [Torch]
    
    clip_ratio (float): Hyperparameter for clipping in the policy objective. Roughly: how far can the new policy 
        go from the old policy while still profiting (improving the objective function)? The new policy
        can still go farther than the clip_ratio says, but it doesn't help on the objective anymore. 
        (Usually small, 0.1 to 0.3.) Typically denoted by :math:`\epsilon`. Basically if the policy wants to 
        perform too large an update, it goes with a clipped value instead.
        
    alpha (float): Entropy reward term scaling used during calculating loss. 
    
    target_kl (float): Roughly what KL divergence we think is appropriate between new and old policies after an update.
        This will get used for early stopping (Usually small, 0.01 or 0.05.) 
    '''        
        
    def __post_init__(self):        
        self.pi_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pi_optimizer, step_size=100, gamma=0.99
        )
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pi_optimizer, step_size=100, gamma=0.99
        )        
        self.pfgru_scheduler = torch.optim.lr_scheduler.StepLR(
            self.model_optimizer, step_size=100, gamma=0.99
        )        


@dataclass
class PPOBuffer:
    obs_dim: int  # Observation space dimensions
    max_size: int  # Max steps per epoch

    episode_lengths: npt.NDArray[np.float32] = field(default_factory=list)  # Episode length storage

    ptr: int = field(init=False)  # For keeping track of location in buffer during update
    path_start_idx: int = field(init=False)  # For keeping track of starting location in buffer during update
    obs_buf: npt.NDArray[np.float32] = field(init=False)  # Observation buffer
    act_buf: npt.NDArray[np.float32] = field(init=False)  # Action buffer
    adv_buf: npt.NDArray[np.float32] = field(init=False)  # Advantages buffer
    rew_buf: npt.NDArray[np.float32] = field(init=False)  # Rewards buffer
    ret_buf: npt.NDArray[np.float32] = field(init=False)  # Cumulative(?) return buffer
    val_buf: npt.NDArray[np.float32] = field(init=False)  # State-value buffer
    source_tar: npt.NDArray[np.float32] = field(init=False) # Source location buffer (for moving targets)
    logp_buf: npt.NDArray[np.float32] = field(init=False)  # action log probabilities buffer
        
    obs_win: npt.NDArray[np.float32] = field(init=False) # TODO artifact - delete?
    obs_win_std: npt.NDArray[np.float32] = field(init=False) # TODO artifact - delete? Appears to be used in the location prediction, but is never updated

    gamma: float = 0.99
    lam: float = 0.90  # smoothing parameter for Generalize Advantage Estimate (GAE) calculations
    beta: float = 0.005

    """
    A buffer for storing histories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs. This is left outside of the
    PPO agent so that A2C architectures can be swapped out as desired.
    
    gamma (float): Discount rate for expected return and Generalize Advantage Estimate (GAE) calculations (Always between 0 and 1.)
    lam (float): Smoothing parameter for Generalize Advantage Estimate (GAE) calculations
    beta (float): Entropy for loss function, encourages exploring different policies
    """
    
    def __post_init__(self):
        self.ptr: int = 0
        self.path_start_idx: int = 0     
           
        self.obs_buf: npt.NDArray[np.float32] = np.zeros(
            combined_shape(self.max_size, self.obs_dim), dtype=np.float32
        )
        self.act_buf: npt.NDArray[np.float32] = np.zeros(
            combined_shape(self.max_size), dtype=np.float32
        )
        self.adv_buf: npt.NDArray[np.float32] = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.rew_buf: npt.NDArray[np.float32] = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.ret_buf: npt.NDArray[np.float32] = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.val_buf: npt.NDArray[np.float32] = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.source_tar: npt.NDArray[np.float32] = np.zeros(
            (self.max_size, 2), dtype=np.float32
        )
        self.logp_buf: npt.NDArray[np.float32] = np.zeros(
            self.max_size, dtype=np.float32
        )

        # TODO artifact - delete? Appears to be used in the location prediction, but is never updated        
        self.obs_win: npt.NDArray[np.float32] = np.zeros(self.obs_dim, dtype=np.float32)
        self.obs_win_std: npt.NDArray[np.float32] = np.zeros(
            self.obs_dim, dtype=np.float32
        )
        
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

    def store(
        self,
        obs: npt.NDArray[np.float32],
        act: npt.NDArray[np.float32],
        rew: npt.NDArray[np.float32],
        val: npt.NDArray[np.float32],
        logp: npt.NDArray[np.float32],
        src: npt.NDArray[np.float32],
    ) -> None:
        """
        Append one timestep of agent-environment interaction to the buffer.
        obs: observation (Usually the one returned from environment for previous step)
        act: action taken 
        rew: reward from environment
        val: state-value from critic
        logp: log probability from actor
        src: source coordinates
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr, :] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.source_tar[self.ptr] = src
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def store_episode_length(self, episode_length: int ) -> None:
        """
        Save episode length at the end of an epoch for later calculations
        """
        self.episode_lengths.append(episode_length)
            
    def finish_path(self, last_val: int = 0) -> None:
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        # gamma determines scale of value function, introduces bias regardless of VF accuracy
        # lambda introduces bias when VF is inaccurate
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self, logger=None) -> dict[str, Union[torch.Tensor, list]]:
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf: npt.NDArray[np.float32] = (self.adv_buf - adv_mean) / adv_std
        # ret_mean, ret_std = self.ret_buf.mean(), self.ret_buf.std()
        # self.ret_buf = (self.ret_buf) / ret_std
        # obs_mean, obs_std = self.obs_buf.mean(), self.obs_buf.std()
        # self.obs_buf_std_ind[:,1:] = (self.obs_buf[:,1:] - obs_mean[1:]) / (obs_std[1:])

        episode_lengths: list[int] = self.episode_lengths # TODO this needs to be cleared before can be used
        epLens: list[int] = logger.epoch_dict["EpLen"]  # TODO add to a buffer instead of pulling from logger
        
        number_episodes = len(episode_lengths)
        numEps = len(epLens)
        
        # TODO clear prior episode length buffer
        total_episode_length = sum(episode_lengths)
        epLenTotal = sum(epLens)
        
        # NOTE: Because rewards are from the shortest-path, these should not be applied intra-episode
        assert number_episodes > 0
        assert numEps > 0
        
        # data = dict(
        #     obs=torch.as_tensor(self.obs_buf, dtype=torch.float32),
        #     act=torch.as_tensor(self.act_buf, dtype=torch.float32),
        #     ret=torch.as_tensor(self.ret_buf, dtype=torch.float32),
        #     adv=torch.as_tensor(self.adv_buf, dtype=torch.float32),
        #     logp=torch.as_tensor(self.logp_buf, dtype=torch.float32),
        #     loc_pred=torch.as_tensor(self.obs_win_std, dtype=torch.float32), # TODO artifact - delete? Appears to be used in the location prediction, but is never updated
        #     ep_len=torch.as_tensor(total_episode_length, dtype=torch.float32),
        #     ep_form = []
        # )           
        
        data = dict(
            obs=torch.as_tensor(self.obs_buf, dtype=torch.float32),
            act=torch.as_tensor(self.act_buf, dtype=torch.float32),
            ret=torch.as_tensor(self.ret_buf, dtype=torch.float32),
            adv=torch.as_tensor(self.adv_buf, dtype=torch.float32),
            logp=torch.as_tensor(self.logp_buf, dtype=torch.float32),
            loc_pred=torch.as_tensor(self.obs_win_std, dtype=torch.float32), # TODO artifact - delete? Appears to be used in the location prediction, but is never updated
            ep_len=torch.as_tensor(epLenTotal, dtype=torch.float32),
            ep_form = []
        )     

        # If they're equal then we don't need to do anything
        # Otherwise we need to add one to make sure that numEps is the correct size
        episode_len_Size = (
            number_episodes
            + int(total_episode_length != len(self.obs_buf))
        )

        # If they're equal then we don't need to do anything
        # Otherwise we need to add one to make sure that numEps is the correct size
        epLenSize = (
            numEps
            + int(epLenTotal != len(self.obs_buf))
        )
        
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
        
        episode_form: list[list[torch.Tensor]] = [[] for _ in range(episode_len_Size)]
        epForm: list[list[torch.Tensor]] = [[] for _ in range(epLenSize)]
        
        slice_b: int = 0
        slice_f: int = 0
        jj: int = 0
        
        # TODO: This is essentially just a sliding window over obs_buf; use a built-in function to do this
        for ep_i in epLens:
            slice_f += ep_i
            epForm[jj].append(
                torch.as_tensor(obs_buf[slice_b:slice_f], dtype=torch.float32)
            )
            slice_b += ep_i
            jj += 1
        if slice_f != len(self.obs_buf):
            epForm[jj].append(
                torch.as_tensor(obs_buf[slice_f:], dtype=torch.float32)
            )

        data["ep_form"] = epForm

        return data


@dataclass
class AgentPPO:
    id: int
    number_of_agents: int
    observation_space: int
    action_space: int
    env_height: int
    environment_scale: int
    steps_per_episode: int    
    detector_step_size: int
    scaled_grid_bounds: tuple # Scaled to match return from env.step(). Can be reinflated with resolution_accuracy
    bounds_offset: tuple # Unscaled "observation area" to match map size to actual boundaries
    steps_per_epoch: int = field(default= 480)
    actor_critic_args: dict[str, Any] = field(default_factory= lambda: dict())
    actor_critic_architecture: str = field(default="cnn")  
    train_pi_iters: int = field(default= 40)
    train_v_iters: int = field(default= 40)
    train_pfgru_iters: int = field(default= 15)
    actor_learning_rate: float = field(default= 3e-4)
    critic_learning_rate: float = field(default= 1e-3)
    pfgru_learning_rate: float = field(default= 5e-3)    
    clip_ratio: float = field(default= 0.2)
    alpha: float = field(default= 0)
    target_kl: float = field(default= 0.07)
    reduce_pfgru_iters: bool = field(default=True) 
    gamma: float = field(default= 0.99)
    lam: float = field(default= 0.9)
    seed: int = field(default= 0)
    minibatch: int = field(default=1)
    enforce_boundaries: bool = field(default=False)  # Informs how large maps need to be to accomodate out-of-grid steps

    bp_args: BpArgs = field(init=False)

    def __post_init__(self):   
        # PFGRU args, from Ma et al. 2020
        self.bp_args = BpArgs(
            bp_decay=0.1,
            l2_weight=1.0,
            l1_weight=0.0,
            elbo_weight=1.0,
            area_scale=self.env_height
        )
                
        # Initialize agents
        match self.actor_critic_architecture: # Type: ignore                   
            case 'ff':
                self.agent = RADFF_core.PPO(self.observation_space, self.action_space, **self.actor_critic_args)              
            case 'cnn':                
                # How much unscaling to do. Current environment returnes scaled coordinates for each agent. A resolution_accuracy value of 1 here 
                #  means no unscaling, so all agents will fit within 1x1 grid. To make it less accurate but less memory intensive, reduce the 
                #  number being multiplied by the 1/env_scale. To return to full inflation, change multipier to 1
                multiplier = 0.01
                resolution_accuracy = multiplier * 1/self.environment_scale
                if self.enforce_boundaries:
                    scaled_offset = self.environment_scale * max(self.bounds_offset)                    
                else:
                    scaled_offset = self.environment_scale * (max(self.bounds_offset) + (self.steps_per_episode * self.detector_step_size))                
                
                # Initialize Agents                
                self.agent = RADCNN_core.CCNBase(
                    state_dim=self.observation_space, 
                    action_dim=self.action_space,
                    grid_bounds=self.scaled_grid_bounds,
                    resolution_accuracy=resolution_accuracy,
                    steps_per_epoch=self.steps_per_epoch,
                    id=self.id,
                    random_seed=self.seed,
                    scaled_offset = scaled_offset,
                    steps_per_episode=self.steps_per_episode,
                    number_of_agents=self.number_of_agents           
                    )
                
                # Initialize learning opitmizers                           
                self.agent_optimizer = OptimizationStorage(
                    train_pi_iters = self.train_pi_iters,                
                    train_v_iters = self.train_v_iters,
                    train_pfgru_iters = self.train_pfgru_iters,              
                    pi_optimizer = Adam(self.agent.pi.parameters(), lr=self.actor_learning_rate),
                    critic_optimizer = Adam(self.agent.critic.parameters(), lr=self.critic_learning_rate),
                    model_optimizer = Adam(self.agent.model.parameters(), lr=self.pfgru_learning_rate),
                    loss = torch.nn.MSELoss(reduction="mean"),
                    clip_ratio = self.clip_ratio,
                    alpha = self.alpha,
                    target_kl = self.target_kl,             
                    )                         
                
            case 'rnn':
                del self.actor_critic_args['enforce_boundaries']
                # Initialize Agents                
                self.agent = RADA2C_core.RNNModelActorCritic(self.observation_space, self.action_space, **self.actor_critic_args)
                
                # Initialize learning opitmizers                           
                self.agent_optimizer = OptimizationStorage(
                    train_pi_iters = self.train_pi_iters,                
                    train_v_iters = self.train_v_iters,
                    train_pfgru_iters = self.train_pfgru_iters,              
                    pi_optimizer = Adam(self.agent.pi.parameters(), lr=self.actor_learning_rate),
                    critic_optimizer = Adam(self.agent.pi.parameters(), lr=self.critic_learning_rate),
                    model_optimizer = Adam(self.agent.model.parameters(), lr=self.pfgru_learning_rate),
                    loss = torch.nn.MSELoss(reduction="mean"),
                    clip_ratio = self.clip_ratio,
                    alpha = self.alpha,
                    target_kl = self.target_kl,             
                    )                
            case 'mlp':
                # Initialize Agents
                self.agent = RADA2C_core.RNNModelActorCritic(self.observation_space, self.action_space, **self.actor_critic_args) 
                 
                # Initialize learning opitmizers           
                self.agent_optimizer = OptimizationStorage(
                    train_pi_iters = self.train_pi_iters,                
                    train_v_iters = self.train_v_iters,
                    train_pfgru_iters = self.train_pfgru_iters,              
                    pi_optimizer = Adam(self.agent.pi.parameters(), lr=self.actor_learning_rate),
                    critic_optimizer = Adam(self.agent.critic.parameters(), lr=self.critic_learning_rate),
                    model_optimizer = Adam(self.agent.model.parameters(), lr=self.pfgru_learning_rate),
                    loss = torch.nn.MSELoss(reduction="mean"),
                    clip_ratio = self.clip_ratio,
                    alpha = self.alpha,
                    target_kl = self.target_kl,             
                    )                         
            case _:
                raise ValueError('Unsupported neural network type')
            
        # Inititalize buffers
        self.ppo_buffer = PPOBuffer(
            obs_dim=self.observation_space, max_size=self.steps_per_epoch, gamma=self.gamma, lam=self.lam
            )
        
    def reduce_pfgru_training(self):
        '''Reduce localization module training iterations after some number of epochs to speed up training'''
        if self.reduce_pfgru_iters:
            self.train_pfgru_iters = 5
            self.reduce_pfgru_iters = False     
    
    def step(self, standardized_observations: npt.NDArray, hiddens: dict = None, save_map: bool = True, message: dict =None) -> RADCNN_core.ActionChoice:
        ''' Wrapper for neural network action selection'''
        if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
            results = self.agent.step(standardized_observations[self.id], hidden=hiddens[self.id])
        elif self.actor_critic_architecture == 'uniform':
            results = self.agent.select_action(observation=standardized_observations, message=message, id=self.id)         
        elif self.actor_critic_architecture == 'cnn':
            results: RADCNN_core.ActionChoice = self.agent.select_action(standardized_observations, self.id, save_map=save_map)  # TODO add in hidden layer shenanagins for PFGRU use
        elif self.actor_critic_architecture == 'ff':
            results = self.agent.select_action(standardized_observations, self.id, save_map=save_map)
        else:
            raise ValueError("Unknown architecture")
        return results         
    
    def reset_neural_nets(self, batch_size: int = 1) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        ''' Reset the neural networks at the end of an episode'''
        if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
            hiddens = self.agent.reset_hidden()
        else:
            # TODO implement reset function that has return values
            # actor_hidden = self.agent.pi._reset_state()
            # critic_hidden = self.agent.critic._reset_state()
            #pfgru_hidden = self.agent.model.init_hidden(batch_size)
            actor_hidden = 0
            critic_hidden = 0
            pfgru_hidden = 0
            hiddens = (actor_hidden, critic_hidden, pfgru_hidden)
            
            self.agent.clear_maps()
        
        return hiddens
     
    #TODO Make this a Ray remote function 
    def update_agent(self, logger = None) -> None: #         (env, bp_args, loss_fcn=loss)
        """
        Update for the localization (PFGRU) and A2C modules
        Note: update functions perform multiple updates per call
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
         
        # Get data from buffers
        data: dict[str, torch.Tensor] = self.ppo_buffer.get(logger)
        
        # NOTE: Not using observation tensor for CNN, using internal map buffer          

        # Update function for the PFGRU localization module. Module will be set to train mode, then eval mode within update_model
        # TODO get this working for CNN
        if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
            model_losses = self.update_model(data)
        else:
            # TODO remove after working for cnn
            model_losses = torch.tensor(0)
            
        # Update function if using the regression GRU
        # model_losses = update_loc_rnn(data,env,loss)

        # Length of data ep_form
        min_iterations = len(data["ep_form"])
        kk = 0
        term = False

        # RADPPO trains both actor and critic in same function/train_pi_iters, while TEAM-RAD needs to enable a global critic so iterates inside update_a2c() instead
        if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
            # Reset gradients 
            self.agent_optimizer.pi_optimizer.zero_grad()
            self.agent_optimizer.critic_optimizer.zero_grad()                
            # Train Actor-Critic policy with multiple steps of gradient descent. train_pi_iters == k_epochs
            while not term and kk < self.train_pi_iters:
                # Early stop training if KL-div above certain threshold
                #pi_l, pi_info, term, loc_loss = self.update_a2c(agent, data, min_iters, kk)  # pi_l = policy loss
                update_results = {}
                (
                    update_results['pi_l'], 
                    update_results['pi_info'], 
                    update_results['term'],  
                    update_results['loc_loss']
                ) = self.update_a2c(data, min_iterations, logger=logger)  # pi_l = policy loss
                kk += 1
                
            # Reduce learning rate
            self.agent_optimizer.pi_scheduler.step()
            self.agent_optimizer.critic_scheduler.step()            
            self.agent_optimizer.pfgru_scheduler.step()

            # Log changes from update
            return UpdateResult(
                StopIter=kk,
                LossPi=update_results['pi_l'].item(),
                LossV=update_results['pi_info']["val_loss"].item(),
                LossModel=model_losses.item(),  # TODO if using the regression GRU
                KL=update_results['pi_info']["kl"],
                Entropy=update_results['pi_info']["ent"],
                ClipFrac=update_results['pi_info']["cf"],
                LocLoss=update_results['loc_loss'],
                VarExplain=0
            )
                            
        else:
            # TODO incorporate maps into PPO buffer and avoid this entire process              
            # Match observation type to data and seperate map stacks from observation key for processing
            map_buffer_observations =  [torch.as_tensor(item[0], dtype=torch.float32) for item in self.agent.maps.observation_buffer]
            map_buffer_maps =  [item[1] for item in self.agent.maps.observation_buffer]  
            assert len(self.agent.maps.observation_buffer) == data['obs'].shape[0]
            
            # Check that maps match observations (round due to floating point precision in python)
            for _, (data_obs, map_obs) in enumerate(zip(data['obs'], map_buffer_observations)):
                assert torch.equal(data_obs, map_obs)            
            
            # Reset gradients 
            self.agent_optimizer.pi_optimizer.zero_grad()
            self.agent_optimizer.critic_optimizer.zero_grad()     
                
            # Train Actor policy with multiple steps of gradient descent. train_pi_iters == k_epochs
            for k_epoch in range(self.train_pi_iters):
                # Reset gradients 
                self.agent_optimizer.pi_optimizer.zero_grad()
                self.agent_optimizer.critic_optimizer.zero_grad()                 
                
                # Get indexes of episodes that will be sampled
                sample_indexes = sample(self, data=data)
                actor_loss_results = self.compute_batched_losses_pi(data=data, map_buffer_maps=map_buffer_maps, sample=sample_indexes)
                
                # Check Actor KL Divergence
                if actor_loss_results['kl'].item() < 1.5 * self.target_kl:
                    actor_loss_results['pi_loss'].backward() # TODO do we need to add entropy/lambda to this?
                    self.agent_optimizer.pi_optimizer.step() 
                else:
                    break  # Skip remaining training
                            
                # TODO add map buffer to PPO buffer and make this happen in get() function. Also rename get() to indicate buffers are reset
                self.agent.reset()                      
                
                # TODO Pull out for global critic
                self.agent_optimizer.critic_optimizer.zero_grad()
                critic_loss_results = self.compute_batched_losses_critic(data=data, map_buffer_maps=map_buffer_maps)
                critic_loss_results['critic_loss'].backward()
                self.agent_optimizer.critic_optimizer.step()
                      
            # # Value function learning
            # for _ in range(self.train_v_iters):
            #     self.agent_optimizer.critic_optimizer.zero_grad()
            #     critic_loss_results = self.compute_batched_losses_critic(self)
            #     critic_loss_results['critic_loss'].backward()
            #     self.agent_optimizer.critic_optimizer.step()
        
            # Reduce learning rate
            self.agent_optimizer.pi_scheduler.step()
            self.agent_optimizer.critic_scheduler.step()            
            
            # TODO Uncomment after implementing PFGRU
            #self.agent_optimizer.pfgru_scheduler.step()
            if self.agent.maps.location_map.max() !=0.0 or self.agent.maps.readings_map.max() !=0.0 or self.agent.maps.visit_counts_map.max() !=0.0:
                raise ValueError("Maps did not reset")   
        
            # TODO add map buffer to PPO buffer and make this happen in get() function. Also rename get() to indicate buffers are reset
            self.agent.reset()                
            
            # Log changes from update
            return UpdateResult(
                StopIter=k_epoch,  
                LossPi=actor_loss_results['pi_loss'].item(),
                LossV=critic_loss_results['critic_loss'].item(),
                LossModel=model_losses.item(),  # TODO implement when PFGRU is working for CNN
                KL=actor_loss_results["kl"],
                Entropy=actor_loss_results["entropy"],
                ClipFrac=actor_loss_results["clip_fraction"],
                LocLoss=0, # TODO implement when PFGRU is working for CNN
                VarExplain=0
            )
    
    def compute_batched_losses_pi(self, sample, data, map_buffer_maps, minibatch = None):
        ''' Simulates batched processing through CNN. Wrapper for computing single-batch loss for pi'''
                
        if not minibatch:
            minibatch = self.minibatch
        
        # TODO make more concise 
        # Due to linear layer in CNN, this must be run individually
        pi_loss_list = []
        kl_list = []
        entropy_list = []
        clip_fraction_list = []
        
        # Get sampled returns from actor and critic
        for index in sample:
            # Reset existing episode maps
            self.reset_neural_nets()                           
            single_pi_l, single_pi_info = self.compute_loss_pi(data=data, map_stack=map_buffer_maps[index], index=index)
            
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

    def compute_loss_pi(self, data: dict[torch.Tensor, list], map_stack: torch.Tensor, index:int = None):
        ''' Compute loss for actor network
            The difference between the probability of taking the action according to the current policy
            and the probability of taking the action according to the old policy, multiplied by the 
            advantage of the action.            
            
            data (array): data from PPO buffer
                obs (tensor): Unused - batch of observations from the PPO buffer. Currently only used to ensure
                    map buffer observations are correct.
                act (tensor): batch of actions taken
                adv (tensor): batch of advantages cooresponding to actions. 
                    These are the difference between the expected reward for taking that action and the baseline expected reward
                logp (tensor): batch of action logprobabilities
                loc_pred (tensor): batch of predicted location by PFGRU
                ep_len (tensor[int]): single dimension int of length of episode
                ep_form (tensor): 
                
            map_stacks (tensor): Either a single observations worth of maps, or a batch of maps
            index (int): If doing a single observation at a time, index for data[]
            
            Adapted from https://github.com/nikhilbarhate99/PPO-PyTorch
            
            Process:
                Calculate how much the policy has changed: 
                    ratio = policy_new / policy_old
                Take log form of this: 
                    ratio = [log(policy_new) - log(policy_old)].exp()
                Calculate Actor loss as the minimum of two functions: 
                    p1 = ratio * advantage
                    p2 = clip(ratio, 1-epsilon, 1+epsilon) * advantage
                    actor_loss = min(p1, p2)   
        '''
        # NOTE: Not using observation tensor, using internal map buffer
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Get action probabilities and entropy for an state's mapstack and action, then put the action probabilities on the CPU (if on the GPU)
        action_logprobs, dist_entropy = self.agent.pi.evaluate(map_stack, act[index])  
        action_logprobs = action_logprobs.cpu() # TODO do we need this on the CPU here?
        
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

    def compute_batched_losses_critic(self, data, map_buffer_maps):
        ''' Simulates batched processing through CNN. Wrapper for single-batch computing critic loss'''
        
        # Randomize and sample observation batch indexes
        ep_length = data["ep_len"].item()
        indexes = np.arange(0, ep_length, dtype=np.int32)
        number_of_samples = int((ep_length / self.minibatch))
        sample = np.random.choice(indexes, size=number_of_samples, replace=False) # Uniform            
        
        # TODO make more concise 
        # Due to linear layer in CNN, this must be run fully online (read: every map)
        critic_loss_list = []
        
        # Get sampled returns from actor and critic
        for index in sample:
            # Reset existing episode maps
            self.reset_neural_nets()                           
            critic_loss_list.append(self.compute_loss_critic(data=data, map_stack=map_buffer_maps[index], index=index))

        #take mean of everything for batch update
        results = {'critic_loss': torch.stack(critic_loss_list).mean()}
        return results
            
    def compute_loss_critic(self, data: dict[torch.Tensor, list], map_stack: torch.Tensor, index: int = None):
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
        # NOTE: Using mapstack from map buffer instead of observation
        # TODO add mapstack to PPO buffer instead of CNN
        true_return = data['ret'][index]
        
        # Compare predicted return with true return and use MSE to indicate loss
        predicted_value = self.agent.critic.evaluate(map_stack)
        critic_loss = self.agent.MseLoss(torch.squeeze(predicted_value), true_return)
        return critic_loss

    def update_model(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        ''' Update a single agent's PFGRU location prediction module (see Ma et al. 2020 for more details) '''      
        # Initial values and compatability
        args: BpArgs = self.bp_args
        ep_form = data["ep_form"]
        source_loc_idx = 15
        o_idx = 3
        
        # Put into training mode        
        self.agent.model.train() # PFGRU 
        
        for _ in range(self.train_pfgru_iters):
            model_loss_arr: torch.Tensor = torch.autograd.Variable(
                torch.tensor([], dtype=torch.float32)
            )
            for ep in ep_form:
                sl = len(ep[0])
                hidden = self.agent.reset_hidden()[0] 
                #src_tar: npt.NDArray[np.float32] = ep[0][:, source_loc_idx:].clone()
                src_tar: torch.Tensor = ep[0][:, source_loc_idx:].clone()
                src_tar[:, :2] = src_tar[:, :2] / args.area_scale
                obs_t = torch.as_tensor(ep[0][:, :o_idx], dtype=torch.float32)
                loc_pred = torch.empty_like(src_tar)
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

                total_loss = pred_loss
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
                total_loss: torch.Tensor = total_loss + args.elbo_weight * belief_loss

                model_loss_arr = torch.hstack((model_loss_arr, total_loss.unsqueeze(0)))

            model_loss: torch.Tensor = model_loss_arr.mean()
            self.agent_optimizer.model_optimizer.zero_grad()
            model_loss.backward()
            # Clip gradient TODO should 5 be a variable?
            # TODO Pylance error: https://github.com/Textualize/rich/issues/1523. Unable to resolve
            torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), 5) # TODO make multi-agent

            self.agent_optimizer.model_optimizer.step()

        self.agent.model.eval() 
        return model_loss

    def update_a2c(
            self, data: dict[str, torch.Tensor], min_iterations: int,  logger: EpochLogger, minibatch: Union[int, None] = None
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor], bool, torch.Tensor]:
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

        ep_form = data["ep_form"]
        
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
            trajectories = ep[0]
            hidden = self.reset_neural_nets() 
            obs, act, logp_old, adv, ret, src_tar = (
                trajectories[:, :observation_idx],
                trajectories[:, action_idx],
                trajectories[:, logp_old_idx],
                trajectories[:, advantage_idx],
                trajectories[:, return_idx, None],
                trajectories[:, source_loc_idx:].clone(),
            )
            
            # Calculate new action log probabilities

            pi, val, logp, loc = self.agent.grad_step(obs, act, hidden=hidden)
                
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
            
            val_loss = self.agent_optimizer.loss(val, ret) # MSE critc loss 

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
        means = loss_sto.mean(axis=0)
        loss_pi, approx_kl, ent, clipfrac, loss_val = (
            mean_loss,
            means[0].detach(),
            means[1].detach(),
            means[2].detach(),
            means[3].detach(),
        )
        pi_info["kl"].append(approx_kl)
        pi_info["ent"].append(ent)
        pi_info["cf"].append(clipfrac)
        pi_info["val_loss"].append(loss_val)

        kl = pi_info["kl"][-1].mean()
        if kl.item() < 1.5 * self.target_kl:
            self.agent_optimizer.pi_optimizer.zero_grad()
            loss_pi.backward()
            self.agent_optimizer.pi_optimizer.step()
            term = False
        else:
            term = True

        pi_info["kl"], pi_info["ent"], pi_info["cf"], pi_info["val_loss"] = (
            pi_info["kl"][0].numpy(),
            pi_info["ent"][0].numpy(),
            pi_info["cf"][0].numpy(),
            pi_info["val_loss"][0].numpy(),
        )
        loss_sum_new = loss_pi
        return (
            loss_sum_new,
            pi_info,
            term,
            (self.env_height * loc - (src_tar)).square().mean().sqrt(),
        )       
        
    def render(self, savepath: str=None, save_map: bool=True, add_value_text: bool=False, interpolation_method: str='nearest', epoch_count: int=0):
        print(f"Rendering heatmap for Agent {self.id}")
        self.agent.render(
            savepath=savepath, save_map=save_map, add_value_text=add_value_text, interpolation_method=interpolation_method, epoch_count=epoch_count
        )