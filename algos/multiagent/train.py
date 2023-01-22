'''
Originally built from https://github.com/nikhilbarhate99/PPO-PyTorch, however has been merged with RAD-PPO
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

from typing import Any, List, Literal, NewType, Optional, TypedDict, cast, get_args, Dict, NamedTuple, Type, Union
from typing_extensions import TypeAlias
from dataclasses import dataclass, field
from typing_extensions import TypeAlias

# Simulation Environment
import gym
from gym_rad_search.envs import rad_search_env # type: ignore
from gym_rad_search.envs.rad_search_env import RadSearch, StepResult  # type: ignore
from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore

# PPO
from ppo import OptimizationStorage, PPOBuffer, AgentPPO

# Neural Networks
import NeuralNetworkCores.FF_core as RADFF_core
import NeuralNetworkCores.CNN_core as RADCNN_core
import NeuralNetworkCores.RADA2C_core as RADA2C_core

# Data Management Utility
from epoch_logger import EpochLogger, EpochLoggerKwargs, setup_logger_kwargs, convert_json

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   HARDCODE TEST DELETE ME  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
DEBUG = True
CNN = True  # TODO remove after done
SCOOPERS_IMPLEMENTATION = False
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Scaling
# TODO get from env instead, remove from global
DET_STEP = 100.0  # detector step size at each timestep in cm/s
DET_STEP_FRAC = 71.0  # diagonal detector step size in cm/s
DIST_TH = 110.0  # Detector-obstruction range measurement threshold in cm
DIST_TH_FRAC = 78.0  # Diagonal detector-obstruction range measurement threshold in cm

################################### Functions for algorithm/implementation conversions ###################################

def count_variables(module: nn.Module) -> int:
    return sum(np.prod(p.shape) for p in module.parameters())


def convert_nine_to_five_action_space(action):
    ''' Converts 4 direction + idle action space to 9 dimensional equivelant
        Environment action values:
        -1: idle
        0: left
        1: up and left
        2: upfrom gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore
        3: up and right
        4: right
        5: down and right
        6: down
        7: down and left

        Cardinal direction action values:
        -1: idle
        0: left
        1: up
        2: right
        3: down
    '''
    match action:
        # Idle
        case -1:
            return -1
        # Left
        case 0:
            return 0
        # Up
        case 1:
            return 2
        # Right
        case 2:
            return 4
        # Down
        case 3:
            return 6
        case _:
            raise Exception('Action is not within valid [-1,3] range.')


class AgentStepReturn(NamedTuple):
    action: Union[npt.NDArray, None]
    value:  Union[npt.NDArray, None]
    logprob:  Union[npt.NDArray, None]
    hidden:  Union[torch.Tensor, None]
    out_prediction:  Union[npt.NDArray, None]


@dataclass
class StatBuff:
    mu: float = 0.0
    sig_sto: float = 0.0
    sig_obs: float = 1.0
    count: int = 0
    ''' statistics buffer for normalizing returns from environment '''
    
    def update(self, obs: float) -> None:
        self.count += 1
        if self.count == 1:
            self.mu = obs
        else:
            mu_n = self.mu + (obs - self.mu) / (self.count)
            s_n = self.sig_sto + (obs - self.mu) * (obs - mu_n)
            self.mu = mu_n
            self.sig_sto = s_n
            self.sig_obs = max(math.sqrt(s_n / (self.count - 1)), 1)

    def reset(self) -> None:
        self = StatBuff()


################################### Training ###################################
@dataclass
class PPO:
    env: RadSearch
    logger_kwargs: EpochLoggerKwargs
    seed: int = field(default= 0)
    steps_per_epoch: int = field(default= 480)
    steps_per_episode: int = field(default= 120)
    total_epochs: int = field(default= 3000)
    save_freq: int = field(default= 500)
    save_gif_freq: int = field(default= float('inf'))
    render: bool = field(default= False)
    save_gif: bool = field(default= False)
    gamma: float = field(default= 0.99)
    alpha: float = field(default= 0)
    clip_ratio: float = field(default= 0.2)
    mp_mm: tuple[int, int] = field(default= (5, 5))
    pi_lr: float = field(default= 3e-4)
    critic_learning_rate: float = field(default= 1e-3)
    pfgru_learning_rate: float = field(default= 5e-3)
    train_pi_iters: int = field(default= 40)
    train_v_iters: int = field(default= 40)
    train_pfgru_iters: int = field(default= 15)
    reduce_pfgru_iters: bool = field(default=True)
    lam: float = field(default= 0.9)
    number_of_agents: int = field(default= 1)
    target_kl: float = field(default= 0.07)
    ac_kwargs: dict[str, Any] = field(default_factory= lambda: dict())
    #actor_critic: Type[RADA2C_core.RNNModelActorCritic] = field(default=RADA2C_core.RNNModelActorCritic)
    actor_critic_architecture: str = field(default="cnn")
    start_time: float = field(default_factory= lambda: time.time())
    
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Base code from OpenAI:
    https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

    Args:
        env : An environment satisfying the OpenAI Gym API.
        
        logger_kwargs: Arguments for the logging mechanism for saving models and saving/printing progress for each agent.
            Note that the logger is also used for calculating values later on in the episode.
        
        seed (int): Seed for random number generators.     

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch before updating the neural network modules.
            
        steps_per_episode (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each episode before resetting the environment.        

        total_epochs (int): Number of total epochs of interaction (equivalent to
            number of policy updates) to perform.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
            
        save_gif_freq (int): How many epochs to save a gif
            
        render (bool): Indicates whether to render last episode
        
        save_gif (bool): Indicates whether to save render of last episode

        gamma (float): Discount factor for calculating expected return. (Always between 0 and 1.)
        
        alpha (float): Entropy reward term scaling.        

        clip_ratio (float): Usually seen as Epsilon Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help on the objective anymore. 
            (Usually small, 0.1 to 0.3.).Basically if the policy wants to perform too large an update, 
            it goes with a clipped value instead.

        pi_lr (float): Learning rate for Actor/policy optimizer.

        critic_learning_rate (float): Learning rate for Critic (value) function optimizer.
        
        pfgru_learning_rate (float): Learning rate for the source prediction module (PFGRU)

        train_pi_iters (int): Maximum number of gradient descent steps to take on actor policy loss per epoch. 
            (Early stopping may cause optimizer to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            critic state-value function per epoch.
            
        train_pfgru_iters (int): Number of gradient descent steps to take for source localization neural network (the PFGRU unit)           

        reduce_pfgru_iters (bool): Reduces PFGRU training iteration when further along to speed up training

        lam (float): Lambda for GAE-Lambda advantage estimator calculations. (Always between 0 and 1,
            close to 1.)

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.) This is a part of PPO.        

        ac_kwargs (dict): Any kwargs appropriate for the Actor-Critic object
            provided to PPO.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                        | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                        | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                        | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                        | a batch of distributions describing
                                        | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                        | actions is given). Tensor containing
                                        | the log probability, according to
                                        | the policy, of the provided actions.
                                        | If actions not given, will contain
                                        | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                        | for the provided observations. (Critical:
                                        | make sure to flatten this!)
            ===========  ================  ======================================
    """
    def __post_init__(self):  
        
        # TODO get rid of redundant for loops
                
        # Set Pytorch random seed
        torch.manual_seed(self.seed)

        # Set additional Actor-Critic variables
        self.ac_kwargs["seed"] = self.seed
        self.ac_kwargs["pad_dim"] = 2        

        # Get environment information
        self.obs_dim: int = self.env.observation_space.shape[0]
        self.act_dim: int = rad_search_env.A_SIZE
        
        # For logging
        config_json: dict[str, Any] = convert_json(locals())

        self.agents: dict[int, PPO] = {
            i: AgentPPO(
                steps_per_epoch=self.steps_per_epoch,
                actor_critic_architecture=self.actor_critic_architecture,
                observation_space=self.obs_dim, 
                action_space=self.act_dim, 
                actor_critic_args=self.ac_kwargs,
                actor_learning_rate=self.pi_lr,
                critic_learning_rate=self.critic_learning_rate,
                pfgru_learning_rate=self.pfgru_learning_rate,
                train_pi_iters=self.train_pi_iters,
                train_v_iters=self.train_v_iters,
                train_pfgru_iters=self.train_pfgru_iters,
                reduce_pfgru_iters=self.reduce_pfgru_iters,
                clip_ratio=self.clip_ratio,
                alpha=self.alpha,
                target_kl=self.target_kl,
                env_height=self.env.search_area[2][1],
            ) for i in range(self.number_of_agents)
        }
        
        # TODO add PFGRU to FF and CNN networks
        for ac in self.agents.values():
            ac.agent.model.eval() # Sets PFGRU model into "eval" mode # TODO why not in the episode with the other agents?   
        
        # Setup statistics buffers for normalizing returns from environment
        self.stat_buffers = {i: StatBuff() for i in range(self.number_of_agents)}
                
        # Count variables for actor/policy (pi) and PFGRU (model)
        self.pi_var_count, self.model_var_count = {}, {}
        for id, ac in self.agents.items():
            self.pi_var_count[id], self.model_var_count[id] = (
                count_variables(module) for module in [ac.agent.pi, ac.agent.model]
            )   
            
        # Instatiate loggers and set up model saving                 
        logger_kwargs_set = {
            id: setup_logger_kwargs(
                exp_name=f"{self.logger_kwargs['exp_name']}_agent{id}",
                seed=self.logger_kwargs['seed'],
                data_dir=self.logger_kwargs['data_dir'],
                env_name=self.logger_kwargs['env_name']
            ) for id in self.agents
        }
        
        self.loggers = {id: EpochLogger(**(logger_kwargs_set[id])) for id in self.agents}
        
        for id in self.agents:
            self.loggers[id].save_config(config_json)    
            self.loggers[id].log(
                f"\nNumber of parameters: \t actor policy (pi): {self.pi_var_count[0]}, particle filter gated recurrent unit (model): {self.model_var_count[0]} \t"
            )
            self.loggers[id].setup_pytorch_saver(self.agents[id].agent)
            
        # Save env image
        if self.render:
            self.env.render(
            path=f"{self.logger_kwargs['data_dir']}/{self.logger_kwargs['env_name']}",
            epoch_count=0,
            just_env=True
            )                    

    def train(self):
        # Prepare environment variables and reset
        env = self.env
        observations, _,  _, _ = env.reset()  # Obsertvations for each agent, 11 dimensions: [intensity reading, x coord, y coord, 8 directions of distance detected to obstacle]
        source_coordinates = np.array(self.env.src_coords, dtype="float32")  # Target for later NN update after episode concludes
        episode_return = {id: 0 for id in self.agents}
        episode_return_buffer = []  # TODO can probably get rid of this, unless want to keep for logging
        steps_in_episode = 0
        
        # Prepare epoch variables
        out_of_bounds_count = {id: 0 for id in self.agents} # Out of Bounds counter for the epoch (not the episode)
        terminal_counter = {id: 0 for id in self.agents}  # Terminal counter for the epoch (not the episode)

        # Update stat buffers for all agent observations for later observation normalization
        for id in self.agents:
            self.stat_buffers[id].update(observations[id][0])
        
        # TODO Removed features - migrating to pytorch lightning instead of mpi (God willing)
        #local_steps_per_epoch = int(steps_per_epoch / num_procs())    

        print(f"Starting main training loop!", flush=True)
        self.start_time = time.time()        
        # For a total number of epochs, Agent will choose an action using its policy and send it to the environment to take a step in it, yielding a new state observation.
        #   Agent will continue doing this until the episode concludes; a check will be done to see if Agent is at the end of an epoch or not - if so, the agent will use 
        #   its buffer to update/train its networks. Sometimes an epoch ends mid-episode - there is a finish_path() function that addresses this.
        for epoch in range(self.total_epochs):
            # Reset hidden states and sets Actor into "eval" mode 
            if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                hiddens = {id: ac.agent.reset_hidden() for id, ac in self.agents.items()}
                for ac in self.agents.values():
                    ac.agent.pi.logits_net.v_net.eval() # TODO should the pfgru call .eval also?                
            else:
                for ac in self.agents.values():
                    ac.agent.actor.eval()
                    ac.agent.critic.eval() # TODO will need to be changed for global critic
            
            # Start episode!
            for steps in range(self.steps_per_epoch):
                # Standardize prior observation of radiation intensity for the actor-critic input using running statistics per episode
                standardized_observations = {id: observations[id] for id in self.agents}
                for id in self.agents:
                    standardized_observations[id][0] = np.clip((observations[id][0] - self.stat_buffers[id].mu) / self.stat_buffers[id].sig_obs, -8, 8)     
                    
                # Actor: Compute action and logp (log probability); Critic: compute state-value
                agent_thoughts = {id: None for id in self.agents}
                for id, ac in self.agents.items():
                    if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                        action, value, logprob, hiddens[id], out_prediction = ac.agent.step(standardized_observations[id], hidden=hiddens[id])
                    else:
                        # TODO
                        pass  
                          
                    agent_thoughts[id] = AgentStepReturn(
                        action=action, value=value, logprob=logprob, hidden=hiddens[id], out_prediction=out_prediction
                    )
                
                # Create action list to send to environment
                agent_action_decisions = {id: int(agent_thoughts[id].action.item()) for id in agent_thoughts} 
                
                # TODO the above does not include idle action. After working, add an additional state space for 9 potential actions and uncomment:                 
                #agent_action_decisions = {id: int(action)-1 for id, action in agent_thoughts.items()} 
                
                # Ensure no item is above 7 or below -1
                for action in agent_action_decisions.values():
                    assert -1 <= action and action < 8            
                
                # Take step in environment - Critical that this value is saved as "next" observation so we can link
                #  rewards from this new state to the prior step/action
                next_observations, rewards, terminals, infos = env.step(action=agent_action_decisions) 
                
                # Incremement Counters and save new (individual) cumulative returns
                for id in rewards:
                    episode_return[id] += rewards[id]
                episode_return_buffer.append(episode_return)
                steps_in_episode += 1    

                # Store previous observations in buffers, update mean/std for the next observation in stat buffers,
                #   record state values with logger 
                for id, ac in self.agents.items():
                    ac.ppo_buffer.store(
                        obs = observations[id],
                        rew = rewards[id],
                        act = agent_action_decisions[id],
                        val = agent_thoughts[id].value,
                        logp = agent_thoughts[id].logprob,
                        src = source_coordinates,
                        #terminal = terminals[id],  # TODO do we want to store terminal flags?
                    )
                    
                    self.loggers[id].store(VVals=agent_thoughts[id].value)
                    self.stat_buffers[id].update(next_observations[id][0])                    

                # Update obs (critical!)
                assert observations is not next_observations, 'Previous step observation is pointing to next observation'
                observations = next_observations

                # Tally up ending conditions
                
                # Check if there was a terminal state. Note: if terminals are introduced that only affect one agent but not
                #  all, this will need to be changed.
                terminal_reached_flag = False
                for id in terminal_counter:
                    if terminals[id] == True and not timeout:
                        terminal_counter[id] += 1   
                        terminal_reached_flag = True             
                # Check if some agents went out of bounds
                for id in infos:
                    if 'out_of_bounds' in infos[id] and infos[id]['out_of_bounds'] == True:
                        if DEBUG: 
                            print(f"Agent out of bounds at ({observations[id][1]}, {observations[id][2]})")
                        out_of_bounds_count[id] += infos[id]['out_of_bounds_count']
                                    
                # Stopping conditions for episode
                timeout = steps_in_episode == self.steps_per_episode
                terminal = terminal_reached_flag or timeout
                epoch_ended = steps == self.steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not (terminal):
                        print(
                            f"Warning: trajectory cut off by epoch at {steps_in_episode} steps and step count {steps}.",
                            flush=True,
                        )

                    if timeout or epoch_ended:
                        # if trajectory didn't reach terminal state, bootstrap value target with standardized observation using per episode running statistics 
                        standardized_observations = {id: observations[id] for id in self.agents}
                        for id in self.agents:
                            standardized_observations[id][0] = np.clip(
                                (observations[id][0] - self.stat_buffers[id].mu) / self.stat_buffers[id].sig_obs, -8, 8
                            )     
                            for id, ac in self.agents.items():
                                _, value, _, _, _ = ac.agent.step(standardized_observations[id], hidden=hiddens[id])
 
                        if epoch_ended:
                            # Set flag to sample new environment parameters
                            env.epoch_end = True 
                    else:
                        value = 0
                    # Finish the trajectory and compute advantages. See function comments for more information                        
                    for id, ac in self.agents.items():
                        ac.ppo_buffer.finish_path(value)
                        
                    if terminal:
                        # only save episode returns and episode length if trajectory finished
                        for id, ac in self.agents.items():
                            self.loggers[id].store(EpRet=episode_return[id], EpLen=steps_in_episode)
                            # TODO verify matches logger - goal is to get logger out of PPO buffer
                            ac.ppo_buffer.store_episode_length(episode_length=steps_in_episode)

                    # If at the end of an epoch and render flag is set or the save_gif frequency indicates it is time to
                    asked_to_save = epoch_ended and self.render
                    save_first_epoch = (epoch != 0 or self.save_gif_freq == 1)
                    save_time_triggered = (epoch % self.save_gif_freq == 0) if self.save_gif_freq != 0 else False
                    time_to_save = save_time_triggered or ((epoch + 1) == self.total_epochs)
                    if (asked_to_save and save_first_epoch and time_to_save):
                        env.render(
                            path=f"{self.logger_kwargs['data_dir']}/{self.logger_kwargs['env_name']}",
                            epoch_count=epoch,
                        )

                    # Reset the environment and counters
                    episode_return_buffer = []
                    for id in self.agents:
                         self.stat_buffers[id].reset()
                    # If not at the end of an epoch, reset the detector position and episode tracking for incoming new episode                      
                    if not env.epoch_end:
                        for id, ac in self.agents.items():                        
                            hiddens[id] = ac.agent.reset_hidden()
                    else:
                        # Sample new environment parameters, log epoch results
                        for id in self.agents:
                            if 'out_of_bounds_count' in infos[id]:
                                out_of_bounds_count[id] += infos[id]['out_of_bounds_count']  # TODO this was already done above, is this being done twice?
                            self.loggers[id].store(DoneCount=terminal_counter[id], OutOfBound=out_of_bounds_count[id])
                            terminal_counter[id] = 0
                            out_of_bounds_count[id] = 0
                    
                    # Reset environment. Obsertvations for each agent - 11 dimensions: [intensity reading, x coord, y coord, 8 directions of distance detected to obstacle]
                    observations, _,  _, _ = env.reset()                          
                    source_coordinates = np.array(self.env.src_coords, dtype="float32")  # Target for later NN update after episode concludes
                    episode_return = {id: 0 for id in self.agents}
                    episode_return_buffer = []  # TODO can probably get rid of this, unless want to keep for logging
                    steps_in_episode = 0
                    source_coordinates = np.array(env.src_coords, dtype="float32")

                    # Update stat buffers for all agent observations for later observation normalization
                    for id in self.agents:
                        self.stat_buffers[id].update(observations[id][0])                              
                    
            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.total_epochs - 1):
                for id in self.agents:
                    self.loggers[id].save_state({}, None)
                pass

            # Reduce localization module training iterations after 100 epochs to speed up training
            if epoch > 99:
                for ac in self.agents.values():
                    ac.reduce_pfgru_training()

            # Perform PPO update!
            for id, ac in self.agents.items():
                update_results = ac.update_agent()
                
                # Store results
                self.loggers[id].store(
                    StopIter=update_results.StopIter,
                    LossPi=update_results.LossPi,
                    LossV=update_results.LossV,
                    LossModel=update_results.LossModel,
                    KL=update_results.KL,
                    Entropy=update_results.Entropy,
                    ClipFrac=update_results.ClipFrac,
                    LocLoss=update_results.LocLoss,
                    VarExplain=update_results.VarExplain,
                )            
            
            if not terminal:
                pass            

            # Log info about epoch
            for id in self.agents:
                self.loggers[id].log_tabular("AgentID", id)        
                self.loggers[id].log_tabular("Epoch", epoch)      
                self.loggers[id].log_tabular("EpRet", with_min_and_max=True)
                self.loggers[id].log_tabular("EpLen", average_only=True)
                self.loggers[id].log_tabular("VVals", with_min_and_max=True)
                self.loggers[id].log_tabular("TotalEnvInteracts", (epoch + 1) * self.steps_per_epoch)
                self.loggers[id].log_tabular("LossPi", average_only=True)
                self.loggers[id].log_tabular("LossV", average_only=True)
                self.loggers[id].log_tabular("LossModel", average_only=True)  # Specific to the regressive GRU
                self.loggers[id].log_tabular("LocLoss", average_only=True)
                self.loggers[id].log_tabular("Entropy", average_only=True)
                self.loggers[id].log_tabular("KL", average_only=True)
                self.loggers[id].log_tabular("ClipFrac", average_only=True)
                self.loggers[id].log_tabular("DoneCount", sum_only=True)
                self.loggers[id].log_tabular("OutOfBound", average_only=True)
                self.loggers[id].log_tabular("StopIter", average_only=True)
                self.loggers[id].log_tabular("Time", time.time() - self.start_time)                 
                self.loggers[id].dump_tabular()


def train_scaffolding():
    print("============================================================================================")

    # Using this to fold setup in VSCode
    if True:
        ####### initialize environment hyperparameters ######
        env_name = "radppo-v2"

        # max_ep_len = 1000                   # max timesteps in one episode
        #training_timestep_bound = int(3e6)   # break training loop if timeteps > training_timestep_bound TODO DELETE
        epochs = int(3e6)  # Actual epoch will be a maximum of this number + max_ep_len
        steps_per_epoch = 3000
        max_ep_len = 120                      # max timesteps in one episode
        #training_timestep_bound = 100  # Change to epoch count DELETE ME

        # print avg reward in the interval (in num timesteps)
        #print_freq = max_ep_len * 3
        print_freq = max_ep_len * 100
        # log avg rewardfrom gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore in the interval (in num timesteps)
        log_freq = max_ep_len * 2
        # save model frequency (in num timesteps)
        save_model_freq = int(1e5)

        # starting std for action distribution (Multivariate Normal)
        action_std = 0.6
        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        action_std_decay_rate = 0.05
        # minimum action_std (stop decay after action_std <= min_action_std)
        min_action_std = 0.1
        # action_std decay frequency (in num timesteps)
        action_std_decay_freq = int(2.5e5)
        #####################################################

        # Note : print/log frequencies should be > than max_ep_len

        ################ PPO hyperparameters ################
        steps_per_epoch = 480
        K_epochs = 80               # update policy for K epochs in one PPO update
        eps_clip = 0.2          # clip parameter for PPO
        gamma = 0.99            # discount factor
        lamda = 0.95            # smoothing parameter for Generalize Advantage Estimate (GAE) calculations
        beta: float = 0.005     # TODO look up what this is doing

        lr_actor = 0.0003       # learning rate for actor network
        lr_critic = 0.001       # learning rate for critic network

        random_seed = 0        # set random seed if required (0 = no random seed)
        
        #################################################torchinfo####

        ###################### logging ######################
        
        # For render
        render = True
        save_gif_freq = 1

        # log files for multiple runs are NOT overwritten
        log_dir = "PPO_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + env_name + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # get number of log files in log directory
        run_num = 0
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)

        # create new log file for each run
        log_f_name = log_dir + 'PPO_' + env_name + "_log_" + str(run_num) + ".csv"

        print("current logging run number for " + env_name + " : ", run_num)
        print("logging at : " + log_f_name)
        #####################################################

        ################### checkpointing ###################
        # change this to prevent overwriting weights in same env_name folder
        run_num_pretrained = 0

        # directoself.step(action=None, action_list=None)ry = "PPO_preTrained"
        directory = "RAD_PPO"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/' + env_name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
        print("save checkpoint path : " + checkpoint_path)

        print(f"Current directory: {os.getcwd()}")
        #####################################################

        ################ Setup Environment ################

        print("training environment name : " + env_name)
        
        # Generate a large random seed and random generator object for reproducibility
        #robust_seed = _int_list_from_bigint(hash_seed(seed))[0] # TODO get this to work
        #rng = npr.default_rng(robust_seed)
        # Pass np_random=rng, to env creation

        obstruction_count = 0
        number_of_agents = 1
        env: RadSearch = RadSearch(number_agents=number_of_agents, seed=random_seed, obstruction_count=obstruction_count)
        
        resolution_accuracy = 1 * 1/env.scale  
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   HARDCODE TEST DELETE ME  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if DEBUG:
            epochs = 1   # Actual epoch will be a maximum of this number + max_ep_len
            max_ep_len = 5                      # max timesteps in one episode # TODO delete me after fixing
            steps_per_epoch = 5
            K_epochs = 4
                        
            obstruction_count = 0 #TODO error with 7 obstacles
            number_of_agents = 1
            
            seed = 0
            random_seed = _int_list_from_bigint(hash_seed(seed))[0]
            
            log_freq = 2000
            
            render = False
            
            #bbox = tuple(tuple(((0.0, 0.0), (2000.0, 0.0), (2000.0, 2000.0), (0.0, 2000.0))))  
            
            #env: RadSearch = RadSearch(DEBUG=DEBUG, number_agents=number_of_agents, seed=random_seed, obstruction_count=obstruction_count, bbox=bbox) 
            env: RadSearch = RadSearch(DEBUG=DEBUG, number_agents=number_of_agents, seed=random_seed, obstruction_count=obstruction_count)         
            
            # How much unscaling to do. State returnes scaled coordinates for each agent. 
            # A resolution_accuracy value of 1 here means no unscaling, so all agents will fit within 1x1 grid
            resolution_accuracy = 0.01 * 1/env.scale  # Less accurate
            #resolution_accuracy = 1 * 1/env.scale   # More accurate
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        env.render(
            just_env=True,
            path=directory,
        )
        # state space dimension
        state_dim = env.observation_space.shape[0]

        # action space dimension
        action_dim = env.action_space.n
        
        # Search area
        search_area = env.search_area[2][1]
        
        # Scaled grid dimensions
        scaled_grid_bounds = (1, 1)  # Scaled to match return from env.step(). Can be reinflated with resolution_accuracy


        ############# print all hyperparameters #############
        print("--------------------------------------------------------------------------------------------")
        #print("max training timesteps : ", training_timestep_bound)
        print("max training epochs : ", epochs)    
        print("max timesteps per episode : ", max_ep_len)
        print("model saving frequency : " + str(save_model_freq) + " timesteps")
        print("log frequency : " + str(log_freq) + " timesteps")
        print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
        print("--------------------------------------------------------------------------------------------")
        print("state space dimension : ", state_dim)
        print("action space dimension : ", action_dim)
        print("Grid space bounds : ", scaled_grid_bounds)
        print("--------------------------------------------------------------------------------------------")
        print("Initializing a discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("PPO update frequency : " + str(steps_per_epoch) + " timesteps")
        print("PPO K epochs : ", K_epochs)
        print("PPO epsilon clip : ", eps_clip)
        print("discount factor (gamma) : ", gamma)
        print("--------------------------------------------------------------------------------------------")
        print("optimizer learning rate actor : ", lr_actor)
        print("optimizer learning rate critic : ", lr_critic)
        if random_seed:
            print("--------------------------------------------------------------------------------------------")
            print("setting random seed to ", random_seed)
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        #####################################################

        print("============================================================================================")

        ################# training procedure ################

        # initialize PPO agents
        ppo_agents = {i:
            PPO(
                state_dim=state_dim, 
                action_dim=action_dim, 
                grid_bounds=scaled_grid_bounds, 
                lr_actor=lr_actor, 
                lr_critic=lr_critic,
                gamma=gamma, 
                K_epochs=K_epochs, 
                eps_clip=eps_clip,
                resolution_accuracy=resolution_accuracy,
                steps_per_epoch=steps_per_epoch,
                id=i,
                lamda=lamda,
                beta = beta,
                random_seed= _int_list_from_bigint(hash_seed(seed))[0]
                ) 
            for i in range(number_of_agents)
            }
        
        # TODO move to a unit test
        if number_of_agents > 1:
            ppo_agents[0].maps.buffer.adv_buf[0] = 1
            assert ppo_agents[1].maps.buffer.adv_buf[0] != 1, "Singleton pattern in buffer class"
            ppo_agents[0].maps.buffer.adv_buf[0] = 0.0


        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        print("============================================================================================")

        # logging file
        # TODO Need to log for each agent?
        log_f = open(log_f_name, "w+")
        log_f.write('episode,timestep,reward\n')

        total_time_step = 0

        # Initial values
        source_coordinates = np.array(env.src_coords, dtype="float32")  # Target for later NN update after episode concludes
        episode_return = {id: 0 for id in ppo_agents}
        episode_return_buffer = []  # TODO can probably get rid of this, unless want to keep for logging
        out_of_bounds_count = np.zeros(number_of_agents)
        terminal_counter = 0
        steps_in_episode = 0
        #local_steps_per_epoch = int(steps_per_epoch / num_procs()) # TODO add this after everything is working
        local_steps_per_epoch = steps_per_epoch
        
        # state = env.reset()['state'] # All agents begin in same location
        # Returns aggregate_observation_result, aggregate_reward_result, aggregate_done_result, aggregate_info_result
        # Unpack relevant results. This primes the pump for agents to choose an action.    
        observations = env.reset().observation # All agents begin in same location, only need one state
        
    # Training loop
    #while total_time_step < training_timestep_bound:
    #while steps_in_epoch < epochs:
    for epoch_counter in range(epochs):
        
        # Put actors into evaluation mode
        # TODO why here and not in update?
        for agent in ppo_agents.values(): 
            agent.policy.eval()

        for steps_in_epoch in range(local_steps_per_epoch):
            
            # TODO From philippe - is this necessary and should it be added? Appears to make observations between 0 and 1
            #Standardize input using running statistics per episode
            # obs_std = o
            # obs_std[0] = np.clip((o[0]-stat_buff.mu)/stat_buff.sig_obs,-8,8)            
            
            # Get actions
            if CNN:
                agent_action_returns = {id: agent.select_action(observations, id) for id, agent in ppo_agents.items()} # TODO is this running the same state twice for every step?                
            else:
                # Vanilla FFN
                agent_action_returns = {id: agent.select_action(observations[id].state) -1 for id, agent in ppo_agents.items()} # TODO is this running the same state twice for every step?
            
            if number_of_agents == 1:
                assert ppo_agents[0].maps.others_locations_map.max() == 0.0
            
            # Convert actions to include -1 as "idle" option
            # TODO Make this work in the env calculation for actions instead of here, and make 0 the idle state            
            # TODO REMOVE convert_nine_to_five_action_space AFTER WORKING WITH DIAGONALS
            if CNN:
                if SCOOPERS_IMPLEMENTATION:
                    include_idle_action_list = {id: action.action - 1 for id, action in agent_action_returns.items()}                    
                    action_list = {id: convert_nine_to_five_action_space(action) for id, action in include_idle_action_list.items()}
                else:
                    action_list = {id: action.action - 1 for id, action in agent_action_returns.items()}      
            else:
                # Vanilla FFN
                action_list = agent_action_returns
            
            # Sanity check
            # Ensure no item is above 7 or below -1
            for action in action_list.values():
                assert action < 8 and action >= -1

            next_results = env.step(action_list=action_list, action=None)

            # Unpack information
            next_observations = next_results.observation
            rewards = next_results.reward
            successes = next_results.success
            infos = next_results.info
                
            # Sanity Check
            # Ensure Agent moved in a direction
            for id in ppo_agents:
                # Because of collision avoidance, this assert will not hold true for multiple agents
                if number_of_agents == 1 and action_list[id] != -1 and not infos[id]["out_of_bounds"] and not infos[id]['blocked']:
                    assert (next_observations[id][1] != observations[id][1] or next_observations[id][2] !=  observations[id][2]), "Agent coodinates did not change when should have"

            # Incremement Counters and save new cumulative return
            for id in rewards:
                episode_return[id] += rewards[id]

            episode_return_buffer.append(episode_return)
            total_time_step += 1
            steps_in_episode += 1            
            
            # saving prior state, and current reward/is_terminals etc
            if CNN:
                for id, agent in ppo_agents.items():
                    obs: npt.NDArray[Any] = observations[id]
                    rew: npt.NDArray[np.float32] = rewards[id]
                    terminal: npt.NDArray[np.bool] = successes[id]                                        
                    act: npt.NDArray[np.int32] = agent_action_returns[id].action           
                    val: npt.NDArray[np.float32] = agent_action_returns[id].state_value      
                    logp: npt.NDArray[np.float32] = agent_action_returns[id].action_logprob
                    src: npt.NDArray[np.float32] = source_coordinates                    
                
                    agent.store(
                        obs = obs,
                        act = act,
                        rew = rew,
                        val = val,
                        logp = logp,
                        src = src,
                        terminal = terminal,
                    )

            # Update obs (critical!)
            observations = next_observations
            
            # Check if there was a success
            sucess_counter = 0
            for id in successes:
                if successes[id] == True:
                    sucess_counter += 1

            # Check if some agents went out of bounds
            for id in infos:
                if 'out_of_bounds' in infos[id] and infos[id]['out_of_bounds'] == True:
                        out_of_bounds_count[id] += 1
                                    
            # Stopping conditions for episode
            timeout = steps_in_episode == max_ep_len
            terminal = sucess_counter > 0 or timeout
            epoch_ended = steps_in_epoch == local_steps_per_epoch - 1
            
            if terminal or epoch_ended:
                if terminal and not timeout:
                    success_count += 1

                if epoch_ended and not (terminal):
                    print(f"Warning: trajectory cut off by epoch at {steps_in_episode} steps in episode, at epoch count {steps_in_epoch}.", flush=True)

                if timeout or epoch_ended:
                    # if trajectory didn't reach terminal state, bootstrap value target
                    
                    # TODO Philippes normalizing thing, see if we want this
                    #obs_std[0] = np.clip((o[0]-stat_buff.mu)/stat_buff.sig_obs,-8,8)
                    
                    # TODO Investigate why all state_values are identical
                    agent_state_values = {id: agent.select_action(observations, id).state_value for id, agent in ppo_agents.items()}
                                                            
                    if epoch_ended:
                        # Set flag to sample new environment parameters
                        env.epoch_end = True # TODO make multi-agent?
                else:
                    agent_state_values = {id: 0 for id in ppo_agents}        
                
                # Finish the path and compute advantages
                for id, agent in ppo_agents.items():
                    agent.maps.buffer.finish_path_and_compute_advantages(agent_state_values[id])
                    
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    agent.maps.buffer.store_episode_length(steps_in_episode)

                if (epoch_ended and render and (((epoch_counter + 1) == epochs) or epoch_counter % save_gif_freq == 0)):
                    # Render agent progress during training
                    #if proc_id() == 0 and epoch != 0:
                    if epoch_counter != 0:
                        for agent in ppo_agents.values():
                            agent.render(
                                add_value_text=True, 
                                savepath=directory,
                                epoch_count=epoch_counter,
                            )                   
                        env.render(
                            path=directory,
                            epoch_count=epoch_counter,
                            )
                if DEBUG and render:
                    for agent in ppo_agents.values():
                        agent.render(
                            add_value_text=True, 
                            savepath=directory,
                            epoch_count=epoch_counter,
                        )                   
                    env.render(
                        path=directory,
                        epoch_count=epoch_counter,
                        )                     

                episode_return_buffer = []
                # Reset the environment
                if not epoch_ended: 
                    # Reset detector position and episode tracking
                    # hidden = self.ac.reset_hidden()
                    pass 
                else:
                    # Sample new environment parameters, log epoch results
                    #oob += env.oob_count
                    #logger.store(DoneCount=done_count, OutOfBound=oob)
                    success_count = 0
                    out_of_bounds_count = np.zeros(number_of_agents)

                # Unpack relevant results. This primes the pump for agents to choose an action.                
                observations = env.reset().observation
                source_coordinates = np.array(env.src_coords, dtype="float32")  # Target for later NN update after episode concludes
                episode_return = {id: 0 for id in ppo_agents}
                steps_in_episode = 0
                
                # Update stats buffer for normalizer
                #stat_buff.update(o[0])

        # update PPO agents
        if CNN:
            for id, agent in ppo_agents.items():
                agent.update()
        
        # Vanilla FFN
        else:
            for id, agent in ppo_agents.items():
                agent.buffer.rewards.append(rewards[id])
                agent.buffer.is_terminals.append(successes[id])
                ####
                # update PPO agent
                agent.update()
                                    
        # printing average reward
        if total_time_step % print_freq == 0:
            print("Epoch : {} \t\t  Total Timestep: {} \t\t Cumulative Returns: {} \t\t Out of bounds count: {}".format(
                epoch_counter, total_time_step, episode_return_buffer, out_of_bounds_count))
            
        # TODO Logging logger goes here
                 

        # Save model weights
        # if total_time_step % save_model_freq == 0:
        #     print(
        #         "--------------------------------------------------------------------------------------------")
        #     print("saving model at : " + checkpoint_path)
        #     # Render last episode
        #     print("TEST RENDER - Delete Me Later")
        #     episode_rewards = {id: render_buffer_rewards[-max_ep_len:] for id, agent in ppo_agents.items()}
        #     env.render(
        #         path=directory,
        #         epoch_count=epoch_counter,
        #         episode_rewards=episode_rewards
        #     )
        #     #ppo_agent.save(checkpoint_path)
        #     print("model saved")
        #     print("Elapsed Time  : ", datetime.now().replace(
        #         microsecond=0) - start_time)
        #     print(
        #         "--------------------------------------------------------------------------------------------")

    # Render last episode
    env.render(
        path=directory,
        epoch_count=epoch_counter,
    )
    for agent in ppo_agents.values():
        agent.render(add_value_text=True, savepath=directory)   

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


