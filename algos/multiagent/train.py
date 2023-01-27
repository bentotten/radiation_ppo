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
class train_PPO:
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
    start_time: float = field(default_factory= lambda: time.time()),
    minibatch: int = field(default=1)
    DEBUG: bool = field(default=False)
    
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

        gamma (float):  Discount rate for expected return and Generalize Advantage Estimate (GAE) calculations (Always between 0 and 1.)
        
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
            
        minibatch (int): How many observations to sample out of a batch. Used to reduce the impact of fully online learning

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
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)      

        # Set additional Actor-Critic variables
        self.ac_kwargs["seed"] = self.seed
        self.ac_kwargs["pad_dim"] = 2        

        # Get environment information
        self.obs_dim: int = self.env.observation_space.shape[0]
        self.act_dim: int = rad_search_env.A_SIZE
        self.env_scale: int = self.env.scale
        self.bounds_offset: tuple = self.env.observation_area
        self.step_size: int = self.env.step_size
        scaled_grid_bounds = (1, 1)  # Scaled to match current return from env.step(). Can be reinflated with resolution_accuracy        
        
        # For logging
        config_json: dict[str, Any] = convert_json(locals())

        self.agents: dict[int, AgentPPO] = {
            i: AgentPPO(
                id=i,
                steps_per_epoch=self.steps_per_epoch,
                actor_critic_architecture=self.actor_critic_architecture,
                observation_space=self.obs_dim, 
                action_space=self.act_dim,
                bounds_offset=self.bounds_offset,
                steps_per_episode=self.steps_per_episode,
                detector_step_size=self.step_size,
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
                environment_scale=self.env_scale,                
                env_height=self.env.search_area[2][1],
                scaled_grid_bounds=scaled_grid_bounds,
                seed=self.seed,
                minibatch=self.minibatch
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
            self.loggers[id].setup_pytorch_saver(self.agents[id].agent.pi)  # Only setup to save one nn module currently, here saving the policy
            
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
        
        # Obsertvations for each agent, 11 dimensions: [intensity reading, x coord, y coord, 8 directions of distance detected to obstacle]        
        observations, _,  _, _ = env.reset()
        for id in self.agents: 
            self.agents[id].reset_neural_nets()  # NOTE: buffers are cleared during update
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
            # Reset hidden layers and sets Actor into "eval" mode. For CNN, resets maps
            hiddens = {id: ac.reset_neural_nets() for id, ac in self.agents.items()}            
            if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                for ac in self.agents.values():
                    ac.agent.pi.logits_net.v_net.eval() # TODO should the pfgru call .eval also?                
            else:
                for ac in self.agents.values():
                    #ac.model.eval()  # TODO add PFGRU
                    ac.agent.pi.eval()
                    ac.agent.critic.eval() # TODO will need to be changed for global critic
            
            # Start episode!
            for steps in range(self.steps_per_epoch):
                # Standardize prior observation of radiation intensity for the actor-critic input using running statistics per episode
                if self.actor_critic_architecture == 'cnn':
                    # TODO add back in for PFGRU
                    standardized_observations = observations
                else:
                    # TODO observation is overwritten by obs_std; was this intentional? If so,why does it exist?                
                    standardized_observations = {id: observations[id] for id in self.agents}
                    for id in self.agents:
                        standardized_observations[id][0] = np.clip((observations[id][0] - self.stat_buffers[id].mu) / self.stat_buffers[id].sig_obs, -8, 8)     
                        
                # Actor: Compute action and logp (log probability); Critic: compute state-value
                agent_thoughts = {id: None for id in self.agents}
                for id, ac in self.agents.items():
                    agent_thoughts[id] = ac.step(standardized_observations, hiddens)
                    #action, value, logprob, hiddens[self.id], out_prediction = ac.step
                
                for id in self.agents:                                
                    if self.DEBUG:
                        if int(agent_thoughts[id].action.item()) == self.act_dim-1:
                            print("Max Action!")
                    
                # Create action list to send to environment
                agent_action_decisions = {id: int(agent_thoughts[id].action.item()) for id in agent_thoughts} 
                
                # TODO the above does not include idle action. After working, add an additional state space for 9 potential actions and uncomment:                 
                #agent_action_decisions = {id: int(action)-1 for id, action in agent_thoughts.items()} 
                
                # Ensure no item is above max actions or below 0
                # Idle action is max action dimension (here 8)
                for action in agent_action_decisions.values():
                    assert 0 <= action and action < self.act_dim            
                
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
                        val = agent_thoughts[id].state_value,
                        logp = agent_thoughts[id].action_logprob,
                        src = source_coordinates,
                        #terminal = terminals[id],  # TODO do we want to store terminal flags?
                    )
                    
                    self.loggers[id].store(VVals=agent_thoughts[id].state_value)
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
                        #if DEBUG: 
                            #print(f"Agent out of bounds at ({observations[id][1]}, {observations[id][2]})")
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
                        if self.actor_critic_architecture == 'cnn':
                            # TODO add back in for PFGRU
                            standardized_observations = observations
                        else:
                            standardized_observations = {id: observations[id] for id in self.agents}
                            for id in self.agents:
                                standardized_observations[id][0] = np.clip(
                                    (observations[id][0] - self.stat_buffers[id].mu) / self.stat_buffers[id].sig_obs, -8, 8
                                )     
                        for id, ac in self.agents.items():
                            results = ac.step(standardized_observations, hiddens=hiddens, save_map=False)  # Ensure next map is not buffered when going to compare to logger for update
                            value = results.state_value
 
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
                        # Render gif
                        env.render(
                            path=f"{self.logger_kwargs['data_dir']}/{self.logger_kwargs['env_name']}",
                            epoch_count=epoch,
                        )
                        # Render Agent heatmaps
                        for id, ac in self.agents.items():
                            ac.render(
                                savepath=f"{self.logger_kwargs['data_dir']}/{self.logger_kwargs['env_name']}", 
                                epoch_count=epoch,
                                add_value_text=True
                            )

                    # Reset the environment and counters
                    episode_return_buffer = []
                    for id in self.agents:
                         self.stat_buffers[id].reset()
                         
                    # If not at the end of an epoch, reset hidden layers for incoming new episode    
                    # TODO why not reset hidden layers at the end of an epoch?                  
                    if not env.epoch_end:
                        for id, ac in self.agents.items():                        
                            hiddens[id] = ac.reset_neural_nets()
                    # Else log epoch results                    
                    else:
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
                update_results = ac.update_agent(self.loggers[id])
                
                # Store results
                # TODO some of these are getting updated within the update_agent function
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

