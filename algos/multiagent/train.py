'''
Train agents and update neural networks using simulation environment.
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

from typing import Any, List, Literal, NewType, Optional, TypedDict, cast, get_args, Dict, NamedTuple, Type, Union, Tuple
from typing_extensions import TypeAlias
from dataclasses import dataclass, field

# Simulation Environment
import gym  # type: ignore
from gym_rad_search.envs import rad_search_env # type: ignore
from gym_rad_search.envs.rad_search_env import RadSearch, StepResult  # type: ignore
from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore

# PPO and logger
try:
    from ppo import OptimizationStorage, PPOBuffer, AgentPPO  # type: ignore
    from epoch_logger import EpochLogger, EpochLoggerKwargs, setup_logger_kwargs, convert_json  # type: ignore
except ModuleNotFoundError:
    from algos.multiagent.ppo import OptimizationStorage, PPOBuffer, AgentPPO  # type: ignore
    from algos.multiagent.epoch_logger import EpochLogger, EpochLoggerKwargs, setup_logger_kwargs, convert_json
except: 
    raise Exception

# Neural Networks
try:
    import NeuralNetworkCores.FF_core as RADFF_core # type: ignore
    import NeuralNetworkCores.RADTEAM_core as RADCNN_core # type: ignore
    import NeuralNetworkCores.RADA2C_core as RADA2C_core # type: ignore
    from NeuralNetworkCores.RADTEAM_core import StatisticStandardization # type: ignore
except ModuleNotFoundError:
    import algos.multiagent.NeuralNetworkCores.FF_core as RADFF_core # type: ignore
    import algos.multiagent.NeuralNetworkCores.RADTEAM_core as RADCNN_core # type: ignore
    import algos.multiagent.NeuralNetworkCores.RADA2C_core as RADA2C_core # type: ignore
    from algos.multiagent.NeuralNetworkCores.RADTEAM_core import StatisticStandardization # type: ignore
except: 
    raise Exception

# TODO add to a argument
COMPETATIVE_MODE = False  # Dictates whether individual rewards or team rewards

################################### Training ###################################
@dataclass
class train_PPO:
    ''' Proximal Policy Optimization (by clipping) with early stopping based on approximate KL Divergence. Base code from OpenAI: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
    This class focuses on the coordination part of training an actor-critic model, including coordinating agent objects, interacting with simulation environment for a certain number of epochs, and 
    calling an agents update function according to a seperate neural network module.
    
    Steps:
    
    #. Set seed for pytorch and numpy
    #. Set up logger. Will save to a directory named "models" and the chosen experiment name. Configurations are stored in the first agents directory.

    :param env: An environment satisfying the OpenAI Gym API.    
    :param logger_kwargs: Static parameters for the logging mechanism for saving models and saving/printing progress for each agent. Note that the logger is also used for calculating values later on in the episode.
    :param ppo_kwargs: Static parameters for ppo method. Also contains arguments for actor-critic neural networks.
    :param seed: (int) Seed for random number generators.
    :param number_of_agents: (int) Number of agents
    :param actor_critic_architecture: (string) Short-version indication for what neural network core to use for actor-critic agent
    :param steps_per_epoch: (int) Number of steps of interaction (state-action pairs) for the agent and the environment in each epoch before updating the neural network modules.
    :param steps_per_episode: (int) Number of steps of interaction (state-action pairs) for the agent and the environment in each episode before resetting the environment.        
    :param total_epochs: (int) Number of total epochs of interaction (equivalent to number of policy updates) to perform.
    :param render: (bool) Indicates whether to render last episode    
    :param save_freq: (int) How often (in terms of gap between epochs) to save the current policy and value function.
    :param save_gif_freq: (int) How many epochs to save a gif
    :param save_gif: (bool) Indicates whether to save render of last episode
    :param render_first_episode: (bool) If render, render the first episode and then follow save-gif-freq parameter
    :param DEBUG: (bool) indicate whether in debug mode with hardcoded start/stopping locations   
    
    '''
    # Environment
    env: RadSearch    
    
    # Pass-through arguments
    logger_kwargs: EpochLoggerKwargs
    ppo_kwargs: Dict[str, Any] = field(default_factory= lambda: dict())

    # Random seed
    seed: int = field(default= 0)    

    # Agent information
    number_of_agents: int = field(default= 1)
    actor_critic_architecture: str = field(default="cnn")
    
    # Simulation parameters
    steps_per_epoch: int = field(default= 480)
    steps_per_episode: int = field(default= 120)
    total_epochs: int = field(default= 3000)
    
    # Rendering information
    render: bool = field(default= False)    
    save_freq: int = field(default= 500)
    save_gif_freq: Union[int, float] = field(default_factory= lambda:  float('inf'))
    save_gif: bool = field(default= False)
    render_first_episode: bool = field(default=True)     
    
    #: DEBUG mode adds extra print statements/rendering to train function
    DEBUG: bool = field(default=False)

    # Initialized elsewhere
    #: Time experiment was started
    start_time: float = field(default_factory= lambda: time.time())
    #: Object that normalizes returns from environment for RAD-A2C. RAD-TEAM does so from within PPO module
    stat_buffers: Dict[int, StatisticStandardization] = field(default_factory=lambda:dict())
    #: Object that holds agents
    agents: Dict[int, AgentPPO] = field(default_factory=lambda:dict())
    #: Object that holds agent loggers
    loggers: Dict[int, EpochLogger] = field(default_factory=lambda:dict())
    
    def __post_init__(self)-> None:  
        # Set Pytorch random seed
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)    

        # Save configuration   
        config_json: Dict[str, Any] = convert_json(locals())                       
                                  
        # Instatiate loggers and save initial parameters in the first agent slot
        # TODO save configurations in parent directory
        for id in range(self.number_of_agents):
            logger_kwargs_set: Dict = setup_logger_kwargs(
                exp_name=f"{self.logger_kwargs['exp_name']}_agent{id}",
                seed=self.logger_kwargs['seed'],
                data_dir=self.logger_kwargs['data_dir'],
                env_name=self.logger_kwargs['env_name']
            )
        
            self.loggers[id] = EpochLogger(**(logger_kwargs_set))
        self.loggers[0].save_config(config_json) 
        
        # Initialize agents        
        for i in range(self.number_of_agents):
            # If RAD-A2C, set up statistics buffers         
            if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                self.stat_buffers[i] = StatisticStandardization()          
                
            self.agents[i] = AgentPPO(id=i, **self.ppo_kwargs)
            
            self.loggers[i].setup_pytorch_saver(self.agents[i].agent.pi)  # Only setup to save one nn module currently, here saving the policy        
                      
    def train(self)-> None:
        ''' Function that executes training simulation. 
            #. Begin experiment.
            #. While epoch count is less than max epochs, 
            - Reset the environment
            - Begin epoch. While stepcount is less than max steps per epoch:
            -- Each agent chooses an action from reset environment
            -- Send actions to environment and receive rewards, observations, whether or not the source was found, and boundary information back
            -- Save the observations and returns in buffers
            -- Check if the episode or epoch has ended because of a timeout or a terminal condition.
            --- If the episode has ended but not the epoch, reset environment and hidden layers/map stacks
            --- If the epoch has ended, use existing buffers to update the networks, then reset all buffers, hidden networks, and the environment        
        '''       
        # Reset environment and load initial observations
        #   Obsertvations for each agent, 11 dimensions: [intensity reading, x coord, y coord, 8 directions of distance detected to obstacle]        
        observations, _,  _, infos = self.env.reset()
        
        # Prepare environment variables and reset
        source_coordinates: npt.NDArray = np.array(self.env.src_coords, dtype="float32")  # Target for later NN update after episode concludes
        episode_return: Dict[int, float] = {id: 0.0 for id in self.agents}
        steps_in_episode: int = 0
        
        # Prepare epoch variables
        out_of_bounds_count: Dict[int, int] = {id: 0 for id in self.agents} # Out of Bounds counter for the epoch (not the episode)
        terminal_counter: Dict[int, int] = {id: 0 for id in self.agents}  # Terminal counter for the epoch (not the episode)

        # TODO move to PPO
        # Update stat buffers for all agent observations for later observation normalization
        # Set initial reading value for standardization
        if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
            for id in self.agents:
                self.stat_buffers[id].update(observations[id][0])
        # For RAD-TEAM, the update is done inside the step

        # TODO add PFGRU to FF and CNN networks
        for id in self.agents: 
            self.agents[id].reset_neural_nets()      
            self.agents[id].agent.model.eval() # Sets PFGRU model into "eval" mode # TODO why not in the episode with the other agents?      

        print(f"Starting main training loop!", flush=True)
        self.start_time: float = time.time()        
        
        # For a total number of epochs, Agent will choose an action using its policy and send it to the environment to take a step in it, yielding a new state observation.
        #   Agent will continue doing this until the episode concludes; a check will be done to see if Agent is at the end of an epoch or not - if so, the agent will use 
        #   its buffer to update/train its networks. Sometimes an epoch ends mid-episode - there is a finish_path() function that addresses this.
        for epoch in range(self.total_epochs):
            # Reset hidden layers and sets Actor into "eval" mode. For CNN, resets maps
            hiddens: Dict[int, Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = {id: ac.reset_neural_nets() for id, ac in self.agents.items()}            
            if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                for ac in self.agents.values():
                    ac.agent.pi.logits_net.v_net.eval() # TODO should the pfgru call .eval also?
            else:
                for ac in self.agents.values():
                    if ac.agent.maps.location_map.max() !=0.0 or ac.agent.maps.readings_map.max() !=0.0 or ac.agent.maps.visit_counts_map.max() !=0.0:
                        raise ValueError("Maps did not reset")                       
            
            # Start episode!
            for steps_in_epoch in range(self.steps_per_epoch):             
                # Standardize prior observation of radiation intensity for the actor-critic input using running statistics per episode
                if self.actor_critic_architecture == 'cnn':
                    # TODO add back in for PFGRU
                    standardized_observations = observations
                else:
                    # TODO observation is overwritten by obs_std; was this intentional? If so, why does it exist?                
                    standardized_observations = {id: observations[id] for id in self.agents}
                    if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':            
                        for id in self.agents:
                            standardized_observations[id][0] = self.stat_buffers[id].standardize(observations[id][0])
                        
                # Actor: Compute action and logp (log probability); Critic: compute state-value
                agent_thoughts: Dict[int, RADCNN_core.ActionChoice] = dict()
                for id, ac in self.agents.items():
                    agent_thoughts[id] = ac.step(standardized_observations=standardized_observations, hiddens = hiddens, save_map = True, message=infos)
                    #action, value, logprob, hiddens[self.id], out_prediction = ac.step
                    
                # Create action list to send to environment
                agent_action_decisions = {id: int(agent_thoughts[id].action.item()) for id in agent_thoughts} 
                
                # Ensure no item is above max actions or below 0. Idle action is max action dimension (here 8)
                for action in agent_action_decisions.values():
                    assert 0 <= action and action < int(self.env.number_actions)
                
                # Take step in environment - Critical that this value is saved as "next" observation so we can link
                #  rewards from this new state to the prior step/action
                next_observations, rewards, terminals, infos = self.env.step(action=agent_action_decisions) 
                
                # Incremement Counters and save new (individual) cumulative returns
                if COMPETATIVE_MODE:
                    for id in rewards['individual_reward']:
                        episode_return[id] += np.array(rewards['individual_reward'][id], dtype="float32").item()
                else:
                    for id in self.agents:
                        episode_return[id] += np.array(rewards['team_reward'], dtype="float32").item() # TODO if saving team reward, no need to keep duplicates for each agent
                    
                steps_in_episode += 1    

                # Store previous observations in buffers, update mean/std for the next observation in stat buffers,
                #   record state values with logger 
                # TODO Change away from numpy array
                for id, ac in self.agents.items():
                    if COMPETATIVE_MODE:
                        reward = rewards['individual_reward'][id]
                    else:
                        reward = rewards['team_reward']
                    ac.ppo_buffer.store(
                        obs = observations[id],
                        rew = reward,
                        act = agent_action_decisions[id],
                        val = agent_thoughts[id].state_value,
                        logp = agent_thoughts[id].action_logprob,
                        src = source_coordinates,
                        #terminal = terminals[id],  # TODO do we want to store terminal flags?
                    )
                    
                    self.loggers[id].store(VVals=agent_thoughts[id].state_value)
                    
                    if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                        self.stat_buffers[id].update(next_observations[id][0])                    

                # Update obs (critical!)
                assert observations is not next_observations, 'Previous step observation is pointing to next observation'
                observations = next_observations

                # Tally up ending conditions
                # TODO move this to seperate function
                # Check if there was a terminal state. Note: if terminals are introduced that only affect one agent but not all, this will need to be changed.
                terminal_reached_flag = False
                for id in terminal_counter:
                    if terminals[id] == True and not timeout:
                        terminal_counter[id] += 1   
                        terminal_reached_flag = True             
                # Check if some agents went out of bounds
                for id in infos:
                    if 'out_of_bounds' in infos[id] and infos[id]['out_of_bounds'] == True:
                        out_of_bounds_count[id] += 1
                                    
                # Stopping conditions for episode
                timeout: bool = steps_in_episode == self.steps_per_episode
                terminal: bool = terminal_reached_flag or timeout
                epoch_ended: bool = steps_in_epoch == self.steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not (terminal):
                        print(
                            f"Warning: trajectory cut off by epoch at {steps_in_episode} steps and step count {steps_in_epoch}.",
                            flush=True,
                        )

                    if timeout or epoch_ended:
                        if self.actor_critic_architecture == 'cnn':
                            # TODO add back in for PFGRU
                            standardized_observations = observations
                        else:
                            # if trajectory didn't reach terminal state, bootstrap value target with standardized observation using per episode running statistics                            
                            standardized_observations = {id: observations[id] for id in self.agents}
                            if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':    
                                for id in self.agents:
                                    standardized_observations[id][0] = self.stat_buffers[id].standardize(observations[id][0])
                        for id, ac in self.agents.items():
                            if self.actor_critic_architecture == 'uniform':
                                results = ac.step(standardized_observations, hiddens=hiddens, save_map=False, messages=infos)  # Ensure next map is not buffered when going to compare to logger for update
                            else:
                                results = ac.step(standardized_observations, hiddens=hiddens, save_map=False)  # Ensure next map is not buffered when going to compare to logger for update
                            value = results.state_value
 
                        if epoch_ended:
                            # Set flag to sample new environment parameters
                            self.env.epoch_end = True 
                    else:
                        value = 0  # State value 
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
                        # Render Agent heatmaps
                        if self.actor_critic_architecture == 'cnn':
                            for id, ac in self.agents.items():
                                ac.render(
                                    savepath=f"{self.logger_kwargs['data_dir']}/{self.logger_kwargs['env_name']}", 
                                    epoch_count=epoch,
                                    add_value_text=True
                                )
                        # Render gif
                        self.env.render(
                            path=f"{self.logger_kwargs['data_dir']}/{self.logger_kwargs['env_name']}",
                            epoch_count=epoch,
                        )                                
                    # Always render first episode
                    if self.render and epoch == 0 and self.render_first_episode:
                        for id, ac in self.agents.items():
                            if self.actor_critic_architecture == 'cnn':
                                ac.render(
                                    savepath=f"{self.logger_kwargs['data_dir']}/{self.logger_kwargs['env_name']}", 
                                    epoch_count=epoch,
                                    add_value_text=True
                                )                             
                        self.env.render(
                            path=f"{self.logger_kwargs['data_dir']}/{self.logger_kwargs['env_name']}",
                            epoch_count=epoch,
                        
                        )                              
                        self.render_first_episode = False             

                    # Always render last epoch's episode
                    if self.DEBUG and epoch == self.total_epochs-1:
                        self.env.render(
                            path=f"{self.logger_kwargs['data_dir']}/{self.logger_kwargs['env_name']}",
                            epoch_count=epoch,
                        )                        
                        for id, ac in self.agents.items():
                            if self.actor_critic_architecture == 'cnn':
                                ac.render(
                                    savepath=f"{self.logger_kwargs['data_dir']}/{self.logger_kwargs['env_name']}", 
                                    epoch_count=epoch,
                                    add_value_text=True
                                )                            

                    # Reset the environment and counters
                    if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':            
                        for id in self.agents:
                            self.stat_buffers[id].reset()
                         
                    # If not at the end of an epoch, reset hidden layers for incoming new episode    
                    if timeout and not epoch_ended: # not env.epoch_end:
                        for id, ac in self.agents.items():                        
                            hiddens[id] = ac.reset_neural_nets()
                    # Else log epoch results                    
                    else:
                        for id in self.agents:
                            # TODO this was already done above, is this being done twice?                            
                            # if 'out_of_bounds_count' in infos[id]:
                            #     out_of_bounds_count[id] += infos[id]['out_of_bounds_count'] 
                            self.loggers[id].store(DoneCount=terminal_counter[id], OutOfBound=out_of_bounds_count[id])
                            terminal_counter[id] = 0
                            out_of_bounds_count[id] = 0
                    
                    # Reset environment. Obsertvations for each agent - 11 dimensions: [intensity reading, x coord, y coord, 8 directions of distance detected to obstacle]
                    observations, _,  _, _ = self.env.reset()                                         
                    source_coordinates = np.array(self.env.src_coords, dtype="float32")  # Target for later NN update after episode concludes
                    episode_return = {id: 0 for id in self.agents}
                    steps_in_episode = 0
                    # Reset maps for new episode
                    if self.actor_critic_architecture == 'cnn':
                        _ = ac.reset_neural_nets()

                    # Update stat buffers for all agent observations for later observation normalization
                    if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':            
                        for id in self.agents:
                            self.stat_buffers[id].update(observations[id][0])                              
                    
            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.total_epochs - 1):
                for id in self.agents:
                    self.loggers[id].save_state({}, None)

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
                    stop_iteration=update_results.stop_iteration,
                    loss_policy=update_results.loss_policy,
                    loss_critic=update_results.loss_critic,
                    loss_predictor=update_results.loss_predictor,
                    kl_divergence=update_results.kl_divergence,
                    Entropy=update_results.Entropy,
                    ClipFrac=update_results.ClipFrac,
                    LocLoss=update_results.LocLoss,
                    VarExplain=update_results.VarExplain, # TODO what is this?
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
                self.loggers[id].log_tabular("loss_policy", average_only=True)
                self.loggers[id].log_tabular("loss_critic", average_only=True)
                self.loggers[id].log_tabular("loss_predictor", average_only=True)  # Specific to the regressive GRU
                self.loggers[id].log_tabular("LocLoss", average_only=True)
                self.loggers[id].log_tabular("Entropy", average_only=True)
                self.loggers[id].log_tabular("kl_divergence", average_only=True)
                self.loggers[id].log_tabular("ClipFrac", average_only=True)
                self.loggers[id].log_tabular("DoneCount", sum_only=True)
                self.loggers[id].log_tabular("OutOfBound", average_only=True)
                self.loggers[id].log_tabular("stop_iteration", average_only=True)
                self.loggers[id].log_tabular("Time", time.time() - self.start_time)                 
                self.loggers[id].dump_tabular()
