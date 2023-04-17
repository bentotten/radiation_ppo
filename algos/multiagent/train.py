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
    from rl_tools.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads # type: ignore
    from rl_tools.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs # type: ignore
except ModuleNotFoundError:
    from algos.multiagent.ppo import OptimizationStorage, PPOBuffer, AgentPPO  # type: ignore
    from algos.multiagent.epoch_logger import EpochLogger, EpochLoggerKwargs, setup_logger_kwargs, convert_json
    from algos.multiagent.rl_tools.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads # type: ignore
    from algos.multiagent.rl_tools.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs # type: ignore
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
    :param actor_critic_architecture: (string) Short-version indication for what neural network core to use for actor-critic agent. Defaults to 'cnn' for convolutional neural networks.
    :param global_critic_flag: [bool] Indicate if a global critic will be set after agent intialization
    :param steps_per_epoch: (int) Number of steps of interaction (state-action pairs) for the agent and the environment in each epoch before updating the neural network modules.
    :param steps_per_episode: (int) Number of steps of interaction (state-action pairs) for the agent and the environment in each episode before resetting the environment.
    :param total_epochs: (int) Number of total epochs of interaction (equivalent to number of policy updates) to perform.
    :param render: (bool) Indicates whether to render last episode
    :param save_path: (str) Parent path to save configuration and models to. NOTE: Logs and progress still being saved with Logger and logger kwargs.
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
    global_critic_flag: bool = field(default=True)

    # Simulation parameters
    steps_per_epoch: int = field(default= 480)
    steps_per_episode: int = field(default= 120)
    total_epochs: int = field(default= 3000)

    # Rendering information
    render: bool = field(default= False)
    save_path: str = field(default = '.')
    save_freq: int = field(default= 500)
    save_gif_freq: Union[int, float] = field(default_factory= lambda:  float('inf'))
    save_gif: bool = field(default= False)
    render_first_episode: bool = field(default=True)
    episode_count: int = field(default=0)

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
    #: Global Critic for Centralized Training scenarios
    GlobalCritic: Union[RADCNN_core.Critic, None] = field(default=None)
    #: Global Optimizer for Critic for Centralized training scenarios
    GlobalCriticOptimizer: Union[torch.optim.Optimizer, None] = field(default=None)

    def __post_init__(self)-> None:
        if self.actor_critic_architecture != 'cnn' and (self.global_critic_flag or self.GlobalCritic or self.GlobalCriticOptimizer):
            raise ValueError("Global critic not supported in RAD-A2C")
        # Set Pytorch random seed
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # Save configuration
        config_json: Dict[str, Any] = convert_json(locals())

        # Set up parent directory logger and save initial configurations
        parent_kwargs = setup_logger_kwargs(
                exp_name=f"general",
                seed=self.logger_kwargs['seed'],
                data_dir=self.logger_kwargs['data_dir'],
                env_name=self.logger_kwargs['env_name']
            )
        self.parent_logger = EpochLogger(**(parent_kwargs))
        self.parent_logger.save_config(config_json)

        # Instatiate loggers
        for id in range(self.number_of_agents):
            logger_kwargs_set: Dict = setup_logger_kwargs(
                exp_name=f"{id}_agent_{self.logger_kwargs['exp_name']}",
                seed=self.logger_kwargs['seed'],
                data_dir=self.logger_kwargs['data_dir'],
                env_name=self.logger_kwargs['env_name']
            )

            self.loggers[id] = EpochLogger(**(logger_kwargs_set))

        # Initialize Global Critic
        if self.global_critic_flag:
            prototype = RADCNN_core.CNNBase(id=0, **self.ppo_kwargs['actor_critic_args'])
            self.GlobalCritic = RADCNN_core.Critic(map_dim=prototype.get_map_dimensions(), batches=prototype.get_batch_size(), map_count=prototype.get_critic_map_count())
            self.GlobalCriticOptimizer = Adam(self.GlobalCritic.parameters(), lr=self.ppo_kwargs['critic_learning_rate'])

            self.ppo_kwargs['actor_critic_args']['GlobalCritic'] = self.GlobalCritic
            self.ppo_kwargs['GlobalCriticOptimizer'] = self.GlobalCriticOptimizer

        # Initialize agents
        for i in range(self.number_of_agents):
            # If RAD-A2C, set up statistics buffers
            if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                self.stat_buffers[i] = StatisticStandardization()

            self.agents[i] = AgentPPO(id=i, **self.ppo_kwargs)
            
            if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                self.loggers[i].setup_pytorch_saver(self.agents[i].agent) # RAD-TEAM uses own function

            # Sanity check
            if self.global_critic_flag:
                assert self.agents[i].agent.critic is self.GlobalCritic
                assert self.agents[i].GlobalCriticOptimizer is self.GlobalCriticOptimizer
            elif self.actor_critic_architecture == 'cnn':
                assert self.agents[i].agent.critic is not self.GlobalCritic
                if i > 0:
                    assert self.agents[i].agent.critic is not self.agents[i-1].agent.critic
                    assert not self.agents[i].GlobalCriticOptimizer and not self.GlobalCriticOptimizer

        # Special function to avoid certain slowdowns from PyTorch + MPI combo. Even when MPI is not applied, this speeds up pytorch significantly
        setup_pytorch_for_mpi()

        for agent in self.agents.values():
            # Sync params across processes
            sync_params(agent.agent.pi)

            if not self.actor_critic_architecture == 'rnn' and not self.actor_critic_architecture == 'mlp':
                sync_params(agent.agent.critic)
            #sync_params(agent.agent.model) # TODO: Add when PFGRU up

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
        observations, _,  _, infos = self.env.reset() # Obsertvations for each agent, 11 dimensions: [intensity reading, x coord, y coord, 8 directions of distance detected to obstacle]

        # Prepare environment variables and reset
        source_coordinates: npt.NDArray = np.array(self.env.src_coords, dtype="float32")  # Target for later NN update after episode concludes
        episode_return: Dict[int, float] = {id: 0.0 for id in self.agents}
        steps_in_episode: int = 0

        # Prepare epoch variables
        out_of_bounds_count: Dict[int, int] = {id: 0 for id in self.agents} # Out of Bounds counter for the epoch (not the episode)
        terminal_counter: Dict[int, int] = {id: 0 for id in self.agents}  # Terminal counter for the epoch (not the episode)
        hiddens: Dict[int, Union[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], None]] = {id: None for id in self.agents} # For RAD-A2C compatibility

        # Prepare episode variables
        agent_thoughts: Dict[int, RADCNN_core.ActionChoice] = dict()

        # For RAD-A2C - Update stat buffers for all agent observations for later observation normalization
        if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
            for id in self.agents:
                self.stat_buffers[id].update(observations[id][0])
                self.agents[id].agent.model.eval() # Sets PFGRU model into "eval" mode

        print(f"Starting main training loop!", flush=True)
        self.start_time: float = time.time()

        # For a total number of epochs, Agent will choose an action using its policy and send it to the environment to take a step in it, yielding a new state observation.
        #   Agent will continue doing this until the episode concludes; a check will be done to see if Agent is at the end of an epoch or not - if so, the agent will use
        #   its buffer to update/train its networks. Sometimes an epoch ends mid-episode.
        print(f'Proc id: {proc_id()} -> Starting main training loop!', flush=True)
        for epoch in range(self.total_epochs):

            # For RAD-A2C - Reset hidden layers and sets Actor into "eval" mode.
            if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                for id, ac in self.agents.items():
                    ac.agent.pi.logits_net.v_net.eval()
                    hiddens[id] = ac.reset_hidden()

            # Start epoch! Episodes end when steps_per_episode is reached, steps_per_epoch is reached, or a terminal state is found
            for steps_in_epoch in range(self.steps_per_epoch):

                # For RAD-A2C - Standardize prior observation of radiation intensity for the actor-critic input using running statistics per episode. This is done within RAD-TEAMs CNN framework.
                if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                    for id in self.agents:
                        observations[id][0] = self.stat_buffers[id].standardize(observations[id][0])

                # Get agent thoughts on current state. Actor: Compute action and logp (log probability); Critic: compute state-value
                agent_thoughts.clear()
                for id, ac in self.agents.items():
                    agent_thoughts[id], heatmaps = ac.step(observations=observations, hiddens = hiddens[id], message=infos)
                    hiddens[id] = agent_thoughts[id].hiddens # For RAD-A2C - save latest hiddens for use in next steps.

                # Create action list to send to environment
                agent_action_decisions = {id: int(agent_thoughts[id].action) for id in agent_thoughts}
                for action in agent_action_decisions.values():
                    assert 0 <= action and action < int(self.env.number_actions)

                # Take step in environment - Critical that this value is saved as "next" observation so we can link rewards from this new state to the prior step/action
                next_observations, rewards, terminals, infos = self.env.step(action=agent_action_decisions)

                # Incremement Counters and save new cumulative returns
                if not self.global_critic_flag:
                    for id in rewards['individual_reward']:
                        episode_return[id] += np.array(rewards['individual_reward'][id], dtype="float32").item()
                else:
                    for id in self.agents:
                        episode_return[id] += np.array(rewards['team_reward'], dtype="float32").item()
                steps_in_episode += 1

                # Check if some agents went out of bounds
                for id in infos:
                    if 'out_of_bounds' in infos[id] and infos[id]['out_of_bounds'] == True:
                        out_of_bounds_count[id] += 1

                # Check if there was a terminal state. Note: if terminals are introduced that only affect one agent but not all, this will need to be changed.
                terminal_reached_flag = False
                for id in terminal_counter:
                    if terminals[id] == True:
                        terminal_counter[id] += 1
                        terminal_reached_flag = True

                # Stopping conditions for episode
                timeout: bool = steps_in_episode == self.steps_per_episode # Max steps per episode reached
                episode_over: bool = terminal_reached_flag or timeout  # Either timeout or terminal found
                epoch_ended: bool = steps_in_epoch == (self.steps_per_epoch - 1) # Max steps per epoch reached
                episode_reset_next_step: bool = episode_over or epoch_ended  # Either the episode is over or the episode has been cutoff by the max steps per epoch

                # Store previous observations in buffers
                for id, ac in self.agents.items():
                    # Check if global or individual reward
                    if not self.global_critic_flag:
                        reward = rewards['individual_reward'][id]
                    else:
                        reward = rewards['team_reward']

                    # Store in PPO Buffer
                    ac.ppo_buffer.store(
                        obs = observations[id],
                        rew = reward,
                        act = agent_action_decisions[id],
                        val = agent_thoughts[id].state_value,
                        logp = agent_thoughts[id].action_logprob,
                        src = source_coordinates,
                        terminal = episode_reset_next_step,
                        heatmap_stacks= heatmaps,
                        full_observation = observations
                    )

                    # Store in logger
                    self.loggers[id].store(VVals=agent_thoughts[id].state_value)

                    # RAD-A2C - update mean/std for the next observation in stat buffers,record state values with logger
                    if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                        self.stat_buffers[id].update(next_observations[id][0])

                # Update observation (critical!)
                assert observations is not next_observations, 'Previous step observation is pointing to next observation already.'
                observations = next_observations

                ############################################################################################################
                # Check for episode end
                if episode_reset_next_step:
                    if proc_id() == 0:
                        self.process_render(epoch_ended=epoch_ended, epoch=epoch)

                    if epoch_ended and not (episode_over):
                        print(f"Warning: trajectory cut off by epoch at {steps_in_episode} steps and step count {steps_in_epoch}.", flush=True)

                    # Conduct end-of-episode (and potentially end of epoch) activities.
                    self.episode_count += 1

                    # If the epoch is over and the agent didn't reach terminal state, bootstrap value target with standardized observation using per episode running statistics.
                    # ^ In english, this means use the state-value to estimate the next reward, as state-value is just a prediction of such. Because there is no terminal state,
                    # we're short one reward value for training.
                    if timeout or epoch_ended:
                        # For RAD-A2C - Standardize prior observation of radiation intensity for the actor-critic input using running statistics per episode. This is done within RAD-TEAMs CNN framework.
                        if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                            for id in self.agents:
                                observations[id][0] = self.stat_buffers[id].standardize(observations[id][0])

                        # Get prediction of next reward to bootstrap with. Because the rewards are applied to the actions taken prior, this means our last action will be without a reward. This estimate is
                        # used in that place.
                        for id, ac in self.agents.items():
                            bootstrap_results, _ = ac.step(observations, hiddens=hiddens[id])
                            last_state_value = bootstrap_results.state_value

                        if epoch_ended:
                            # Set flag to reset/sample new environment parameters. If epoch has not ended, keep training on the same environment.
                            self.env.epoch_end = True
                    else:
                        # State value. This should be 0 if the trajectory ended because the agent reached a terminal state (found source/timeout) [for use in the GAE() function]
                        last_state_value = 0

                    # Finish the trajectory and compute advantages.
                    for id, ac in self.agents.items():
                        ac.ppo_buffer.GAE_advantage_and_rewardsToGO(last_state_value)

                    # If the episode is over, save episode returns and episode length.
                    if episode_over:
                        for id, ac in self.agents.items():
                            self.loggers[id].store(EpRet=episode_return[id], EpLen=steps_in_episode)
                            ac.ppo_buffer.store_episode_length(episode_length=steps_in_episode)

                    # Reset the environment and counters
                    if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                        for id in self.agents:
                            self.stat_buffers[id].reset()

                    # For RAD-A2C - If not at the end of an epoch, reset RAD-A2C agents for incoming new episode
                    if not epoch_ended: # not env.epoch_end:
                        if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                            for id, ac in self.agents.items():
                                hiddens[id] = ac.reset_hidden()
                    # If at the end of an epoch, log epoch results and reset counters
                    else:
                        for id in self.agents:
                            self.loggers[id].store(DoneCount=terminal_counter[id], OutOfBound=out_of_bounds_count[id])
                            terminal_counter[id] = 0
                            out_of_bounds_count[id] = 0

                    # Reset environment and RAD-TEAM agents
                    observations, _,  _, _ = self.env.reset()
                    source_coordinates = np.array(self.env.src_coords, dtype="float32")  # Target for later NN update after episode concludes
                    episode_return = {id: 0 for id in self.agents}
                    steps_in_episode = 0

                    # Reset agent maps for new episode
                    if self.actor_critic_architecture == 'cnn':
                        for id, ac in self.agents.items():
                            ac.reset_agent()

                    # RAD-A2C Update stat buffers for all agent observations for later observation normalization
                    if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                        for id in self.agents:
                            self.stat_buffers[id].update(observations[id][0])
            ############################################################################################################

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.total_epochs - 1):
                for id, agent in self.agents.items():
                    if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                        self.loggers[id].save_state(None, None)
                    else:
                        test = self.loggers[id].output_dir
                        agent.save(path=test)

            # Reduce localization module training iterations after 100 epochs to speed up training
            if epoch > 99:
                for ac in self.agents.values():
                    ac.reduce_pfgru_training()

            # Perform PPO update!
            for id, ac in self.agents.items():
                # Note: Global critic is updated by first agent within update_agent
                update_results = ac.update_agent(self.loggers[id])

                # Store results
                if self.global_critic_flag:
                    if id == 0:
                        loss_critic = update_results.loss_critic
                    self.loggers[id].store(
                        stop_iteration=update_results.stop_iteration,
                        loss_policy=update_results.loss_policy,
                        loss_critic=loss_critic,
                        loss_predictor=update_results.loss_predictor,
                        kl_divergence=update_results.kl_divergence,
                        Entropy=update_results.Entropy,
                        ClipFrac=update_results.ClipFrac,
                        LocLoss=update_results.LocLoss,
                        VarExplain=update_results.VarExplain, 
                    )
                else:
                    self.loggers[id].store(
                        stop_iteration=update_results.stop_iteration,
                        loss_policy=update_results.loss_policy,
                        loss_critic=update_results.loss_critic,
                        loss_predictor=update_results.loss_predictor,
                        kl_divergence=update_results.kl_divergence,
                        Entropy=update_results.Entropy,
                        ClipFrac=update_results.ClipFrac,
                        LocLoss=update_results.LocLoss,
                        VarExplain=update_results.VarExplain, 
                    )

            if not episode_over:
                pass

            # Log info about epoch
            for id in self.agents:
                self.loggers[id].log_tabular("AgentID", id)
                self.loggers[id].log_tabular("Epoch", epoch)
                self.loggers[id].log_tabular("VVals", with_min_and_max=True)
                self.loggers[id].log_tabular("TotalEnvInteracts", (epoch + 1) * self.steps_per_epoch)
                self.loggers[id].log_tabular("loss_policy", average_only=True)
                self.loggers[id].log_tabular("loss_critic", average_only=True)
                self.loggers[id].log_tabular("loss_predictor", average_only=True)  # Specific to the regressive GRU
                self.loggers[id].log_tabular("LocLoss", average_only=True)
                self.loggers[id].log_tabular("Entropy", average_only=True)
                self.loggers[id].log_tabular("kl_divergence", average_only=True)
                self.loggers[id].log_tabular("ClipFrac", average_only=True)
                self.loggers[id].log_tabular("OutOfBound", average_only=True)
                self.loggers[id].log_tabular("stop_iteration", average_only=True)
                self.loggers[id].log_tabular("EpRet", with_min_and_max=True)                
                self.loggers[id].log_tabular("DoneCount", sum_only=True)
                self.loggers[id].log_tabular("EpLen", average_only=True)                
                self.loggers[id].log_tabular("Time", time.time() - self.start_time)
                self.loggers[id].dump_tabular()

    def process_render(self, epoch_ended: bool, epoch: int)-> None:
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
                        episode_count=self.episode_count,
                        add_value_text=True
                    )
            # Render gif
            self.env.render(
                path=f"{self.logger_kwargs['data_dir']}/{self.logger_kwargs['env_name']}",
                epoch_count=epoch,
                episode_count=self.episode_count,
            )
        # Always render first episode
        elif self.render and epoch == 0 and self.render_first_episode:
            for id, ac in self.agents.items():
                if self.actor_critic_architecture == 'cnn':
                    ac.render(
                        savepath=f"{self.logger_kwargs['data_dir']}/{self.logger_kwargs['env_name']}",
                        epoch_count=epoch,
                        episode_count=self.episode_count,
                        add_value_text=True
                    )
            self.env.render(
                path=f"{self.logger_kwargs['data_dir']}/{self.logger_kwargs['env_name']}",
                epoch_count=epoch,
                episode_count=self.episode_count,
            )
            self.render_first_episode = False
        # Always render last epoch's episode
        elif self.DEBUG and epoch == self.total_epochs-1:
            self.env.render(
                path=f"{self.logger_kwargs['data_dir']}/{self.logger_kwargs['env_name']}",
                epoch_count=epoch,
                episode_count=self.episode_count,
            )
            for id, ac in self.agents.items():
                if self.actor_critic_architecture == 'cnn':
                    ac.render(
                        savepath=f"{self.logger_kwargs['data_dir']}/{self.logger_kwargs['env_name']}",
                        epoch_count=epoch,
                        add_value_text=True,
                        episode_count=self.episode_count,
                    )
