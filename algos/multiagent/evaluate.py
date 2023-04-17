'''
Evaluate agents and update neural networks using simulation environment.
'''
# TODO!!!!
DELETE_PI_AFTER_NEW_MODEL_TRAINED = False

import os
import sys
import time
import random
from datetime import datetime
import math
from statsmodels.stats.weightstats import DescrStatsW # type: ignore

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

import json
import joblib # type: ignore
import ray

# Simulation Environment
import gym  # type: ignore
from gym_rad_search.envs import rad_search_env # type: ignore
from gym_rad_search.envs.rad_search_env import RadSearch, StepResult  # type: ignore
from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore

# PPO and logger
try:
    from ppo import OptimizationStorage, PPOBuffer, AgentPPO  # type: ignore
    from ppo import BpArgs  # type: ignore

except ModuleNotFoundError:
    from algos.multiagent.ppo import AgentPPO  # type: ignore
    from algos.multiagent.ppo import BpArgs  # type: ignore    
    
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


# Helpful functions
def median(data: List)-> np.float32:
    return np.median(data) if len(data) > 0 else np.nan


def variance(data: List)-> np.float32:
    return np.var(np.array(data) / len(data)) if len(data) > 0 else np.nan


@dataclass
class Results():
    episode_length: List[int] = field(default_factory=lambda: list())
    episode_return: List[float] = field(default_factory=lambda: list())
    intensity: List[float] = field(default_factory=lambda: list())
    background_intensity: List[float] = field(default_factory=lambda: list())
    success_count: List[int] = field(default_factory=lambda: list())


@dataclass
class MonteCarloResults():
    id: int
    completed_runs: int = field(init=False, default=0)
    successful: Results = field(default_factory=lambda: Results())
    unsuccessful: Results = field(default_factory=lambda: Results())
    total_episode_length: List[int] = field(default_factory=lambda: list())
    success_counter: int = field(default=0)


@dataclass
class Metrics():
    medians: List = field(default_factory=lambda: list())
    variances: List = field(default_factory=lambda: list())


@dataclass
class Distribution():
    unique: Dict = field(default_factory=lambda: dict())
    counts: Dict = field(default_factory=lambda: dict())


# Uncomment when ready to run with Ray
@ray.remote(num_cpus=2, num_gpus=0.08) 
@dataclass
class EpisodeRunner:
    '''
        Remote function to execute requested number of episodes for requested number of monte carlo runs each episode.
        
        Process from RAD-A2C:
        - 100 episodes classes:
            - [done] create environment
            - [done] refresh environment with test env
            - [done] create and upload agent
            - [done] Get initial environment observation
            - Do monte-carlo runs
                - Get action
                - Take step in env
                - Save return and increment steps-in-episode
                - If terminal or timeout: 
                    - Save det_sto from environment (why?)
                    - If first monte-carlo:
                        - If terminal, save intensity/background intenity into "done" list
                        - If not terminal, save intensity/background intenity into "not done" list
                    - If Terminal, increment done counter, add episode length to "done" list, and add episode return to "done" list
                    - If not Terminal, add episode length to "not done" list, and add episode return to "not done" list
                    - Render if desired
                    - Refresh environment and reset episode tracking
                    - ? #Reset model in action selection fcn. get_action(0)
                    - ? #Get initial location prediction
            - Render
            - Save stats/results and return:
                mc_stats['dEpLen'] = d_ep_len
                mc_stats['ndEpLen'] = nd_ep_len
                mc_stats['dEpRet'] = d_ep_ret
                mc_stats['ndEpRet'] = nd_ep_ret
                mc_stats['dIntDist'] = done_dist_int
                mc_stats['ndIntDist'] = not_done_dist_int
                mc_stats['dBkgDist'] = done_dist_bkg
                mc_stats['ndBkgDist'] = not_done_dist_bkg
                mc_stats['DoneCount'] = np.array([done_count])
                mc_stats['TotEpLen'] = tot_ep_len
                mc_stats['LocEstErr'] = loc_est_err
                results = [loc_est_ls, FIM_bound, J_score_ls, det_ls]
                print(f'Finished episode {n}!, completed count: {done_count}')
                return (results,mc_stats)            
        
    ''' 
    id: int
    #env_sets: Dict    
    current_dir: str
    
    env_name: str
    env_kwargs: Dict
    steps_per_episode: int
    team_mode: str
    resolution_multiplier: float
    
    render: bool
    save_gif_freq: int
    render_path: str
    
    model_path: str
    save_path_for_ac: str
    test_env_path: str = field(default='./evaluation/test_environments')
    save_path: str = field(default='.')
    seed: Union[int, None] = field(default=0)
    
    obstruction_count: int = field(default=0) 
    enforce_boundaries: bool = field(default=False)
    actor_critic_architecture: str = field(default="cnn")    
    number_of_agents: int = field(default=1)
    episodes: int = field(default=100)
    montecarlo_runs: int = field(default=100)
    snr: str = field(default='high')
    
    render_first_episode: bool = field(default=True)
  
    # Initialized elsewhere
    #: Object that holds agents    
    agents: Dict[int, Union[RADCNN_core.CNNBase, RADA2C_core.RNNModelActorCritic]] = field(default_factory=lambda:dict())
    
    def __post_init__(self)-> None:
        # Change to correct directory
        os.chdir(self.current_dir)
           
        # Load test environments
        self.env_sets = joblib.load(self.test_env_path)
                
        # Create own instatiation of environment
        self.env = self.create_environment()
        
        # Get agent model paths and saved agent configurations
        agent_models = {}
        for child in os.scandir(self.model_path):
            if child.is_dir() and 'agent' in child.name:
                agent_models[int(child.name[0])] = child.path  # Read in model path by id number. NOTE: Important that ID number is the first element of file name 
            if child.is_dir() and 'general' in child.name:
                general_config_path = child.path  
        original_configs = list(json.load(open(f"{general_config_path}/config.json"))['self'].values())[0]['ppo_kwargs']['actor_critic_args']
        
        # Set up static A2C actor-critic args
        if self.actor_critic_architecture == 'cnn':
            actor_critic_args=dict(
                action_space=self.env.detectable_directions,
                observation_space=self.env.observation_space.shape[0], # Also known as state dimensions: The dimensions of the observation returned from the environment
                steps_per_episode=self.steps_per_episode,
                number_of_agents=self.number_of_agents,
                detector_step_size=self.env.step_size,
                environment_scale=self.env.scale,
                bounds_offset=self.env.observation_area,
                enforce_boundaries=self.enforce_boundaries,
                grid_bounds=self.env.scaled_grid_max,
                resolution_multiplier=self.resolution_multiplier,
                GlobalCritic=None,
                no_critic=True,
                save_path=self.save_path_for_ac
            )            
        elif self.actor_critic_architecture =='rnn':
            actor_critic_args=dict(
                obs_dim=self.env.observation_space.shape[0],
                act_dim=self.env.detectable_directions,
                hidden_sizes_pol=[[32]],
                hidden_sizes_val=[[32]],
                hidden_sizes_rec=[24],
                hidden=[[24]],
                net_type='rnn',
                batch_s=1,
                seed=self.seed,
                pad_dim=2
            )
        else:
            raise ValueError("Unsupported net type")  
          
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ######################################################################################################
        # TODO delete me after training RAD-A2C with robust seed
        if DELETE_PI_AFTER_NEW_MODEL_TRAINED:
            actor_critic_args['seed'] = 2
        ######################################################################################################
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        if self.actor_critic_architecture !='cnn':
            assert self.team_mode == 'individual' # No global critic for RAD-A2C
        
        # Check current important parameters match parameters read in 
        for arg in actor_critic_args:
            if arg != 'no_critic' and arg != 'GlobalCritic' and arg != 'save_path':
                if type(original_configs[arg]) == int or type(original_configs[arg]) == float or type(original_configs[arg]) == bool:
                    assert actor_critic_args[arg] == original_configs[arg], f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                elif type(original_configs[arg]) is str:
                    if arg == 'net_type':
                        assert actor_critic_args[arg] == original_configs[arg]
                    else:
                        to_list = original_configs[arg].strip('][').split(' ')
                        config = np.array([float(x) for x in to_list], dtype=np.float32)
                        assert np.array_equal(config, actor_critic_args[arg]), f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                elif type(original_configs[arg]) is list:
                    for a,b in zip(original_configs[arg], actor_critic_args[arg]):
                        assert a == b, f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"
                else:
                    assert actor_critic_args[arg] == original_configs[arg], f"Agent argument mismatch: {arg}.\nCurrent: {actor_critic_args[arg]}; Model: {original_configs[arg]}"             
        
        # Initialize agents and load agent models
        for i in range(self.number_of_agents):
            if self.actor_critic_architecture == 'cnn':
                self.agents[i] = RADCNN_core.CNNBase(id=i, **actor_critic_args)  # NOTE: No updates, do not need PPO
                self.agents[i].load(checkpoint_path=agent_models[i])

                # Sanity check
                assert self.agents[i].critic.is_mock_critic()                
                
            elif self.actor_critic_architecture == 'rnn':
                self.agents[i] = RADA2C_core.RNNModelActorCritic(**actor_critic_args)
                if DELETE_PI_AFTER_NEW_MODEL_TRAINED:
                    self.agents[i].pi.load_state_dict(torch.load(f"{agent_models[i]}/pyt_save/model.pt"))
                else:
                    self.agents[i].load_state_dict(torch.load(f"{agent_models[i]}/pyt_save/model.pt"))                    
            else:
                raise ValueError("Unsupported net type")

    
    def run(self)-> MonteCarloResults:
        # Prepare tracking buffers and counters
        episode_return: Dict[int, float] = {id: 0.0 for id in self.agents}
        steps_in_episode: int = 0
        terminal_counter: Dict[int, int] = {id: 0 for id in self.agents}  # Terminal counter for the epoch (not the episode)        
        run_counter = 0
        
        # Prepare results buffers
        results = MonteCarloResults(id=self.id)
        # For RAD-A2C Compatibility
        stat_buffers: Dict[int, StatisticStandardization] = dict()
        
        # Refresh environment with test env parameters
        observations = self.env.refresh_environment(env_dict=self.env_sets, id=0, num_obs=self.obstruction_count)
        
        for agent in self.agents.values(): 
            agent.set_mode('eval')
            
        # Prepare episode variables
        agent_thoughts: Dict[int, RADCNN_core.ActionChoice] = dict()            
        hiddens: Dict[int, Union[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], None]] = {id: self.agents[id].reset_hidden() for id in self.agents} # For RAD-A2C compatibility
        
        # If RAD-A2C, instatiate stat buffer and load/standardize first observation
        if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
            initial_prediction = np.zeros((3,))
            for id, ac in self.agents.items():
                stat_buffers[id] = StatisticStandardization()                
                stat_buffers[id].update(observations[id][0])   
                observations[id][0] = stat_buffers[id].standardize(observations[id][0])                                       
        
        while run_counter < self.montecarlo_runs:
            # Get agent thoughts on current state. Actor: Compute action and logp (log probability); Critic: compute state-value
            agent_thoughts.clear()
            for id, ac in self.agents.items():
                with torch.no_grad():
                    if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':                    
                        agent_thoughts[id], heatmaps = ac.step(observations[id], hiddens[id])
                    else:
                        agent_thoughts[id], heatmaps = ac.step(observations, hiddens)
                    
                hiddens[id] = agent_thoughts[id].hiddens # For RAD-A2C - save latest hiddens for use in next steps.

            # Create action list to send to environment
            agent_action_decisions = {id: int(agent_thoughts[id].action) for id in agent_thoughts}
            for action in agent_action_decisions.values():
                assert 0 <= action and action < int(self.env.number_actions)

            # Take step in environment - Note: will be missing last reward, rewards link to previous observation in env
            observations, rewards, terminals, _ = self.env.step(action=agent_action_decisions)
            
            if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                for id, ac in self.agents.items():
                    stat_buffers[id].update(observations[id][0])   
                    observations[id][0] = stat_buffers[id].standardize(observations[id][0])
            
            # Incremement Counters and save new (individual) cumulative returns
            if self.team_mode == 'individual':
                for id in rewards['individual_reward']:
                    episode_return[id] += np.array(rewards['individual_reward'][id], dtype="float32").item()
            else:
                for id in self.agents:
                    episode_return[id] += np.array(rewards['team_reward'], dtype="float32").item() # TODO if saving team reward, no need to keep duplicates for each agent
                
            steps_in_episode += 1
            
            # Tally up ending conditions
            # Check if there was a terminal state. Note: if terminals are introduced that only affect one agent but not all, this will need to be changed.
            terminal_reached_flag = False
            for id in terminal_counter:
                if terminals[id] == True and not timeout:
                    terminal_counter[id] += 1   
                    terminal_reached_flag = True
                    
            # Stopping conditions for episode
            # timeout: bool = steps_in_episode == self.steps_per_episode
            # terminal: bool = terminal_reached_flag or timeout  
               
            # Stopping conditions for episode
            timeout: bool = steps_in_episode == self.steps_per_episode # Max steps per episode reached
            episode_over: bool = terminal_reached_flag or timeout  # Either timeout or terminal found

            if episode_over:
                self.process_render(run_counter=run_counter, id=self.id)
                
                # Save results
                if run_counter < 1:
                    if terminal_reached_flag:
                        results.successful.intensity.append(self.env.intensity)
                        results.successful.background_intensity.append(self.env.bkg_intensity)
                    else:
                        results.unsuccessful.intensity.append(self.env.intensity)
                        results.unsuccessful.background_intensity.append(self.env.bkg_intensity)
                results.total_episode_length.append(steps_in_episode)
                
                if terminal_reached_flag:
                    results.success_counter += 1
                    results.successful.episode_length.append(steps_in_episode)
                    results.successful.episode_return.append(episode_return[0]) # TODO change for individual mode
                else:
                    results.unsuccessful.episode_length.append(steps_in_episode)
                    results.unsuccessful.episode_return.append(episode_return[0]) # TODO change for individual mode

                # Incremenet run counter
                run_counter +=1 
                
                # Reset environment without performing an env.reset()
                episode_return = {id: 0.0 for id in self.agents}
                steps_in_episode = 0
                terminal_counter = {id: 0 for id in self.agents}  # Terminal counter for the epoch (not the episode)
                
                observations = self.env.refresh_environment(env_dict=self.env_sets, id=0, num_obs=self.obstruction_count)
                
                # Reset stat buffer for RAD-A2C
                if self.actor_critic_architecture == 'rnn' or self.actor_critic_architecture == 'mlp':
                    for id, ac in self.agents.items():
                        stat_buffers[id].reset()
                        stat_buffers[id].update(observations[id][0])   
                        observations[id][0] = stat_buffers[id].standardize(observations[id][0])                    

                # Reset agents
                for agent in self.agents.values():
                    agent.reset()
                    
        results.completed_runs = run_counter
                                
        print(f'Finished episode {self.id}! Success count: {results.success_counter} out of {self.montecarlo_runs}')
        return results

    def create_environment(self) -> RadSearch:
        env = gym.make(self.env_name, **self.env_kwargs) 
        env.reset()
        return env
    
    def getattr(self, attr):
        return getattr(self, attr)    
    
    def say_hello(self):
        return self.id   

    def process_render(self, run_counter: int, id: int)-> None:
        # Render
        save_time_triggered = (run_counter % self.save_gif_freq == 0) if self.save_gif_freq != 0 else False
        time_to_save = save_time_triggered or ((run_counter + 1) == self.montecarlo_runs)
        if (self.render and time_to_save):
            # Render Agent heatmaps
            if self.actor_critic_architecture == 'cnn':
                for id, ac in self.agents.items():
                    # TODO add episode counter
                    ac.render(
                        savepath=self.render_path, 
                        episode_count=id,
                        epoch_count=run_counter,
                        add_value_text=True
                    )
            # Render gif           
            self.env.render(
                path=self.render_path, 
                epoch_count=run_counter,
                episode_count=id,    
                silent = True                            
            )     
            # Render environment image
            self.env.render(
                path=self.render_path, 
                epoch_count=run_counter, 
                just_env=True,
                episode_count=id,      
                silent = True                          
            )                                               
        # Always render first episode
        if self.render and run_counter == 0 and self.render_first_episode:
            # Render Agent heatmaps
            if self.actor_critic_architecture == 'cnn':
                for id, ac in self.agents.items():
                    ac.render(
                        savepath=self.render_path, 
                        epoch_count=run_counter, 
                        add_value_text=True,
                        episode_count=id,                        
                    )
            # Render gif           
            self.env.render(
                path=self.render_path, 
                epoch_count=run_counter, 
                episode_count=id,
                silent = True
            )     
            # Render environment image
            self.env.render(
                path=self.render_path, 
                epoch_count=run_counter, 
                just_env=True,
                episode_count=id, 
                silent = True                               
            )  
            self.render_first_episode = False             

        # Always render last episode
        if self.render and run_counter == self.montecarlo_runs-1: 
            # Render Agent heatmaps
            if self.actor_critic_architecture == 'cnn':
                for id, ac in self.agents.items():
                    ac.render(
                        savepath=self.render_path, 
                        epoch_count=run_counter, 
                        add_value_text=True,
                        episode_count=id,                        
                    )
            # Render gif           
            self.env.render(
                path=self.render_path, 
                epoch_count=run_counter, 
                episode_count=id,   
                silent = True                             
            )     
            # Render environment image
            self.env.render(
                path=self.render_path, 
                epoch_count=run_counter, 
                just_env=True,
                episode_count=id,  
                silent = True                              
            )  
                            

@dataclass
class evaluate_PPO:
    '''
        Test existing model across random episodes for a set number of monte carlo runs per episode.
    '''
    eval_kwargs: Dict
    
    # Initialized elsewhere
    #: Directory containing test environments. Each test environment file contains 1000 environment configurations.
    test_env_dir: str = field(init=False)
    #: Full path to file containing chosen test environment. Each test environment file contains 1000 environment configurations.
    test_env_path: str = field(init=False)    
    #: Sets of environments for specifications. Comes in sets of 1000.
    environment_sets: Dict = field(init=False)
    #: runners
    runners: Dict = field(init=False)

    def __post_init__(self)-> None:
        obs = self.eval_kwargs['obstruction_count']
        if obs == -1:
            raise ValueError("Random sample of obstruction counts indicated. Please indicate a specific count between 1 and 7")
        self.test_env_dir = self.eval_kwargs['test_env_path']
        self.test_env_path = self.test_env_dir + f"/test_env_dict_obs{self.eval_kwargs['obstruction_count']}_{self.eval_kwargs['snr']}_v4"
        self.eval_kwargs['test_env_path'] = self.test_env_path
                
        # Uncomment when ready to run with Ray   
        #Initialize ray                                     
        try:
            ray.init(address='auto', num_cpus=112, num_gpus=8)
        except:
            print("Ray failed to initialize. Running on single server.")

    def evaluate(self):
        ''' Driver '''    
        start_time = time.time()           
        #Uncomment when ready to run with Ray
        runners = {i: EpisodeRunner
                   .remote(
                        id=i, 
                        current_dir=os.getcwd(),
                        **self.eval_kwargs
                    ) 
                for i in range(self.eval_kwargs['episodes'])
            } 
                
        full_results = ray.get([runner.run.remote() for runner in runners.values()])
        print(full_results)
        
        #Uncomment when to run without Ray        
        # self.runners = {i: EpisodeRunner(
        #                 id=0, 
        #                 current_dir=os.getcwd(),
        #                 **self.eval_kwargs
        #         ) for i in range(self.eval_kwargs['episodes'])
        #     }
        # full_results = [runner.run() for runner in self.runners.values()]
        
        # print('Runtime: {}', time.time() - start_time)
        
        #self.parse_results(full_results)
        pass

    def parse_results(self, results: List):
        ''' Get the weighted median episode length and the weighted success rate for each  environment configuration (scenario) '''
        
        # Succesful objects
        successful_runs = []
        success_episode_return  = Metrics()
        success_episode_lengths  = Metrics()        
        success_intensity  = Metrics()
        success_background_intensity = Metrics()
        
        # Unsuccesful objects
        unsuccess_episode_return  = Metrics()
        unsuccess_episode_lengths  = Metrics()        
        unsuccess_intensity  = Metrics()
        unsuccess_background_intensity = Metrics()
        
        # Successful episode length object
        success_episode_len_dist = Distribution()
        
        for scenario in results:
            # Get medians and variances for successful runs
            successful_runs.append(scenario.success_counter)

            success_episode_return.medians.append( median(scenario.successful.background_intensity) )
            success_episode_return.variances.append( variance(scenario.successful.background_intensity) )

            success_episode_lengths.medians.append( median(scenario.successful.background_intensity) )
            success_episode_lengths.variances.append( variance(scenario.successful.background_intensity) )            

            success_intensity.medians.append( median(scenario.successful.background_intensity) )
            success_intensity.variances.append( variance(scenario.successful.background_intensity) )

            success_background_intensity.medians.append( median(scenario.successful.background_intensity) )
            success_background_intensity.variances.append( variance(scenario.successful.background_intensity) )

            # Get medians and variances for unsuccessful runs
            unsuccess_episode_return.medians.append( median(scenario.successful.background_intensity) )
            unsuccess_episode_return.variances.append( variance(scenario.successful.background_intensity) )

            unsuccess_episode_lengths.medians.append( median(scenario.successful.background_intensity) )
            unsuccess_episode_lengths.variances.append( variance(scenario.successful.background_intensity) )            

            unsuccess_intensity.medians.append( median(scenario.successful.background_intensity) )
            unsuccess_intensity.variances.append( variance(scenario.successful.background_intensity) )

            unsuccess_background_intensity.medians.append( median(scenario.successful.background_intensity) )
            unsuccess_background_intensity.variances.append( variance(scenario.successful.background_intensity) )
        
            # Get 'weighted median' for successful episode lengths
            unique, counts = np.unique(scenario.successful.episode_length, return_counts=True)
            sort_idx = np.argsort(counts)
            
            success_episode_len_dist.unique[scenario.id] = list()
            success_episode_len_dist.counts[scenario.id] = list()
            
            for index in sort_idx:
                success_episode_len_dist.unique[scenario.id].append(unique[index])
                success_episode_len_dist.counts[scenario.id].append(counts[index])

        for ii, key in enumerate(keys):
            if key in ['dIntDist','ndIntDist', 'dBkgDist','ndBkgDist','dEpRet','ndEpRet','ndEpLen','TotEpLen']:
                pass
            else:
                if 'LocEstErr' in key:
                    tot_mean = np.mean(stats[:,ii,0])
                    std_error = math.sqrt(np.nansum(stats[:,ii,1]/stats[:,ii,2]))
                    #print('Mean '+ key +': ' +str(np.round(tot_mean,decimals=2))+ ' +/- ' +str(np.round(std_error,3)))
                else:
                    if np.nansum(stats[:,ii,0]) > 1:
                        d1 = DescrStatsW(stats[:,ii,0], weights=stats[:,ii,2])
                        lp_w, weight_med, hp_w = d1.quantile([0.025,0.5,0.975],return_pandas=False)
                        q1, q3 = d1.quantile([0.25,0.75],return_pandas=False)
                        print('Weighted Median '+ key +': ' +str(np.round(weight_med,decimals=2))+ ' Weighted Percentiles (' +str(np.round(lp_w,3))+','+str(np.round(hp_w,3))+')')
        pass
                   
    def calc_stats(results, mc=None, plot=False, snr=None, control=None, obs=None):
        """
        Calculate results from the evaluation
        """
        
        # Metrics we care about:
        #   - Success Count: save number if greater than 0, otherwise NaN
        #       - [0.025,0.5,0.975] quantiles?
        #       - [0.25,0.75] quantiles?        
        #   - Successful Epsiode Lengths: 
        #       - unsure, but saving unique lengths and their counts to a distribution matrix?
        #       - [0.025,0.5,0.975] quantiles?
        #       - [0.25,0.75] quantiles?
        #   - All other metrics: Median/Variance/shape for epsiode length,
        
        # mc_stats['dEpLen'] = d_ep_len
        # mc_stats['ndEpLen'] = nd_ep_len
        # mc_stats['dEpRet'] = d_ep_ret
        # mc_stats['ndEpRet'] = nd_ep_ret
        # mc_stats['dIntDist'] = done_dist_int
        # mc_stats['ndIntDist'] = not_done_dist_int
        # mc_stats['dBkgDist'] = done_dist_bkg
        # mc_stats['ndBkgDist'] = not_done_dist_bkg
        # mc_stats['DoneCount'] = np.array([done_count])
        # mc_stats['TotEpLen'] = tot_ep_len
        # mc_stats['LocEstErr'] = loc_est_err
        # results = [loc_est_ls, FIM_bound, J_score_ls, det_ls]
        # print(f'Finished episode {n}!, completed count: {done_count}')
        # return (results,mc_stats)   
        # Assuming these got switched at some point?     
            
        # Results[0] only has one element that contains all 100 mc runs
        #   - each run contains two elements:
        #   [0] is [loc_est_ls, FIM_bound, J_score_ls, det_ls]
        #   [1] is mc_stats
        
        # [Mean, Variance, Size]
        stats = np.zeros( 
                        (len(results[0]), # 100
                        len(results[0][0][1]), # 11 
                        3)
                        )
        keys = results[0][0][1].keys() 
        num_elem = 101
        d_count_dist = np.zeros((len(results[0]),2,num_elem)) 
        
        for jj, data in enumerate(results[0]):
            for ii, key in enumerate(keys):
                # if 'Count' in key:
                #     stats[jj,ii,0:2] = data[1][key] if data[1][key].size > 0 else np.nan
                # elif 'LocEstErr' in key:
                #     stats[jj,ii,0] = np.mean(data[1][key]) if data[1][key].size > 0 else np.nan
                #     stats[jj,ii,1] = np.var(data[1][key])/data[1][key].shape[0] if data[1][key].size > 0 else np.nan
                # else:
                #     stats[jj,ii,0] = np.median(data[1][key]) if data[1][key].size > 0 else np.nan
                #     stats[jj,ii,1] = np.var(data[1][key])/data[1][key].shape[0] if data[1][key].size > 0 else np.nan
                # stats[jj,ii,2] = data[1][key].shape[0]
                
                if key in 'dEpLen': #and isinstance(data[0],np.ndarray):
                    uni,counts = np.unique(data[1][key],return_counts=True)
                    sort_idx = np.argsort(counts)
                    if len(sort_idx) > num_elem:
                        d_count_dist[jj,0,:] = uni[sort_idx][-num_elem:]
                        d_count_dist[jj,1,:] = counts[sort_idx][-num_elem:]
                    else:
                        d_count_dist[jj,0,num_elem-len(sort_idx):] = uni[sort_idx][-num_elem:]
                        d_count_dist[jj,1,num_elem-len(sort_idx):] = counts[sort_idx][-num_elem:]

        for ii, key in enumerate(keys):
            if key in ['dIntDist','ndIntDist', 'dBkgDist','ndBkgDist','dEpRet','ndEpRet','ndEpLen','TotEpLen']:
                pass
            else:
                if 'LocEstErr' in key:
                    tot_mean = np.mean(stats[:,ii,0])
                    std_error = math.sqrt(np.nansum(stats[:,ii,1]/stats[:,ii,2]))
                    #print('Mean '+ key +': ' +str(np.round(tot_mean,decimals=2))+ ' +/- ' +str(np.round(std_error,3)))
                else:
                    if np.nansum(stats[:,ii,0]) > 1:
                        d1 = DescrStatsW(stats[:,ii,0], weights=stats[:,ii,2])
                        lp_w, weight_med, hp_w = d1.quantile([0.025,0.5,0.975],return_pandas=False)
                        q1, q3 = d1.quantile([0.25,0.75],return_pandas=False)
                        print('Weighted Median '+ key +': ' +str(np.round(weight_med,decimals=2))+ ' Weighted Percentiles (' +str(np.round(lp_w,3))+','+str(np.round(hp_w,3))+')')

        return stats, d_count_dist
    
    
    