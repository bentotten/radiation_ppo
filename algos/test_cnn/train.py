# type: ignore
''' OG: ppo.py '''

import numpy as np
import torch
from torch.optim import Adam
import gym # type: ignore
import time
import os

from gym.utils.seeding import _int_list_from_bigint, hash_seed # type: ignore

from rl_tools.logx import EpochLogger # type: ignore
from rl_tools.mpi_pytorch import setup_pytorch_for_mpi, sync_params,synchronize, mpi_avg_grads, sync_params_stats # type: ignore
from rl_tools.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar,mpi_statistics_vector, num_procs, mpi_min_max_scalar # type: ignore

from ppo import PPOBuffer as NEWPPO
from ppo import OptimizationStorage, AgentPPO, BpArgs

import RADTEAM_core as core

TEST_OPTIMIZER = False 
TEST_PPO = True  


def compare_dicts(dict1, dict2):
    """ Recursively compare all the values of objects """
    if type(dict1) != type(dict2):
        #return False
        raise Exception
    
    elif isinstance(dict1, dict):
        if dict1.keys() != dict2.keys():
            #return False
            raise Exception
        for key in dict1.keys():
            if not compare_dicts(dict1[key], dict2[key]):
                #return False
                raise Exception
        return True    
       
    elif isinstance(dict1, np.ndarray):
        return (dict1 == dict2).all()
    
    elif isinstance(dict1, list):
        for element1, element2 in zip(dict1, dict2):
            if not compare_dicts(element1, element2):
                #return False
                raise Exception
        return True  
             
    elif isinstance(dict1, tuple):
        for element1, element2 in zip(dict1, dict2):
            if not compare_dicts(element1, element2):
                #return False
                raise Exception
        return True         
    else:
        if isinstance(dict1, torch.Tensor) and isinstance(dict2, torch.Tensor):
            return torch.equal(dict1, dict2)
        else:
            if dict1 != dict2:
                raise Exception
            return True


def ppo(env_fn, actor_critic=core.CNNBase, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, alpha=0, clip_ratio=0.2, pi_lr=3e-4, mp_mm=[5,5],
        vf_lr=5e-3, train_pi_iters=40, train_v_iters=15, lam=0.9, max_ep_len=120, save_gif=False,
        target_kl=0.07, logger_kwargs=dict(), save_freq=500, render= False,dims=None):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Base code from OpenAI: 
    https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

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


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()
    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    #Set Pytorch random seed
    torch.manual_seed(seed)

    # Instantiate environment
    env = env_fn()
    
    #PFGRU args, from Ma et al. 2020
    bp_args = {
        'bp_decay' : 0.1,
        'l2_weight':1.0, 
        'l1_weight':0.0,
        'elbo_weight':1.0,
        'area_scale':env.search_area[2][1]
        }    
    
    # Setup args for actor-critic and prediction module (pfgru)
    dict(
            hidden_sizes_pol=[args.hid_pol]*args.l_pol,hidden_sizes_val=[args.hid_val]*args.l_val,
            hidden_sizes_rec=args.hid_rec, hidden=[args.hid_gru], net_type=args.net_type,batch_s=args.batch
            )
    
    #ac_kwargs['seed'] = seed
    #ac_kwargs['pad_dim'] = 2
    ac_kwargs['id'] = 0
    ac_kwargs['action_space'] = env.detectable_directions # Usually 8
    ac_kwargs['observation_space']= env.observation_space.shape[0]  # Also known as state dimensions: The dimensions of the observation returned from the environment. Usually 11
    ac_kwargs['detector_step_size'] = env.step_size # Usually 100 cm
    ac_kwargs['environment_scale'] = env.scale
    ac_kwargs['bounds_offset'] = env.observation_area
    ac_kwargs['grid_bounds'] = env.scaled_grid_max       

    #Instantiate A2C
    ac = actor_critic(**ac_kwargs)
        
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.critic, ac.model])
    logger.log('\nNumber of parameters: \t actior: %d, critic: %d, prediction model: %d \t'%var_counts)

    # Set up trajectory buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    
    new_buffer = NEWPPO(
                observation_dimension = env.observation_space.shape[0],
                max_size = local_steps_per_epoch,
                max_episode_length = max_ep_len,
                gamma = gamma,
                lam = lam,
                number_agents = 1
            )
    
    save_gif_freq = epochs // 3
    if proc_id() == 0:
        print(f'Local steps per epoch: {local_steps_per_epoch}')

    def update_a2c(data, env_sim, minibatch=None,iter=None):
        observation_idx = 11
        action_idx = 14
        logp_old_idx = 13
        advantage_idx = 11
        return_idx = 12
        source_loc_idx = 15
        
        ep_form= data['ep_form']
        pi_info = dict(kl=[], ent=[], cf=[], val= np.array([]), val_loss = [])
        ep_select = np.random.choice(np.arange(0,len(ep_form)),size=int(minibatch),replace=False)
        ep_form = [ep_form[idx] for idx in ep_select]
        loss_sto = torch.zeros((len(ep_form),4),dtype=torch.float32)
        loss_arr_buff = torch.zeros((len(ep_form),1),dtype=torch.float32)
        loss_arr = torch.autograd.Variable(loss_arr_buff)

        for ii,ep in enumerate(ep_form):
            #For each set of episodes per process from an epoch, compute loss 
            trajectories = ep[0]
            
            hidden = ac.reset_hidden()
            obs, act, logp_old, adv, ret, src_tar = trajectories[:,:observation_idx], trajectories[:,action_idx],trajectories[:,logp_old_idx], \
                                                     trajectories[:,advantage_idx], trajectories[:,return_idx,None], trajectories[:,source_loc_idx:].clone()
            #Calculate new log prob.
            pi, val, logp, loc = ac.grad_step(obs, act, hidden=hidden)
            logp_diff = logp_old - logp 
            ratio = torch.exp(logp - logp_old)

            clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
            clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)

            #Useful extra info
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).detach().mean().item()
            approx_kl = logp_diff.detach().mean().item()
            ent = pi.entropy().detach().mean().item()
            val_loss = loss(val,ret)
            
            if TEST_OPTIMIZER:
                assert optimization.MSELoss(val, ret) == val_loss
            
            loss_arr[ii] = -(torch.min(ratio * adv, clip_adv).mean() - 0.01*val_loss + alpha * ent)
            loss_sto[ii,0] = approx_kl; loss_sto[ii,1] = ent; loss_sto[ii,2] = clipfrac; loss_sto[ii,3] = val_loss.detach()
            

        mean_loss = loss_arr.mean()
        means = loss_sto.mean(axis=0)
        loss_pi, approx_kl, ent, clipfrac, loss_val = mean_loss, means[0].detach(), means[1].detach(), means[2].detach(), means[3].detach()
        pi_info['kl'].append(approx_kl), pi_info['ent'].append(ent), pi_info['cf'].append(clipfrac), pi_info['val_loss'].append(loss_val)
        
        #Average KL across processes 
        kl = mpi_avg(pi_info['kl'][-1])
        if kl.item() < 1.5 * target_kl:
            
            if TEST_OPTIMIZER:
                state1 = pi_optimizer.state_dict() 
                state2 = optimization.pi_optimizer.state_dict()          
                assert compare_dicts(state1, state2)
            
            pi_optimizer.zero_grad() 
            
            if TEST_OPTIMIZER:
                optimization.pi_optimizer.zero_grad()
            
            loss_pi.backward()
            #Average gradients across processes
            mpi_avg_grads(ac.pi)
            pi_optimizer.step()
            
            if TEST_OPTIMIZER:
                optimization.pi_optimizer.step()
            
                state1 = pi_optimizer.state_dict() 
                state2 = optimization.pi_optimizer.state_dict()          
                assert compare_dicts(state1, state2)
            
            term = False
        else:
            term = True
            if proc_id() == 0:
                logger.log('Terminated at %d steps due to reaching max kl.'%iter)

        pi_info['kl'], pi_info['ent'], pi_info['cf'], pi_info['val_loss'] = pi_info['kl'][0].numpy(), pi_info['ent'][0].numpy(), pi_info['cf'][0].numpy(), pi_info['val_loss'][0].numpy()
        loss_sum_new = loss_pi
        return loss_sum_new, pi_info, term, (env_sim.search_area[2][1]*loc-(src_tar)).square().mean().sqrt()

    
    def update_model(data, args, loss=None):
        #Update the PFGRU, see Ma et al. 2020 for more details
        ep_form= data['ep_form']
        model_loss_arr_buff = torch.zeros((len(ep_form),1),dtype=torch.float32)
        source_loc_idx = 15
        o_idx = 3

        if TEST_OPTIMIZER: 
            assert optimization.train_pfgru_iters == train_v_iters

        for jj in range(train_v_iters):
            model_loss_arr_buff.zero_()
            model_loss_arr = torch.autograd.Variable(model_loss_arr_buff)
            for ii,ep in enumerate(ep_form):
                sl = len(ep[0])
                hidden = ac.reset_hidden()[0]
                src_tar =  ep[0][:,source_loc_idx:].clone()
                src_tar[:,:2] = src_tar[:,:2]/args['area_scale']
                obs_t = torch.as_tensor(ep[0][:,:o_idx], dtype=torch.float32)
                loc_pred = torch.empty_like(src_tar)
                particle_pred = torch.empty((sl,ac.model.num_particles,src_tar.shape[1]))
                
                bpdecay_params = np.exp(args['bp_decay'] * np.arange(sl))
                bpdecay_params = bpdecay_params / np.sum(bpdecay_params)
                for zz,meas in enumerate(obs_t):
                    loc, hidden = ac.model(meas, hidden)
                    particle_pred[zz] = ac.model.hid_obs(hidden[0])
                    loc_pred[zz,:] = loc

                bpdecay_params = torch.FloatTensor(bpdecay_params)
                bpdecay_params = bpdecay_params.unsqueeze(-1)
                l2_pred_loss = torch.nn.functional.mse_loss(loc_pred.squeeze(), src_tar.squeeze(), reduction='none') * bpdecay_params
                l1_pred_loss = torch.nn.functional.l1_loss(loc_pred.squeeze(), src_tar.squeeze(), reduction='none') * bpdecay_params
                
                l2_loss = torch.sum(l2_pred_loss)
                l1_loss = 10*torch.mean(l1_pred_loss)

                pred_loss = args['l2_weight'] * l2_loss + args['l1_weight'] * l1_loss

                total_loss = pred_loss
                particle_pred = particle_pred.transpose(0, 1).contiguous()

                particle_gt = src_tar.repeat(ac.model.num_particles, 1, 1)
                l2_particle_loss = torch.nn.functional.mse_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params
                l1_particle_loss = torch.nn.functional.l1_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params

                # p(y_t| \tau_{1:t}, x_{1:t}, \theta) is assumed to be a Gaussian with variance = 1.
                # other more complicated distributions could be used to improve the performance
                y_prob_l2 = torch.exp(-l2_particle_loss).view(ac.model.num_particles, -1, sl, 2)
                l2_particle_loss = - y_prob_l2.mean(dim=0).log()

                y_prob_l1 = torch.exp(-l1_particle_loss).view(ac.model.num_particles, -1, sl, 2)
                l1_particle_loss = - y_prob_l1.mean(dim=0).log()

                xy_l2_particle_loss = torch.mean(l2_particle_loss)
                l2_particle_loss = xy_l2_particle_loss

                xy_l1_particle_loss = torch.mean(l1_particle_loss)
                l1_particle_loss = 10 * xy_l1_particle_loss

                belief_loss = args['l2_weight'] * l2_particle_loss + args['l1_weight'] * l1_particle_loss
                total_loss = total_loss + args['elbo_weight'] * belief_loss

                model_loss_arr[ii] = total_loss
            
            model_loss = model_loss_arr.mean()
            
            if TEST_OPTIMIZER:            
                state1 = model_optimizer.state_dict() 
                state2 = optimization.model_optimizer.state_dict()          
                assert compare_dicts(state1, state2)
                                        
            model_optimizer.zero_grad()
            
            if TEST_OPTIMIZER:
                optimization.model_optimizer.zero_grad()
            
            model_loss.backward()

            #Average gradients across the processes
            mpi_avg_grads(ac.model)
            torch.nn.utils.clip_grad_norm_(ac.model.parameters(), 5)
            
            model_optimizer.step() 
            
            if TEST_OPTIMIZER:
                optimization.model_optimizer.step()
                state1 = model_optimizer.state_dict() 
                state2 = optimization.model_optimizer.state_dict()          
                assert compare_dicts(state1, state2)
                                    
        
        return model_loss
    
    # OG
    # Set up optimizers and learning rate decay for policy and localization module
    # pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    # model_optimizer = Adam(ac.model.parameters(), lr=vf_lr)
    # pi_scheduler = torch.optim.lr_scheduler.StepLR(pi_optimizer,step_size=100,gamma=0.99)
    # model_scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer,step_size=100,gamma=0.99)
    # loss = torch.nn.MSELoss(reduction='mean')    
    
    optimization = OptimizationStorage(
                train_pi_iters=train_pi_iters,
                train_v_iters=None,
                train_pfgru_iters=train_v_iters,
                pi_optimizer=Adam(ac.pi.parameters(), lr=pi_lr),
                critic_optimizer=None, 
                model_optimizer=Adam(ac.model.parameters(), lr=vf_lr),
                MSELoss=torch.nn.MSELoss(reduction="mean"),
            )

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(env, args, loss_fcn):
        """Update for the localization and A2C modules"""

        data = new_buffer.get()
    
        #Update function if using the PFGRU, fcn. performs multiple updates per call
        ac.model.train()
        loss_mod = update_model(data, args, loss=loss_fcn)

        ac.model.eval()
        min_iters = len(data['ep_form'])
        kk = 0; term = False

        # Train policy with multiple steps of gradient descent (mini batch)
        if TEST_OPTIMIZER: 
            assert optimization.train_pi_iters == train_pi_iters
            
        while (not term and kk < train_pi_iters):
            #Early stop training if KL-div above certain threshold
            pi_l, pi_info, term, loc_loss = update_a2c(data, env, minibatch=min_iters,iter=kk)
            kk += 1
        
        if TEST_OPTIMIZER:
            state1 = pi_scheduler.state_dict() 
            state2 = optimization.pi_scheduler.state_dict()          
            assert compare_dicts(state1, state2)
            state1 = model_scheduler.state_dict() 
            state2 = optimization.pfgru_scheduler.state_dict()          
            assert compare_dicts(state1, state2)                    
        
        #Reduce learning rate
        pi_scheduler.step()
        model_scheduler.step()
        
        if TEST_OPTIMIZER:
            optimization.pi_scheduler.step()
            optimization.pfgru_scheduler.step()

            state1 = pi_scheduler.state_dict() 
            state2 = optimization.pi_scheduler.state_dict()          
            assert compare_dicts(state1, state2)
            state1 = model_scheduler.state_dict() 
            state2 = optimization.pfgru_scheduler.state_dict()          
            assert compare_dicts(state1, state2)            

        logger.store(StopIter=kk)

        # Log changes from update
        kl, ent, cf, loss_v = pi_info['kl'], pi_info['ent'], pi_info['cf'], pi_info['val_loss']

        logger.store(LossPi=pi_l.item(), LossV=loss_v.item(), LossModel= loss_mod.item(),
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     LocLoss=loc_loss, VarExplain=0)

    # Prepare for interaction with environment
    start_time = time.time()
    o, _, _, _ = env.reset()
    o = o[0]
    ep_ret, ep_len, done_count, a = 0, 0, 0, -1
    
    ep_ret_ls = []
    oob = 0
    reduce_v_iters = True
    
    ac.set_mode('eval')
    if not TEST_PPO:
        ac.model.eval()
        assert ac.model.training == ac_ppo.agent.model.training
    
    # Main loop: collect experience in env and update/log each epoch
    print(f'Proc id: {proc_id()} -> Starting main training loop!', flush=True)
    for epoch in range(epochs):
        #Reset hidden state
        hidden = ac.reset_hidden()
        
        # # If not testing PPO, save the og hiddens here
        # if not TEST_PPO:
        #     torch.save(hidden, f'./hidden/{HIDDEN_SAVES}')
        #     HIDDEN_SAVES += 1
        # # Else, ensure matches og parameters
        # else:
        #     to_test = torch.load(f'./hidden/{HIDDEN_SAVES}')
        #     assert compare_dicts(to_test, hidden)   
        #     HIDDEN_SAVES += 1                     
        
        if not TEST_PPO:    
            ac.pi.logits_net.v_net.eval()                
            assert ac.pi.logits_net.v_net.training == ac_ppo.agent.pi.logits_net.v_net.training
        
        for t in range(local_steps_per_epoch):
            # Artifact - standardization done inside CNN
            obs_std = o
            
            #compute action and logp (Actor), compute value (Critic)
            if TEST_PPO:
                a, v, logp, hidden, out_pred = ac.step({0: obs_std}, hidden=hidden)   
                
                # # Test against saved values
                # to_test = torch.load(f'./hidden/{HIDDEN_SAVES}')
                # assert compare_dicts(to_test, [a, v, logp, hidden, out_pred])   
                # HIDDEN_SAVES += 1      
                                            
            if not TEST_PPO:
                a, v, logp, hidden, out_pred = ac.step(obs_std, hidden=hidden)
                
                # # Save values
                # torch.save([a, v, logp, hidden, out_pred], f'./hidden/{HIDDEN_SAVES}')
                # HIDDEN_SAVES += 1                                         
                
            next_o, r, d, _ = env.step({0: a})
            next_o, r, d = next_o[0], r['individual_reward'][0], d[0]
            ep_ret += r
            ep_len += 1
            ep_ret_ls.append(ep_ret)

            if not TEST_PPO:
                new_buffer.store(obs=obs_std, act=a, rew=r, val=v, logp=logp, src=env.src_coords, full_observation={0: obs_std}, heatmap_stacks=None, terminal=d)
            
            if TEST_PPO:
                ac.store(obs=obs_std, act=a, rew=r, val=v, logp=logp, src=env.src_coords, full_observation={0: obs_std}, heatmap_stacks=None, terminal=d)     
            
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            # Ending conditions
            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1
            
            if terminal or epoch_ended:
                if d and not timeout:
                    done_count += 1
                if env.get_agent_outOfBounds_count(id=0) > 0:
                    #Log if agent went out of bounds
                    oob += 1
                if epoch_ended and not(terminal):
                    print(f'Warning: trajectory cut off by epoch at {ep_len} steps and time {t}.', flush=True)

                if timeout or epoch_ended:
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if TEST_PPO:
                        _, v, _, _, _ = ac.step({0: obs_std}, hidden=hidden)                                                        
                    
                    if not TEST_PPO:
                        _, v, _, _, _ = ac.step(obs_std, hidden=hidden)             
                    
                    if epoch_ended:
                        #Set flag to sample new environment parameters
                        env.epoch_end = True
                else:
                    v = 0
                    
                if not TEST_PPO:
                    new_buffer.GAE_advantage_and_rewardsToGO(v)
                
                if TEST_PPO:
                    ac.GAE_advantage_and_rewardsToGO(v)
                
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    if not TEST_PPO:
                        new_buffer.store_episode_length(episode_length=ep_len)
                    
                    if TEST_PPO:
                        ac.store_episode_length(episode_length=ep_len)

                if epoch_ended and render and (epoch % save_gif_freq == 0 or ((epoch + 1 ) == epochs)):
                    #Check agent progress during training
                    if proc_id() == 0 and epoch != 0:
                        env.render(
                            save_gif=save_gif,
                            path=logger.output_dir,
                            epoch_count=epoch,
                            episode_rewards=ep_ret_ls
                            )                
                ep_ret_ls = []
                if not env.epoch_end:
                    #Reset detector position and episode tracking
                    hidden = ac.reset_hidden()
                    
                    # If not testing PPO, save the og state here
                    # if not TEST_PPO:
                    #     torch.save(hidden, f'./hidden/{HIDDEN_SAVES}')
                    #     HIDDEN_SAVES += 1
                    # # Else, ensure matches og parameters
                    # else:
                    #     to_test = torch.load(f'./hidden/{HIDDEN_SAVES}')
                    #     assert compare_dicts(to_test, hidden)   
                    #     HIDDEN_SAVES += 1     
                        
                    o, _, _, _ = env.reset()
                    o = o[0]
                    ep_ret, ep_len, a = 0, 0, -1                    
                else:
                    #Sample new environment parameters, log epoch results
                    oob += env.get_agent_outOfBounds_count(id=0)
                    logger.store(DoneCount=done_count, OutOfBound=oob)
                    done_count = 0; oob = 0
                    o, _, _, _ = env.reset()
                    o = o[0]
                    ep_ret, ep_len, a = 0, 0, -1

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state(None, None)
            pass

        
        #Reduce localization module training iterations after 100 epochs to speed up training
        if TEST_PPO:
            if epoch > 99:                    
                ac.reduce_pfgru_training()
        if not TEST_PPO:
            if reduce_v_iters and epoch > 99:        
                train_v_iters = 5
                reduce_v_iters = False        
       
        # Perform PPO update!
        if TEST_PPO:
            results = ac.update_agent()
            
            logger.store(
                StopIter=results.stop_iteration,
                LossPi= results.loss_policy,
                LossV= results.loss_critic,
                LossModel = results.loss_predictor,
                KL=results.kl_divergence, 
                Entropy= results.Entropy, 
                ClipFrac= results.ClipFrac,
                LocLoss= results.LocLoss, 
                VarExplain=0
                )            
        if not TEST_PPO:
            update(env, bp_args, loss_fcn=loss)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('OutOfBound', average_only=True)     
        logger.log_tabular('StopIter', average_only=True)           
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('LossModel', average_only=True)
        logger.log_tabular('LocLoss', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('DoneCount', sum_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='gym_rad_search:RadSearchMulti-v1')
    parser.add_argument('--hid_gru', type=int, default=[24],help='A2C GRU hidden state size')
    parser.add_argument('--hid_pol', type=int, default=[32],help='Actor linear layer size') 
    parser.add_argument('--hid_val', type=int, default=[32],help='Critic linear layer size') 
    parser.add_argument('--hid_rec', type=int, default=24,help='PFGRU hidden state size')
    parser.add_argument('--l_pol', type=int, default=1,help='Number of layers for Actor MLP')
    parser.add_argument('--l_val', type=int, default=1,help='Number of layers for Critic MLP')
    parser.add_argument('--gamma', type=float, default=0.99,help='Reward attribution for advantage estimator')
    parser.add_argument('--seed', '-s', type=int, default=2,help='Random seed control')
    parser.add_argument('--cpu', type=int, default=1,help='Number of cores/environments to train the agent with')
    parser.add_argument('--steps_per_epoch', type=int, default=480,help='Number of timesteps per epoch per cpu. Default is equal to 4 episodes per cpu per epoch.')      
    parser.add_argument('--epochs', type=int, default=3000,help='Number of epochs to train the agent')
    parser.add_argument('--exp_name', type=str,default='alpha01_tkl07_val01_lam09_npart40_lr3e-4_proc10_obs-1_iter40_blr5e-3_2_tanh',help='Name of experiment for saving')
    parser.add_argument('--dims', type=list, default=[[0.0,0.0],[2700.0,0.0],[2700.0,2700.0],[0.0,2700.0]],
                        help='Dimensions of radiation source search area in cm, decreased by area_obs param. to ensure visilibity graph setup is valid.')
    parser.add_argument('--area_obs', type=list, default=[200.0,500.0], help='Interval for each obstruction area in cm')
    parser.add_argument('--obstruct', type=int, default=-1, 
                        help='Number of obstructions present in each episode, options: -1 -> random sampling from [1,5], 0 -> no obstructions, [1-7] -> 1 to 7 obstructions')
    parser.add_argument('--net_type',type=str, default='rnn', help='Choose between recurrent neural network A2C or MLP A2C, option: rnn, mlp') 
    parser.add_argument('--alpha',type=float,default=0.1, help='Entropy reward term scaling') 
    args = parser.parse_args()

    #Change mini-batch size, only been tested with size of 1
    args.batch = 1

    #Save directory and experiment name
    args.env_name = 'bpf'
    args.exp_name = (args.exp_name)
    init_dims = {
        'bbox':args.dims,
        'observation_area':args.area_obs, 
        'obstruction_count':args.obstruct,
        "number_agents": 1, # TODO change for MARL
        "enforce_grid_boundaries": False
        }
    
    save_dir_name: str = args.exp_name
    timestamp = "og_plus_cnn_test" #datetime.now().replace(microsecond=0).strftime("%Y-%m-%d-%H:%M:%S")
    exp_name = timestamp # + "_" + exp_name
    save_dir_name = save_dir_name + "/" + timestamp    

    max_ep_step = 120
    if args.cpu > 1:
        #max cpus, steps in batch must be greater than the max eps steps times num. of cpu
        tot_epoch_steps = args.cpu * args.steps_per_epoch
        args.steps_per_epoch = tot_epoch_steps if tot_epoch_steps > args.steps_per_epoch else args.steps_per_epoch
        print(f'Sys cpus (avail, using): ({os.cpu_count()},{args.cpu}), Steps set to {args.steps_per_epoch}')
        mpi_fork(args.cpu)  # run parallel code with mpi
    
    #Generate a large random seed and random generator object for reproducibility
    robust_seed = _int_list_from_bigint(hash_seed((1+proc_id())*args.seed))[0]
    rng = np.random.default_rng(robust_seed)
    init_dims['np_random'] = rng

    #Setup logger for tracking training metrics
    from rl_tools.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed,data_dir='../../models/train',env_name=args.env_name)
    
    # CNN args
    ac_kwargs = dict(
        steps_per_episode = max_ep_step,
        number_of_agents = 1,        
        enforce_boundaries = True,
        resolution_multiplier = 0.01,
        GlobalCritic = None,
        save_path = [f"../../models/train/{save_dir_name}", f"{exp_name}_s{args.seed}"],
        predictor_hidden_size = args.hid_rec 
        )    
    
    #Run ppo training function
    ppo(
        lambda : gym.make(args.env,**init_dims), 
        #actor_critic=core.RNNModelActorCritic,
        actor_critic= core.CNNBase,
        ac_kwargs= ac_kwargs,
        gamma=args.gamma, 
        alpha=args.alpha,
        seed=robust_seed, 
        steps_per_epoch=args.steps_per_epoch, 
        epochs=args.epochs,
        dims= init_dims,
        logger_kwargs=logger_kwargs,
        render=False, 
        save_gif=False
        )
    