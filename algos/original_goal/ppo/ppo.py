import numpy as np
import torch
from torch.optim import Adam
import gym # type: ignore
import time
import os
import core # type: ignore
from ppo_tools import PPOBuffer, OptimizationStorage # type: ignore
from gym.utils.seeding import _int_list_from_bigint, hash_seed # type: ignore
from rl_tools.logx import EpochLogger # type: ignore
from rl_tools.mpi_pytorch import setup_pytorch_for_mpi, sync_params,synchronize, mpi_avg_grads, sync_params_stats # type: ignore
from rl_tools.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar,mpi_statistics_vector, num_procs, mpi_min_max_scalar # type: ignore


def ppo(env_fn, actor_critic=core.RNNModelActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, alpha=0, clip_ratio=0.2, pi_lr=3e-4, mp_mm=[5,5],
        vf_lr=5e-3, train_pi_iters=40, train_v_iters=15, lam=0.9, max_ep_len=120, save_gif=False,
        target_kl=0.07, logger_kwargs=dict(), save_freq=500, render= False, dims=None, number_of_agents=1):
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

    # Global stats buffer
    stat_buff = core.StatBuff()    

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()
    
    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    
    # Set up saver using logger class
    model_saver = [EpochLogger(**logger_kwargs) for _ in range(number_of_agents)]

    #Set Pytorch random seed
    torch.manual_seed(seed)

    # Instantiate environment
    env = env_fn()
    ac_kwargs['seed'] = seed
    ac_kwargs['pad_dim'] = 2

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape

    #Instantiate A2C
    # ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac = [actor_critic(env.observation_space, env.action_space, **ac_kwargs) for _ in range(number_of_agents)]
    
    # Sync params across processes
    for id in range(number_of_agents):
        sync_params(ac[id])

    #PFGRU args, from Ma et al. 2020
    bp_args = {
        'bp_decay' : 0.1,
        'l2_weight':1.0, 
        'l1_weight':0.0,
        'elbo_weight':1.0,
        'area_scale':env.search_area[2][1]}

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac[0].pi, ac[0].model])
    logger.log('\nNumber of parameters: \t pi: %d, model: %d \t'%var_counts)

    # Set up trajectory buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    
    #buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, ac_kwargs['hidden_sizes_rec'][0])
    
    PPObuffer = [ 
        PPOBuffer(
            observation_dimension=env.observation_space.shape[0],
            max_size=local_steps_per_epoch,
            max_episode_length=max_ep_len,
            gamma=gamma,
            lam=lam,
            number_agents=number_of_agents,
        ) for _ in range(number_of_agents)
    ]
    
    save_gif_freq = epochs // 3
    if proc_id() == 0:
        print(f'Local steps per epoch: {local_steps_per_epoch}')

    # Set up optimizers and learning rate decay for policy and localization module
    # pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    # model_optimizer = Adam(ac.model.parameters(), lr=vf_lr)
    # pi_scheduler = torch.optim.lr_scheduler.StepLR(pi_optimizer,step_size=100,gamma=0.99)
    # model_scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer,step_size=100,gamma=0.99)
    optimization = [
        OptimizationStorage(
                pi_optimizer=Adam(ac[id].pi.parameters(), 
                lr=pi_lr),
                critic_optimizer=None,
                model_optimizer=Adam(ac[id].model.parameters(), 
                lr=vf_lr),
                critic_flag=False
            ) 
        for id in range(number_of_agents)
        ]    
    loss = torch.nn.MSELoss(reduction='mean')

    # Set up model saving
    #logger.setup_pytorch_saver(ac)
    for id in range(number_of_agents):
        model_saver[id].setup_pytorch_saver(ac[id])

    def update(agent, agent_optimizer, buf, env, args, loss_fcn=loss):
        """Update for the localization and A2C modules"""
        
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
                hidden = agent.reset_hidden()
                obs, act, logp_old, adv, ret, src_tar = trajectories[:,:observation_idx], trajectories[:,action_idx],trajectories[:,logp_old_idx], \
                                                        trajectories[:,advantage_idx], trajectories[:,return_idx,None], trajectories[:,source_loc_idx:].clone()
                #Calculate new log prob.
                pi, val, logp, loc = agent.grad_step(obs, act, hidden=hidden)
                logp_diff = logp_old - logp 
                ratio = torch.exp(logp - logp_old)

                clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
                clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)

                #Useful extra info
                clipfrac = torch.as_tensor(clipped, dtype=torch.float32).detach().mean().item()
                approx_kl = logp_diff.detach().mean().item()
                ent = pi.entropy().detach().mean().item()
                val_loss = loss(val,ret)
                
                loss_arr[ii] = -(torch.min(ratio * adv, clip_adv).mean() - 0.01*val_loss + alpha * ent)
                loss_sto[ii,0] = approx_kl; loss_sto[ii,1] = ent; loss_sto[ii,2] = clipfrac; loss_sto[ii,3] = val_loss.detach()
                

            mean_loss = loss_arr.mean()
            means = loss_sto.mean(axis=0)
            loss_pi, approx_kl, ent, clipfrac, loss_val = mean_loss, means[0].detach(), means[1].detach(), means[2].detach(), means[3].detach()
            pi_info['kl'].append(approx_kl), pi_info['ent'].append(ent), pi_info['cf'].append(clipfrac), pi_info['val_loss'].append(loss_val)
            
            #Average KL across processes 
            kl = mpi_avg(pi_info['kl'][-1])
            if kl.item() < 1.5 * target_kl:
                # pi_optimizer.zero_grad() 
                agent_optimizer.pi_optimizer.zero_grad()
                loss_pi.backward()
                #Average gradients across processes
                mpi_avg_grads(agent.pi)
                # pi_optimizer.step()
                agent_optimizer.pi_optimizer.step()
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

            for jj in range(train_v_iters):
                model_loss_arr_buff.zero_()
                model_loss_arr = torch.autograd.Variable(model_loss_arr_buff)
                for ii,ep in enumerate(ep_form):
                    sl = len(ep[0])
                    hidden = agent.reset_hidden()[0]
                    src_tar =  ep[0][:,source_loc_idx:].clone()
                    src_tar[:,:2] = src_tar[:,:2]/args['area_scale']
                    obs_t = torch.as_tensor(ep[0][:,:o_idx], dtype=torch.float32)
                    loc_pred = torch.empty_like(src_tar)
                    particle_pred = torch.empty((sl,agent.model.num_particles,src_tar.shape[1]))
                    
                    bpdecay_params = np.exp(args['bp_decay'] * np.arange(sl))
                    bpdecay_params = bpdecay_params / np.sum(bpdecay_params)
                    for zz,meas in enumerate(obs_t):
                        loc, hidden = agent.model(meas, hidden)
                        particle_pred[zz] = agent.model.hid_obs(hidden[0])
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

                    particle_gt = src_tar.repeat(agent.model.num_particles, 1, 1)
                    l2_particle_loss = torch.nn.functional.mse_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params
                    l1_particle_loss = torch.nn.functional.l1_loss(particle_pred, particle_gt, reduction='none') * bpdecay_params

                    # p(y_t| \tau_{1:t}, x_{1:t}, \theta) is assumed to be a Gaussian with variance = 1.
                    # other more complicated distributions could be used to improve the performance
                    y_prob_l2 = torch.exp(-l2_particle_loss).view(agent.model.num_particles, -1, sl, 2)
                    l2_particle_loss = - y_prob_l2.mean(dim=0).log()

                    y_prob_l1 = torch.exp(-l1_particle_loss).view(agent.model.num_particles, -1, sl, 2)
                    l1_particle_loss = - y_prob_l1.mean(dim=0).log()

                    xy_l2_particle_loss = torch.mean(l2_particle_loss)
                    l2_particle_loss = xy_l2_particle_loss

                    xy_l1_particle_loss = torch.mean(l1_particle_loss)
                    l1_particle_loss = 10 * xy_l1_particle_loss

                    belief_loss = args['l2_weight'] * l2_particle_loss + args['l1_weight'] * l1_particle_loss
                    total_loss = total_loss + args['elbo_weight'] * belief_loss

                    model_loss_arr[ii] = total_loss
                
                model_loss = model_loss_arr.mean()
                # model_optimizer.zero_grad()
                agent_optimizer.model_optimizer.zero_grad()
                model_loss.backward()

                #Average gradients across the processes
                mpi_avg_grads(agent.model)
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 5)
                
                # model_optimizer.step()
                agent_optimizer.model_optimizer.step() 
            
            return model_loss
        
        ### UPDATE ###
        # data = buf.get(logger=logger)
        data = buf.get()

        #Update function if using the PFGRU, fcn. performs multiple updates per call
        agent.model.train()
        loss_mod = update_model(data, args, loss=loss_fcn)

        #Update function if using the regression GRU
        #loss_mod = update_loc_rnn(data,env,loss)

        agent.model.eval()
        min_iters = len(data['ep_form'])
        kk = 0; term = False

        # Train policy with multiple steps of gradient descent (mini batch)
        while (not term and kk < train_pi_iters):
            #Early stop training if KL-div above certain threshold
            pi_l, pi_info, term, loc_loss = update_a2c(data, env, minibatch=min_iters,iter=kk)
            kk += 1
        
        #Reduce learning rate
        agent_optimizer.pi_scheduler.step()
        agent_optimizer.model_scheduler.step()

        logger.store(StopIter=kk)

        # Log changes from update
        kl, ent, cf, loss_v = pi_info['kl'], pi_info['ent'], pi_info['cf'], pi_info['val_loss']

        logger.store(LossPi=pi_l.item(), LossV=loss_v.item(), LossModel= loss_mod.item(),
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     LocLoss=loc_loss, VarExplain=0)
    
    ############# START #############
    # Prepare for interaction with environment
    start_time = time.time()
    # o, ep_ret, ep_len, done_count, a = env.reset(), 0, 0, 0, -1
    o, _, _, _ = env.reset()
    
    numpy_shape = core.combined_shape(number_of_agents)    
    
    ep_ret = np.zeros(numpy_shape)
    ep_len = 0
    done_count = 0
    
    for id in range(number_of_agents):
        stat_buff.update(o[id][0]) # update with all agent obs
    oob = 0
    reduce_v_iters = True
    hidden = [_ for _ in range(number_of_agents)]
    
    actions = {id: None for id in range(number_of_agents)}
    state_values = np.zeros(numpy_shape)
    logprobs = np.zeros(numpy_shape)
            
    for id in range(number_of_agents):
        ac[id].model.eval()
        
    # Main loop: collect experience in env and update/log each epoch
    print(f'Proc id: {proc_id()} -> Starting main training loop!', flush=True)
    for epoch in range(epochs):

        #Reset hidden state
        for id in range(number_of_agents):
            hidden[id] = ac[id].reset_hidden()
            ac[id].pi.logits_net.v_net.eval()
            
        for t in range(local_steps_per_epoch):
            #Standardize input using running statistics per episode
            obs_std = o
            
            for id in range(number_of_agents):
                obs_std[id] = np.clip((o[id]-stat_buff.mu)/stat_buff.sig_obs,-8,8)
            
            #compute action and logp (Actor), compute value (Critic)
            # a, v, logp, hidden[id], out_pred = ac[id].step(obs_std, hidden=hidden[id])

            for id in range(number_of_agents):
                actions[id], state_values[id], logprobs[id], hidden[id], _ = ac[id].step(obs_std[id], hidden=hidden[id])

            # next_o, r, d, _ = env.step(a)
            next_o, r, terminals, _ = env.step(actions)
            
            d = True if True in terminals.values() else False
            
            for id in range(number_of_agents):
                ep_ret[id] += r["individual_reward"][id]
                
            ep_len += 1

            # buf.store(obs_std, a, r, v, logp, env.src_coords)
            for id in range(number_of_agents):
                PPObuffer[id].store(
                    obs=obs_std[id], 
                    act=actions[id],
                    rew=r["individual_reward"][id],
                    val=state_values[id],
                    logp=logprobs[id], 
                    src=env.src_coords, 
                    full_observation=obs_std, 
                    heatmap_stacks=None, 
                    terminal=d
                    )
            
            logger.store(VVals=state_values.mean())

            # Update obs (critical!)
            o = next_o

            #Update running mean and std
            for id in range(number_of_agents):
                stat_buff.update(o[id][0])

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1
            
            if terminal or epoch_ended:
                if d and not timeout:
                    done_count += 1
                # if env.oob:
                    #Log if agent went out of bounds
                    # oob += 1
                # TODO needs MARL
                if env.get_agent_outOfBounds_count(id=0) > 0:
                    # Log if agent went out of bounds
                    oob += 1
                if epoch_ended and not(terminal):
                    print(f'Warning: trajectory cut off by epoch at {ep_len} steps and time {t}.', flush=True)

                if timeout or epoch_ended:
                    # if trajectory didn't reach terminal state, bootstrap value target
                    for id in range(number_of_agents):
                        obs_std[id] = np.clip((o[id]-stat_buff.mu)/stat_buff.sig_obs,-8,8)                    
                        _, state_values[id], _, _, _ = ac[id].step(obs_std[id], hidden=hidden[id])
                    if epoch_ended:
                        #Set flag to sample new environment parameters
                        env.epoch_end = True
                else:
                    for id in range(number_of_agents):
                        state_values[id] = 0
                # buf.finish_path(v)
                for id in range(number_of_agents):
                    PPObuffer[id].GAE_advantage_and_rewardsToGO(state_values[id])
                
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret.max(), EpLen=ep_len)
                    for id in range(number_of_agents):
                        PPObuffer[id].store_episode_length(episode_length=ep_len)
                    
                if epoch_ended and render and (epoch % save_gif_freq == 0 or ((epoch + 1 ) == epochs)):
                    #Check agent progress during training
                    if proc_id() == 0 and epoch != 0:
                        print("Environment render not implemented")
                        pass
                        # env.render(save_gif=save_gif,path=logger.output_dir,epoch_count=epoch)
                
                stat_buff.reset()
                if not env.epoch_end:
                    #Reset detector position and episode tracking
                    for id in range(number_of_agents):
                        hidden[id] = ac[id].reset_hidden()
                    # o, ep_ret, ep_len, a = env.reset(), 0, 0, -1
                    o, _, _, _ = env.reset()
                    ep_ret = np.zeros(numpy_shape)
                    ep_len = 0                
                else:
                    # TODO needs MARL
                    #Sample new environment parameters, log epoch results
                    # oob += env.oob_count
                    oob += env.get_agent_outOfBounds_count(id=0)
                    logger.store(DoneCount=done_count, OutOfBound=oob)
                    done_count = 0; 
                    oob = 0
                    # o, ep_ret, ep_len, a = env.reset(), 0, 0, -1
                    o, _, _, _ = env.reset()
                    ep_ret = np.zeros(numpy_shape)
                    ep_len = 0

                for id in range(number_of_agents):
                    reading = o[id][0]
                    stat_buff.update(reading)

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            for id in range(number_of_agents):
                model_saver[id].save_state(state_dict=None, itr=id)
                pass    

        #Reduce localization module training iterations after 100 epochs to speed up training
        if reduce_v_iters and epoch > 99:
            train_v_iters = 5
            reduce_v_iters = False

        # Perform PPO update!
        for id in range(number_of_agents):
            update(agent=ac[id], agent_optimizer=optimization[id], buf=PPObuffer[id], env=env, args=bp_args, loss_fcn=loss)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
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
        logger.log_tabular('OutOfBound', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='gym_rad_search:RadSearchMulti-v1')
    parser.add_argument('--hid_gru', type=int, default=[24],help='A2C GRU hidden state size')
    parser.add_argument('--hid_pol', type=int, default=[32],help='Actor linear layer size') 
    parser.add_argument('--hid_val', type=int, default=[32],help='Critic linear layer size') 
    parser.add_argument('--hid_rec', type=int, default=[24],help='PFGRU hidden state size')
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
    parser.add_argument('--num_agents',type=int, default=1, help='Number of agents') 
    
    args = parser.parse_args()

    #Change mini-batch size, only been tested with size of 1
    args.batch = 1

    #Save directory and experiment name
    args.env_name = 'bpf'
    args.exp_name = ('loc'+str(args.hid_rec[0])+'_hid' + str(args.hid_gru[0]) + '_pol'+str(args.hid_pol[0]) +'_val'
                    +str(args.hid_val[0])+'_'+args.exp_name + f'_ep{args.epochs}'+f'_steps{args.steps_per_epoch}')
    
    init_dims = {
        "bbox": args.dims,
        "observation_area": args.area_obs,
        "obstruction_count": args.obstruct,
        "number_agents": args.num_agents,
        "enforce_grid_boundaries": False,
        "DEBUG": False
    }
    
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
    init_dims["np_random"] = rng

    #Setup logger for tracking training metrics
    from rl_tools.run_utils import setup_logger_kwargs  # type: ignore
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed,data_dir='../../models/train',env_name=args.env_name)
    
    #Run ppo training function
    ppo(lambda : gym.make(args.env,**init_dims), actor_critic=core.RNNModelActorCritic,
        ac_kwargs=dict(hidden_sizes_pol=[args.hid_pol]*args.l_pol,hidden_sizes_val=[args.hid_val]*args.l_val,
        hidden_sizes_rec=args.hid_rec, hidden=[args.hid_gru], net_type=args.net_type,batch_s=args.batch), gamma=args.gamma, alpha=args.alpha,
        seed=robust_seed, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs,dims= init_dims,
        logger_kwargs=logger_kwargs,render=False, save_gif=False)
    