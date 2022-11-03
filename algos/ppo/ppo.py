from cgitb import reset
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Type, Union
from gym_rad_search.envs import rad_search_env  # type: ignore
from gym_rad_search.envs.rad_search_env import RadSearch, Action  # type: ignore
import numpy as np
import numpy.typing as npt
import torch
from torch.optim import Adam
import torch.nn.functional as F
import time
import core
from epoch_logger import EpochLogger


# ################################## set device ##################################
# print("============================================================================================")
# # set device to cpu or cuda
# device = torch.device('cpu')
# if(torch.cuda.is_available()): 
#     device = torch.device('cuda:0') 
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")
# print("============================================================================================")


class BpArgs(NamedTuple):
    bp_decay: float
    l2_weight: float
    l1_weight: float
    elbo_weight: float
    area_scale: float


@dataclass
class PPOBuffer:
    obs_dim: core.Shape
    max_size: int

    obs_buf: npt.NDArray[np.float32] = field(init=False)
    act_buf: npt.NDArray[np.float32] = field(init=False)
    adv_buf: npt.NDArray[np.float32] = field(init=False)
    rew_buf: npt.NDArray[np.float32] = field(init=False)
    ret_buf: npt.NDArray[np.float32] = field(init=False)
    val_buf: npt.NDArray[np.float32] = field(init=False)
    source_tar: npt.NDArray[np.float32] = field(init=False)
    logp_buf: npt.NDArray[np.float32] = field(init=False)
    obs_win: npt.NDArray[np.float32] = field(init=False)
    obs_win_std: npt.NDArray[np.float32] = field(init=False)

    gamma: float = 0.99
    lam: float = 0.90
    beta: float = 0.005
    ptr: int = 0
    path_start_idx: int = 0

    """
    A buffer for storing histories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __post_init__(self):
        self.obs_buf: npt.NDArray[np.float32] = np.zeros(
            core.combined_shape(self.max_size, self.obs_dim), dtype=np.float32
        )
        self.act_buf: npt.NDArray[np.float32] = np.zeros(
            core.combined_shape(self.max_size), dtype=np.float32
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
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr, :] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.source_tar[self.ptr] = src
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

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
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self, logger: EpochLogger) -> dict[str, Union[torch.Tensor, list]]:
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

        epLens: list[int] = logger.epoch_dict["EpLen"]
        numEps = len(epLens)
        epLenTotal = sum(epLens)
        data = dict(
            obs=torch.as_tensor(self.obs_buf, dtype=torch.float32),
            act=torch.as_tensor(self.act_buf, dtype=torch.float32),
            ret=torch.as_tensor(self.ret_buf, dtype=torch.float32),
            adv=torch.as_tensor(self.adv_buf, dtype=torch.float32),
            logp=torch.as_tensor(self.logp_buf, dtype=torch.float32),
            loc_pred=torch.as_tensor(self.obs_win_std, dtype=torch.float32),
            ep_len=torch.as_tensor(epLenTotal, dtype=torch.float32),
            ep_form = []
        )

        if logger:
            epLenSize = (
                # If they're equal then we don't need to do anything
                # Otherwise we need to add one to make sure that numEps is the correct size
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


class PPO:
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Base code from OpenAI:
    https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
    """

    def __init__(
        self,
        env: RadSearch,
        logger: EpochLogger,
        actor_critic: Type[core.RNNModelActorCritic] = core.RNNModelActorCritic,
        ac_kwargs: dict[str, Any] = dict(),
        seed: int = 0,
        steps_per_epoch: int = 4000,
        epochs: int = 50,
        gamma: float = 0.99,
        alpha: float = 0,
        clip_ratio: float = 0.2,
        pi_lr: float = 3e-4,
        mp_mm: tuple[int, int] = (5, 5),
        vf_lr: float = 5e-3,
        train_pi_iters: int = 40,
        train_v_iters: int = 15,
        lam: float = 0.9,
        max_ep_len: int = 120,
        number_of_agents: int = 1,
        save_gif: bool = False,
        target_kl: float = 0.07,
        save_freq: int = 500,
        render: bool = False,
    ) -> None:
        """
        Proximal Policy Optimization (by clipping),

        with early stopping based on approximate KL

        Base code from OpenAI:
        https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

        Args:
            env : An environment satisfying the OpenAI Gym API.

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
        # Set Pytorch random seed
        torch.manual_seed(seed)

        # Instantiate environment
        ac_kwargs["seed"] = seed
        ac_kwargs["pad_dim"] = 2

        obs_dim: int = env.observation_space.shape[0]
        act_dim: int = rad_search_env.A_SIZE

        # Instantiate A2C
        self.ac = actor_critic(obs_dim, act_dim, **ac_kwargs)  # TODO make multi-agent

        # PFGRU args, from Ma et al. 2020
        bp_args = BpArgs(
            bp_decay=0.1,
            l2_weight=1.0,
            l1_weight=0.0,
            elbo_weight=1.0,
            area_scale=env.search_area[2][
                1
            ],  # retrieves the height of the created environment
        )

        # Count variables
        pi_var_count, model_var_count = (
            core.count_vars(module) for module in [self.ac.pi, self.ac.model] # TODO make multi-agent
        )
        self.logger = logger
        self.logger.log(
            f"\nNumber of parameters: \t pi: {pi_var_count}, model: {model_var_count} \t"
        )

        # Set up trajectory buffer
        self.buf = PPOBuffer(
            obs_dim=obs_dim, max_size=steps_per_epoch, gamma=gamma, lam=lam,
        )
        save_gif_freq = epochs // 3

        # Set up optimizers and learning rate decay for policy and localization module
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr) # TODO make multi-agent
        self.train_pi_iters = train_pi_iters
        self.model_optimizer = Adam(self.ac.model.parameters(), lr=vf_lr) # TODO make multi-agent
        self.pi_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pi_optimizer, step_size=100, gamma=0.99
        )
        self.train_v_iters = train_v_iters
        self.model_scheduler = torch.optim.lr_scheduler.StepLR(
            self.model_optimizer, step_size=100, gamma=0.99
        )
        self.loss = torch.nn.MSELoss(reduction="mean")
        self.clip_ratio = clip_ratio
        self.alpha = alpha
        self.target_kl = target_kl

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac) # TODO make multi-agent

        # Prepare for interaction with environment
        start_time = time.time()
        o, ep_ret, ep_len, done_count, a = env.reset()[0].state, 0, 0, 0, -1 # TODO make multi-agent
        source_coordinates = np.array(env.src_coords, dtype="float32")
        stat_buff = core.StatBuff()
        stat_buff.update(o[0])
        ep_ret_ls = []
        oob = 0
        reduce_v_iters = True
        self.ac.model.eval() # TODO make multi-agent
        # Main loop: collect experience in env and update/log each epoch
        print(f"Starting main training loop!", flush=True)
        for epoch in range(epochs):
            # Reset hidden state
            hidden = self.ac.reset_hidden() # TODO make multi-agent
            self.ac.pi.logits_net.v_net.eval() # Pylance note - this seems to call just fine # TODO make multi-agent
            for t in range(steps_per_epoch):
                # Standardize input using running statistics per episode
                obs_std = o
                obs_std[0] = np.clip((o[0] - stat_buff.mu) / stat_buff.sig_obs, -8, 8)
                # compute action and logp (Actor), compute value (Critic)
                a, v, logp, hidden, out_pred = self.ac.step(obs_std, hidden=hidden) # TODO make multi-agent
                result = env.step(action=int(a)) # TODO make multi-agent # TODO figure out how to cast a to Action()
                next_o = result[0].state
                r = np.array(result[0].reward, dtype="float32")
                d = result[0].done
                msg = result[0].error
                ep_ret += r
                ep_len += 1
                ep_ret_ls.append(ep_ret)

                self.buf.store(obs_std, a, r, v, logp, source_coordinates) # TODO make multi-agent?
                logger.store(VVals=v)

                # Update obs (critical!)
                o = next_o

                # Update running mean and std
                stat_buff.update(o[0])

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t == steps_per_epoch - 1

                if terminal or epoch_ended:
                    if d and not timeout:
                        done_count += 1
                    #if env.out_of_bounds:
                    # Artifact - TODO decouple from rad_ppo agent
                    if 'out_of_bounds' in msg and msg['out_of_bounds'] == True:
                        # Log if agent went out of bounds
                        oob += 1
                    if epoch_ended and not (terminal):
                        print(
                            f"Warning: trajectory cut off by epoch at {ep_len} steps and time {t}.",
                            flush=True,
                        )

                    if timeout or epoch_ended:
                        # if trajectory didn't reach terminal state, bootstrap value target
                        obs_std[0] = np.clip(
                            (o[0] - stat_buff.mu) / stat_buff.sig_obs, -8, 8
                        )
                        _, v, _, _, _ = self.ac.step(obs_std, hidden=hidden) # TODO make multi-agent
                        if epoch_ended:
                            # Set flag to sample new environment parameters
                            env.epoch_end = True # TODO make multi-agent?
                    else:
                        v = 0
                    self.buf.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        logger.store(EpRet=ep_ret, EpLen=ep_len)

                    if (
                        epoch_ended
                        and render
                        and (epoch % save_gif_freq == 0 or ((epoch + 1) == epochs))
                    ):
                        # Check agent progress during training
                        if epoch != 0:
                            env.render(
                                save_gif=save_gif,
                                path=str(logger.output_dir),
                                epoch_count=epoch,
                                #ep_rew=ep_ret_ls,
                            )

                    ep_ret_ls = []
                    stat_buff.reset()
                    if not env.epoch_end: # TODO make multi-agent
                        # Reset detector position and episode tracking
                        hidden = self.ac.reset_hidden() # TODO make multi-agent
                        o, ep_ret, ep_len, a = env.reset()[0].state, 0, 0, -1 # TODO make multi-agent
                        source_coordinates = np.array(env.src_coords, dtype="float32")
                    else:
                        # Sample new environment parameters, log epoch results
                        if 'out_of_bounds_count' in msg:
                            oob += msg['out_of_bounds_count']
                        logger.store(DoneCount=done_count, OutOfBound=oob)
                        done_count = 0
                        oob = 0
                        o, ep_ret, ep_len, a = env.reset()[0].state, 0, 0, -1 # TODO make multi-agent
                        source_coordinates = np.array(env.src_coords, dtype="float32")
                        
                    stat_buff.update(o[0])

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({}, None)
                pass

            # Reduce localization module training iterations after 100 epochs to speed up training
            if reduce_v_iters and epoch > 99:
                self.train_v_iters = 5
                reduce_v_iters = False

            # Perform PPO update!
            self.update(env, bp_args) # TODO make multi-agent

            # Log info about epoch
            self.logger.log_tabular("Epoch", epoch)
            self.logger.log_tabular("EpRet", with_min_and_max=True)
            self.logger.log_tabular("EpLen", average_only=True)
            self.logger.log_tabular("VVals", with_min_and_max=True)
            self.logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
            self.logger.log_tabular("LossPi", average_only=True)
            self.logger.log_tabular("LossV", average_only=True)
            self.logger.log_tabular("LossModel", average_only=True)
            self.logger.log_tabular("LocLoss", average_only=True)
            self.logger.log_tabular("Entropy", average_only=True)
            self.logger.log_tabular("KL", average_only=True)
            self.logger.log_tabular("ClipFrac", average_only=True)
            self.logger.log_tabular("DoneCount", sum_only=True)
            self.logger.log_tabular("OutOfBound", average_only=True)
            self.logger.log_tabular("StopIter", average_only=True)
            self.logger.log_tabular("Time", time.time() - start_time)
            self.logger.dump_tabular()

    def update_a2c(
        self, data: dict[str, torch.Tensor], env: RadSearch, minibatch: int, iter: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], bool, torch.Tensor]:
        observation_idx = 11
        action_idx = 14
        logp_old_idx = 13
        advantage_idx = 11
        return_idx = 12
        source_loc_idx = 15

        ep_form = data["ep_form"]
        pi_info = dict(kl=[], ent=[], cf=[], val=np.array([]), val_loss=[])
        ep_select = np.random.choice(
            np.arange(0, len(ep_form)), size=int(minibatch), replace=False
        )
        ep_form = [ep_form[idx] for idx in ep_select]
        loss_sto: torch.Tensor = torch.tensor([], dtype=torch.float32)
        loss_arr: torch.Tensor = torch.autograd.Variable(
            torch.tensor([], dtype=torch.float32)
        )

        for ep in ep_form:
            # For each set of episodes per process from an epoch, compute loss
            trajectories = ep[0]
            hidden = self.ac.reset_hidden() # TODO make multi-agent
            obs, act, logp_old, adv, ret, src_tar = (
                trajectories[:, :observation_idx],
                trajectories[:, action_idx],
                trajectories[:, logp_old_idx],
                trajectories[:, advantage_idx],
                trajectories[:, return_idx, None],
                trajectories[:, source_loc_idx:].clone(),
            )
            # Calculate new log prob.
            pi, val, logp, loc = self.ac.grad_step(obs, act, hidden=hidden) # TODO make multi-agent
            logp_diff: torch.Tensor = logp_old - logp
            ratio = torch.exp(logp - logp_old)

            clip_adv = (
                torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            )
            clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)

            # Useful extra info
            clipfrac = (
                torch.as_tensor(clipped, dtype=torch.float32).detach().mean().item()
            )
            approx_kl = logp_diff.detach().mean().item()
            ent = pi.entropy().detach().mean().item()
            val_loss = self.loss(val, ret)

            # TODO: More descriptive name
            new_loss: torch.Tensor = -(
                torch.min(ratio * adv, clip_adv).mean()
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
            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            self.pi_optimizer.step()
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
            (env.search_area[2][1] * loc - (src_tar)).square().mean().sqrt(),
        )

    def update(self, env: RadSearch, args: BpArgs) -> None:
        """Update for the localization and A2C modules"""
        data: dict[str, torch.Tensor] = self.buf.get(self.logger)

        # Update function if using the PFGRU, fcn. performs multiple updates per call
        self.ac.model.train() # TODO make multi-agent
        loss_mod = self.update_model(data, args)

        # Update function if using the regression GRU
        # loss_mod = update_loc_rnn(data,env,loss)

        self.ac.model.eval() # TODO make multi-agent
        min_iters = len(data["ep_form"])
        kk = 0
        term = False

        # Train policy with multiple steps of gradient descent (mini batch)
        while not term and kk < self.train_pi_iters:
            # Early stop training if KL-div above certain threshold
            pi_l, pi_info, term, loc_loss = self.update_a2c(data, env, min_iters, kk)
            kk += 1

        # Reduce learning rate
        self.pi_scheduler.step()
        self.model_scheduler.step()

        self.logger.store(StopIter=kk)

        # Log changes from update
        kl, ent, cf, loss_v = (
            pi_info["kl"],
            pi_info["ent"],
            pi_info["cf"],
            pi_info["val_loss"],
        )

        self.logger.store(
            LossPi=pi_l.item(),
            LossV=loss_v.item(),
            LossModel=loss_mod.item(),
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            LocLoss=loc_loss,
            VarExplain=0,
        )

    #  def update_loc_rnn(self, data, env: RadSearch, loss):
    #     """Update for the simple regression GRU"""
    #     ep_form = data["ep_form"]
    #     model_loss_arr_buff = torch.zeros((len(ep_form), 1), dtype=torch.float32)
    #     for jj in range(train_v_iters):
    #         model_loss_arr_buff.zero_()
    #         model_loss_arr = torch.autograd.Variable(model_loss_arr_buff)
    #         for ii, ep in enumerate(ep_form):
    #             hidden = ac.model.init_hidden(1)
    #             src_tar = ep[0][:, 15:].clone()
    #             src_tar[:, :2] = src_tar[:, :2] / env.search_area[2][1]
    #             obs_t = torch.as_tensor(ep[0][:, :3], dtype=torch.float32)
    #             loc_pred, _ = ac.model(obs_t, hidden, batch=True)
    #             model_loss_arr[ii] = loss(loc_pred.squeeze(), src_tar.squeeze())

    #         model_loss = model_loss_arr.mean()
    #         model_optimizer.zero_grad()
    #         model_loss.backward()
    #         torch.nn.utils.clip_grad_norm_(ac.model.parameters(), 5)
    #         model_optimizer.step()

    #     return model_loss

    def update_model(self, data: dict[str, torch.Tensor], args: BpArgs) -> torch.Tensor:
        # Update the PFGRU, see Ma et al. 2020 for more details
        ep_form = data["ep_form"]
        source_loc_idx = 15
        o_idx = 3

        for _ in range(self.train_v_iters):
            model_loss_arr: torch.Tensor = torch.autograd.Variable(
                torch.tensor([], dtype=torch.float32)
            )
            for ep in ep_form:
                sl = len(ep[0])
                hidden = self.ac.reset_hidden()[0] # TODO make multi-agent
                #src_tar: npt.NDArray[np.float32] = ep[0][:, source_loc_idx:].clone()
                src_tar: torch.Tensor = ep[0][:, source_loc_idx:].clone()
                src_tar[:, :2] = src_tar[:, :2] / args.area_scale
                obs_t = torch.as_tensor(ep[0][:, :o_idx], dtype=torch.float32)
                loc_pred = torch.empty_like(src_tar)
                particle_pred = torch.empty(
                    (sl, self.ac.model.num_particles, src_tar.shape[1]) # TODO make multi-agent
                )

                bpdecay_params = np.exp(args.bp_decay * np.arange(sl))
                bpdecay_params = bpdecay_params / np.sum(bpdecay_params)
                for zz, meas in enumerate(obs_t):
                    loc, hidden = self.ac.model(meas, hidden) # TODO make multi-agent
                    particle_pred[zz] = self.ac.model.hid_obs(hidden[0]) # TODO make multi-agent
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

                particle_gt = src_tar.repeat(self.ac.model.num_particles, 1, 1) # TODO make multi-agent
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
                    self.ac.model.num_particles, -1, sl, 2 # TODO make multi-agent
                )
                l2_particle_loss = -y_prob_l2.mean(dim=0).log()

                y_prob_l1 = torch.exp(-l1_particle_loss).view(
                    self.ac.model.num_particles, -1, sl, 2 # TODO make multi-agent
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
            self.model_optimizer.zero_grad()
            model_loss.backward()
            # Clip gradient TODO should 5 be a variable?
            # TODO Pylance error: https://github.com/Textualize/rich/issues/1523. Unable to resolve
            torch.nn.utils.clip_grad_norm_(self.ac.model.parameters(), 5) # TODO make multi-agent

            self.model_optimizer.step()

        return model_loss
