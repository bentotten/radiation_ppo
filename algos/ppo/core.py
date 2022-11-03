import numpy as np
import numpy.typing as npt
import scipy.signal
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.distributions.categorical import Categorical
from numbers import Number

from typing import (
    Any,
    Callable,
    Literal,
    NoReturn,
    TypeAlias,
    Optional,
    cast,
    overload,
    Union,
)

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


def mlp(
    sizes: list[Shape],
    activation,
    output_activation=nn.Identity,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers = []
    for j in range(len(sizes) - 1):
        layer = [nn.Linear(sizes[j], sizes[j + 1])]

        if layer_norm:
            ln = nn.LayerNorm(sizes[j + 1]) if j < len(sizes) - 1 else None
            layer.append(ln)

        layer.append(activation() if j < len(sizes) - 1 else output_activation())
        layers += layer

    if layer_norm and None in layers:
        layers.remove(None)

    return nn.Sequential(*layers)


def count_vars(module: nn.Module) -> int:
    return sum(np.prod(p.shape) for p in module.parameters())


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


@dataclass
class StatBuff:
    mu: float = 0.0
    sig_sto: float = 0.0
    sig_obs: float = 1.0
    count: int = 0

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


class PFRNNBaseCell(nn.Module):
    """parent class for PFRNNs"""

    def __init__(
        self,
        num_particles: int,
        input_size: int,
        hidden_size: int,
        resamp_alpha: float,
        use_resampling: bool,
        activation: str,
    ):
        """init function

        Arguments:
            num_particles {int} -- number of particles
            input_size {int} -- input size
            hidden_size {int} -- particle vector length
            resamp_alpha {float} -- alpha value for soft-resampling
            use_resampling {bool} -- whether to use soft-resampling
            activation {str} -- activation function to use
        """
        super().__init__()
        self.num_particles: int = num_particles
        self.samp_thresh: float = num_particles * 1.0
        self.input_size: int = input_size
        self.h_dim: int = hidden_size
        self.resamp_alpha: float = resamp_alpha
        self.use_resampling: bool = use_resampling
        self.activation: str = activation
        self.initialize: str = "rand"
        if activation == "relu":
            self.batch_norm: nn.BatchNorm1d = nn.BatchNorm1d(
                self.num_particles, track_running_stats=False
            )

    @overload
    def resampling(self, particles: Tensor, prob: Tensor) -> tuple[Tensor, Tensor]:
        ...

    @overload
    def resampling(
        self, particles: tuple[Tensor, Tensor], prob: Tensor
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        ...

    def resampling(
        self, particles: Tensor | tuple[Tensor, Tensor], prob: Tensor
    ) -> tuple[tuple[Tensor, Tensor] | Tensor, Tensor]:
        """soft-resampling

        Arguments:
            particles {tensor} -- the latent particles
            prob {tensor} -- particle weights

        Returns:
            tuple -- particles
        """

        resamp_prob = (
            self.resamp_alpha * torch.exp(prob)
            + (1 - self.resamp_alpha) * 1 / self.num_particles
        )
        resamp_prob = resamp_prob.view(self.num_particles, -1)
        flatten_indices = (
            torch.multinomial(
                resamp_prob.transpose(0, 1),
                num_samples=self.num_particles,
                replacement=True,
            )
            .transpose(1, 0)
            .contiguous()
            .view(-1, 1)
            .squeeze()
        )

        # PFLSTM
        if type(particles) == tuple:
            particles_new = (
                particles[0][flatten_indices],
                particles[1][flatten_indices],
            )
        # PFGRU
        else:
            particles_new = particles[flatten_indices]

        prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (
            self.resamp_alpha * prob_new + (1 - self.resamp_alpha) / self.num_particles
        )
        prob_new = torch.log(prob_new).view(self.num_particles, -1)
        prob_new = prob_new - torch.logsumexp(prob_new, dim=0, keepdim=True)

        return particles_new, prob_new

    def reparameterize(self, mu: Tensor, var: Tensor) -> Tensor:
        """Implements the reparameterization trick introduced in [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

        Arguments:
            mu {tensor} -- learned mean
            var {tensor} -- learned variance

        Returns:
            tensor -- sample
        """
        std: Tensor = F.softplus(var)
        eps: Tensor = torch.FloatTensor(std.shape).normal_()
        return mu + eps * std


class PFGRUCell(PFRNNBaseCell):
    def __init__(
        self,
        num_particles: int,
        input_size: int,
        obs_size: int,
        hidden_size: int,
        resamp_alpha: float,
        use_resampling: bool,
        activation: str,
    ):
        super().__init__(
            num_particles,
            input_size,
            hidden_size,
            resamp_alpha,
            use_resampling,
            activation,
        )

        self.fc_z: nn.Linear = nn.Linear(self.h_dim + self.input_size, self.h_dim)
        self.fc_r: nn.Linear = nn.Linear(self.h_dim + self.input_size, self.h_dim)
        self.fc_n: nn.Linear = nn.Linear(self.h_dim + self.input_size, self.h_dim * 2)

        self.fc_obs: nn.Linear = nn.Linear(self.h_dim + self.input_size, 1)
        self.hid_obs: nn.Sequential = mlp([self.h_dim, 24, 2], nn.ReLU)
        self.hnn_dropout: nn.Dropout = nn.Dropout(p=0)

    def forward(
        self, input_: Tensor, hx: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """One step forward for PFGRU

        Arguments:
            input_ {tensor} -- the input tensor
            hx {tuple} -- previous hidden state (particles, weights)

        Returns:
            tuple -- new tensor
        """
        h0, p0 = hx
        obs_in = input_.repeat(h0.shape[0], 1)
        obs_cat = torch.cat((h0, obs_in), dim=1)

        z = torch.sigmoid(self.fc_z(obs_cat))
        r = torch.sigmoid(self.fc_r(obs_cat))
        n = self.fc_n(torch.cat((r * h0, obs_in), dim=1))

        mu_n, var_n = torch.split(n, split_size_or_sections=self.h_dim, dim=1)
        n: Tensor = self.reparameterize(mu_n, var_n)

        if self.activation == "relu":
            # if we use relu as the activation, batch norm is require
            n = n.view(self.num_particles, -1, self.h_dim).transpose(0, 1).contiguous()
            n = self.batch_norm(n)
            n = n.transpose(0, 1).contiguous().view(-1, self.h_dim)
            n = torch.relu(n)
        elif self.activation == "tanh":
            n = torch.tanh(n)
        else:
            raise ModuleNotFoundError

        h1: Tensor = (1 - z) * n + z * h0

        p1 = self.observation_likelihood(h1, obs_in, p0)

        if self.use_resampling:
            h1, p1 = self.resampling(h1, p1)

        p1 = p1.view(-1, 1)
        mean_hid = torch.sum(torch.exp(p1) * self.hnn_dropout(h1), dim=0)
        loc_pred: Tensor = self.hid_obs(mean_hid)

        return loc_pred, (h1, p1)

    def observation_likelihood(self, h1: Tensor, obs_in: Tensor, p0: Tensor) -> Tensor:
        """observation function based on compatibility function"""
        logpdf_obs: Tensor = self.fc_obs(torch.cat((h1, obs_in), dim=1))
        p1: Tensor = logpdf_obs + p0
        p1 = p1.view(self.num_particles, -1, 1)
        p1 = F.log_softmax(p1, dim=0)
        return p1

    def init_hidden(self, batch_size: int) -> tuple[Tensor, Tensor]:
        initializer: Callable[[int, int], Tensor] = (
            torch.rand if self.initialize == "rand" else torch.zeros
        )
        h0 = initializer(batch_size * self.num_particles, self.h_dim)
        p0: Tensor = torch.ones(batch_size * self.num_particles, 1) * np.log(
            1 / self.num_particles
        )
        hidden = (h0, p0)
        return hidden


class SeqLoc(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: tuple[Shape, Shape],
        bias: bool = True,
        ln_preact: bool = True,
        weight_init: bool = False,
    ):
        super().__init__()

        self.seq_model: nn.GRU = nn.GRU(input_size, hidden_size[0][0], 1)
        self.Woms: nn.Sequential = mlp([*hidden_size[0], *hidden_size[1], 2], nn.Tanh)
        self.Woms: nn.Sequential = nn.Sequential(*(list(self.Woms.children())[:-1]))

        if weight_init:
            for m in self.named_children():
                self.weights_init(m)

        self.hs = hidden_size[0][0]

    def weights_init(self, m: tuple[str, nn.Module]) -> None:
        if isinstance(m[1], nn.Linear):
            stdv: float = 2 / math.sqrt(max(m[1].weight.size()))
            m[1].weight.data.uniform_(-stdv, stdv)
            if m[1].bias is not None:
                m[1].bias.data.uniform_(-stdv, stdv)

    def forward(
        self, x: Tensor, hidden: Tensor, ep_form=None, batch: bool = False
    ) -> tuple[Tensor, Tensor]:
        _hidden: Tensor = self.seq_model(x.unsqueeze(int(batch)), hidden)[0]
        out_arr: Tensor = self.Woms(_hidden.squeeze())
        return out_arr, _hidden

    def init_hidden(self, bs=None) -> Tensor:
        std = 1.0 / math.sqrt(self.hs)
        init_weights = torch.FloatTensor(1, 1, self.hs).uniform_(-std, std)
        return init_weights[0, :, None]


class SeqPt(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: tuple[Shape, Shape],
        bias: bool = True,
        ln_preact: bool = True,
        weight_init: bool = False,
    ):
        super().__init__()
        self.seq_model: nn.GRU = nn.GRU(input_size, hidden_size[0][0], 1)
        self.Woms: nn.Sequential = mlp([*hidden_size[0], *hidden_size[1], 8], nn.Tanh)
        self.Woms: nn.Sequential = torch.nn.Sequential(
            *(list(self.Woms.children())[:-1])
        )
        self.Valms: nn.Sequential = mlp([*hidden_size[0], *hidden_size[2], 1], nn.Tanh)
        self.Valms = torch.nn.Sequential(*(list(self.Valms.children())[:-1]))

        if weight_init:
            for m in self.named_children():
                self.weights_init(m)

        self.hs = hidden_size[0]

    def weights_init(self, m: tuple[str, nn.Module]) -> None:
        if isinstance(m[1], nn.Linear):
            stdv = 2 / math.sqrt(max(m[1].weight.size()))
            m[1].weight.data.uniform_(-stdv, stdv)
            if m[1].bias is not None:
                m[1].bias.data.uniform_(-stdv, stdv)

    def forward(
        self, x: Tensor, hidden: Tensor, ep_form=None, batch: bool = False
    ) -> tuple[Tensor, Tensor, nn.Sequential]:  # MS POMDP
        _hidden: Tensor = self.seq_model(x, hidden)[0]
        out_arr: Tensor = self.Woms(_hidden.squeeze())
        val: nn.Sequential = self.Valms(_hidden.squeeze())
        return out_arr, _hidden, val

    def _reset_state(self) -> tuple[Tensor, Literal[0]]:
        std = 1.0 / math.sqrt(self.hs)
        init_weights = torch.FloatTensor(1, self.hs).uniform_(-std, std)
        return (init_weights[0, :, None], 0)


class Actor(nn.Module):
    def _distribution(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi: nn.Module, act: nn.Module) -> NoReturn:
        raise NotImplementedError

    def forward(self, obs, act=None, hidden=None) -> Tensor:
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        # pi, hidden = self._distribution(obs,hidden) #should be [4000,5]
        pi, hidden, val = self._distribution(obs, hidden=hidden)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)  # should be [4000]
        return pi, logp_a, hidden, val


class MLPCategoricalActor(Actor):
    def __init__(
        self,
        input_dim: int,
        act_dim: int,
        hidden_sizes: Shape,
        activation: nn.Module,
        net_type: Optional[str] = None,
        batch_s: int = 1,
    ):
        super().__init__()
        if net_type == "rnn":
            self.logits_net = RecurrentNet(
                input_dim,
                act_dim,
                hidden_sizes,
                activation,
                batch_s=batch_s,
                rec_type="rnn",
            )
        else:
            self.logits_net = mlp(
                [
                    input_dim,
                    *(hidden_sizes if type(hidden_sizes) == tuple else []),
                    act_dim,
                ],
                activation,
            )

    def _distribution(self, obs, hidden=None) -> tuple[Categorical, Tensor, Tensor]:
        # logits, hidden = self.logits_net(obs, hidden=hidden)
        logits, hidden, val = self.logits_net.v_net(obs, hidden=hidden)
        return Categorical(logits=logits), hidden, val

    def _log_prob_from_distribution(self, pi: nn.Module, act: nn.Module):
        return pi.log_prob(act)

    def _reset_state(self):
        return self._get_init_states()

    def _get_init_states(self):
        std = 1.0 / math.sqrt(self.logits_net.hs)
        init_weights = torch.FloatTensor(1, 1, self.logits_net.hs).uniform_(-std, std)
        return init_weights[0, :, None]


class RecurrentNet(nn.Module):
    def __init__(
        self, obs_dim, act_dim, hidden_sizes, activation, batch_s=1, rec_type="lstm"
    ):
        super().__init__()
        self.hs = hidden_sizes[0][0]
        self.v_net = SeqPt(obs_dim // batch_s, hidden_sizes)

    def forward(self, obs, hidden, ep_form=None, meas_arr=None):
        return self.v_net(obs, hidden, ep_form=ep_form)

    def _reset_state(self):
        return self._get_init_states()

    def _get_init_states(self):
        std = 1.0 / math.sqrt(self.v_net.hs)
        init_weights = torch.FloatTensor(2, self.v_net.hs).uniform_(-std, std)
        return (init_weights[0, :, None], init_weights[1, :, None])


class RNNModelActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Shape = (32,),
        hidden_sizes_pol: Shape = (64,),
        hidden_sizes_val: Shape = (64, 64),
        hidden_sizes_rec: Shape = (64,),
        activation=nn.Tanh,
        net_type=None,
        pad_dim: int = 2,
        batch_s: int = 1,
        seed: int = 0,
    ):
        super().__init__()
        self.seed_gen: torch.Generator = torch.manual_seed(seed)
        self.hidden: int = hidden[0]
        self.pi_hs: int = hidden_sizes_rec[0]
        self.val_hs: int = hidden_sizes_val[0]
        self.bpf_hsize: int = hidden_sizes_rec[0]
        hidden_sizes: Shape = hidden + hidden_sizes_pol + hidden_sizes_val

        self.pi = MLPCategoricalActor(
            self.pi_hs if hidden_sizes_pol[0][0] == 1 else obs_dim + pad_dim,
            act_dim,
            None if hidden_sizes_pol[0][0] == 1 else hidden_sizes,
            activation,
            net_type=net_type,
            batch_s=batch_s,
        )

        self.num_particles = 40
        self.alpha = 0.7

        # self.model   = SeqLoc(obs_dim-8,[hidden_sizes_rec]+[[24]],1)
        self.model = PFGRUCell(
            self.num_particles,
            obs_dim - 8,
            obs_dim - 8,
            self.bpf_hsize,
            self.alpha,
            True,
            "tanh",
        )  # obs_dim, hidden_sizes_pol[0]

    def step(
        self, obs: npt.NDArray[np.float32], hidden: Union[tuple[tuple[Tensor, Tensor], Tensor], tuple[Tensor, Tensor]]
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        tuple[Tensor, Tensor],
        npt.NDArray[np.float32],
    ]:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            loc_pred, hidden_part = self.model(obs_t[:, :3], hidden[0]) # PFGRU
            obs_t = torch.cat((obs_t, loc_pred.unsqueeze(0)), dim=1)
            pi, hidden2, v = self.pi._distribution(obs_t.unsqueeze(0), hidden[1])
            a = pi.sample()
            logp_a: Tensor = self.pi._log_prob_from_distribution(pi, a)
            _hidden: tuple[Tensor, Tensor] = (hidden_part, hidden2)
        return a.numpy(), v.numpy(), logp_a.numpy(), _hidden, loc_pred.numpy()

    def grad_step(
        #self, obs: npt.NDArray[np.float32], act, hidden: tuple[Tensor, Tensor]
        self, obs: Tensor, act: Tensor, hidden: tuple[tuple[Tensor, Tensor], Tensor]
    #) -> tuple[Any, Any, Any, Tensor]:
    ) -> tuple[Any, Tensor, Tensor, Tensor]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(1) # TODO this is already a tensor, is this necessary?
        loc_pred = torch.empty((obs_t.shape[0], 2))
        hidden_part = hidden[0]  # TODO his contains two tensors, is that accounted for?
        with torch.no_grad():
            for kk, o in enumerate(obs_t):
                loc_pred[kk], hidden_part = self.model(o[:, :3], hidden_part)
        obs_t = torch.cat((obs_t, loc_pred.unsqueeze(1)), dim=2)
        pi, logp_a, hidden2, val = self.pi(obs_t, act=act, hidden=hidden[1])
        return pi, val, logp_a, loc_pred

    def act(
        self, obs: npt.NDArray[np.float32], hidden: tuple[Tensor, Tensor]
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        tuple[Tensor, Tensor],
        npt.NDArray[np.float32],
    ]:
        return self.step(obs, hidden=hidden)

    def reset_hidden(self, batch_size: int = 1) -> tuple[tuple[Tensor, Tensor], Tensor]:
        model_hidden = self.model.init_hidden(batch_size)
        a2c_hidden = self.pi._reset_state()
        return (model_hidden, a2c_hidden)
