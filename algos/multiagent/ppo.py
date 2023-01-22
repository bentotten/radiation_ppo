from dataclasses import dataclass, field
import torch

@dataclass
class OptimizationStorage:
    ''' 
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

