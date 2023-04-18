Overview
=========

A summary of RAD-TEAM components. Click to expand. 

High Level Overview
*********************
Start points.
TODO: Add eval

.. autosummary::
   :toctree: generated

   algos.multiagent.main
   algos.multiagent.train


Radiation Simulation Environment Overview
*******************************************
The multi-agent radiation simulation environment.

.. autosummary::
   :toctree: generated

   gym_rad_search.envs.rad_search_env


Learning Overview
*********************
Updates the neural network and holds algorithm-specific processes for Proximal Policy Optimization.

.. autosummary::
   :toctree: generated

   algos.multiagent.ppo


Neural Networks Overview
*************************
Contains the Actor-Critic neural networks augmented with the Particle Filter Gated Recurrent Unit for location prediction. 

.. autosummary::
   :toctree: generated

   algos.multiagent.NeuralNetworkCores.RADTEAM_core


Auxiliary Overview
*********************

.. autosummary::
   :toctree: generated

   algos.multiagent.plot_results