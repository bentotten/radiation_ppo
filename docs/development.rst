##################
In-Depth
##################

Code flows from CLI -> Main -> Train -> Radiation Environment, PPO -> Neural Networks. For clarity, this documentation starts at the furthest level (Simulation Environment and Neural Networks) and moves backwards to main.


*********
Sample H2
*********

Sample content.


**********
Another H2
**********

Sample H3
=========

Sample H4
---------

Sample H5
^^^^^^^^^

Sample H6
"""""""""



***********************
Command Line Arguments
***********************

.. autoclass:: algos.multiagent.main.CliArgs


***********************
Simulation Environment
***********************

.. autoclass:: gym_rad_search.envs.RadSearch
    :members:


****************
Neural Networks
****************

These are the compatible neural network frameworks.

RAD-TEAM Augmented Actor-Critic with Convolutional Neural Networks
====================================================================
This contains the TEAMRAD framework. See :ref:`Neural Networks Overview` for global types and variables.

Augmented Actor-Critic Model
--------------------------------------
This contains the base class, the actor (policy) class, the critic (value) class, and the particle filter gated recurrent unit class/subclass (location prediction).

.. autoclass:: algos.multiagent.NeuralNetworkCores.RADTEAM_core.CCNBase
    :members:


Map Handling 
------------------------------
Storage of maps and conversion from observations to maps.

.. autoclass:: algos.multiagent.NeuralNetworkCores.RADTEAM_core.ConversionTools
    :members:


.. autoclass:: algos.multiagent.NeuralNetworkCores.RADTEAM_core.MapsBuffer
    :members:


Intensity Sampling and Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: algos.multiagent.NeuralNetworkCores.RADTEAM_core.IntensityEstimator
    :members:


Standardizing Intensity Value and Visits Counts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: algos.multiagent.NeuralNetworkCores.RADTEAM_core.StatisticStandardization
    :members:


Normalizing Intensity Value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: algos.multiagent.NeuralNetworkCores.RADTEAM_core.Normalizer 
    :members:


Maps Buffer
^^^^^^^^^^^^


Auxiliary
-----------

ActionChoice
^^^^^^^^^^^^^^
.. autoclass:: algos.multiagent.NeuralNetworkCores.RADTEAM_core.ActionChoice
    :members:


*********
Training
*********

Train Function
===============
.. autoclass::  algos.multiagent.train.train_PPO
    :members:
    :inherited-members:


Update with Proximal Policy Optimization
=========================================

.. autoclass:: algos.multiagent.ppo.AgentPPO
    :members:


**********
Execution
**********
TODO