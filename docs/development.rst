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

Convolutional Neural Networks (CNN)
====================================
This contains the CNN framework. See :ref:`Neural Networks Overview` for global types and variables.

RAD-TEAM Augmented Actor-Critic Model
--------------------------------------
This contains the base class, the actor (policy) class, the critic (value) class, and the particle filter gated recurrent unit class/subclass (location prediction).

.. autoclass:: algos.multiagent.NeuralNetworkCores.CNN_core.CCNBase
    :members:


Observation to Map Processing 
------------------------------
This contains tools for estimating the true radiation intensity value, standardizing it to one standard deviation from other samples, and normalizing it.

.. autoclass:: algos.multiagent.NeuralNetworkCores.CNN_core.ConversionTools
    :members:


Intensity Sampling and Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: algos.multiagent.NeuralNetworkCores.CNN_core.IntensityEstimator
    :members:


Standardizing Intensity Value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: algos.multiagent.NeuralNetworkCores.CNN_core.StatisticStandardization
    :members:


Normalizing Intensity Value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: algos.multiagent.NeuralNetworkCores.CNN_core.Normalizer 
    :members:



Maps Buffer
^^^^^^^^^^^^


Auxiliary
-----------

ActionChoice
^^^^^^^^^^^^^^
.. autoclass:: algos.multiagent.NeuralNetworkCores.CNN_core.ActionChoice
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