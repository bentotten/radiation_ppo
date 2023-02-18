Development
------------
Code flows from CLI -> Main -> Train -> Radiation Environment, PPO -> Neural Networks


Command Line Arguments
=======================

.. autoclass:: algos.multiagent.main.CliArgs


Train
======

.. autoclass::  algos.multiagent.train.train_PPO
    :members:
    :inherited-members:


Simulation Environment
=======================

.. autoclass:: gym_rad_search.envs.RadSearch
    :members:


Proximal Policy Optimization
=============================

.. autoclass:: algos.multiagent.ppo.AgentPPO
    :members:


Neural Networks
================

.. autoclass:: algos.multiagent.NeuralNetworkCores.CNN_core.CCNBase
    :members: