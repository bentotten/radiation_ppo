# Radiation Source Search Environment

This contains the multi-agent radiation localization architecture created by Totten et al., based on the multi-agent architecture created by Alagha et al. and their published [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167739X22002266).

This also contains the radiation source search environment based on the source search environment created by Proctor et al. and their published [paper](https://www.mdpi.com/2673-4362/2/4/29).

This contains the single-agent radiation source search architecture created by Proctor et al. and their published [paper](https://www.mdpi.com/2673-4362/2/4/29).

Below is a demo of a test episode where the trained deep reinforcement learning agent is controlling a radiation detector to search for a gamma radiation source in a non-convex environment (7 obstructions).

![Radiation Source Search - Animated gif demo](demo/demo.gif)

The obstructions (gray rectangles) block line of sight between the detector and gamma source resulting in the detector only measuring background radiation. The left plot shows the detector positions (black triangles) in the environment, the agent's source location prediction (magenta circles), and the gamma source (red star). The middle plot shows the measured gamma radiation intensity at each timestep and the right plot show the cumulative reward that the agent receives from its selected actions during an episode that is used during training to update the neural network weights. The episode terminates if the detector comes within 1.1 m of the gamma source (success) or if the episode length reaches the episode max.

# Algorithms

## Multi-Agent PPO

This repository contains the multi-agent radiation localization architecture based on [RAD-A2C](https://github.com/peproctor/radiation_ppo).

## PPO

This contains the RAD-A2C architecture and proximal policy optimization (PPO) for radiation source search from our Base code from OpenAI's [Spinningup](https://github.com/openai/spinningup) repo.

## Simple PPO

This repository provides a Minimal PyTorch implementation of Proximal Policy Optimization (PPO) with clipped objective from Nikhil Barhate's [PPO-PyTorch] (https://github.com/nikhilbarhate99/PPO-PyTorch) repo.

## Installation

It is recommended to use the Anaconda package manager. The author did all development with [Micromamba](https://mamba.readthedocs.io/en/latest/installation.html), a fast and light-weight implementation of Anaconda. To use Micromamba, simply replace `conda` commands with `micromamba`. To do a command-line install, simple add `-c conda-forge` or `-c pytorch` or another channel to specify the correct channel.

1. Clone repository.

2. Create a virtual environment with the required packages `conda env create -f <PATH-TO-ALGORTIHM>/environment.yml`.

3. Activate this environment with `conda activate <ENV_NAME>`.

4. \*\* Note: The RAD-A2C implementation requires [OpenMPI](https://www.open-mpi.org/software/ompi/v4.1/) for parallel processing.

## Debugging

In VSCode, you can run the specific algorithm and environment configuration through the Command Palette (Ctrl+Shift+P) by filtering on Debug: Select and Start Debugging or typing 'debug ' and selecting the desired configuration. An existing template has been provided.

## Files

- `/algo`: contains the PPO implementations and neural network architectures
- `/demo`: contains demonstrations.
- `/gym_rad_search`: contains the radiation source search OpenAI gym environment
- `/unit_testing`: contains unit tests. This functionality is under development

## Documentation

Documentation generated with [Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html).

Generate documentation with `sphinx-build -b html sourcedir builddir` from root directory
