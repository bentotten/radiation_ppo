# Radiation Source Search Environment

This contains the multi-agent radiation localization architecture created by Ben Totten, loosely based on 
- The multi-agent architecture created by Alagha et al. and their published [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167739X22002266)
- The single-agent architecture created by Liu et al. and their published [paper](https://www.mdpi.com/1424-8220/19/4/960)
- The aingle-agent radiation source search environment created by Proctor et al. and their published [paper](https://www.mdpi.com/2673-4362/2/4/29).

This also contains an adapted form of the single-agent radiation source search architecture (RAD-A2C) created by Proctor et al. that can be run with multiple independent learners.

# Algorithms

TBD

## Files

TBD

## Documentation

Documentation generated with [Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html).

Generate documentation with `sphinx-build -b html docs doc_build` from root directory

## Quick-Start Installation

It is recommended to use the Anaconda package manager. The author did all development with [Micromamba](https://mamba.readthedocs.io/en/latest/installation.html), a fast and light-weight implementation of Anaconda. To use Micromamba, simply replace `conda` commands with `micromamba`. To do a command-line install, simple add `-c conda-forge` or `-c pytorch` or another channel to specify the correct channel.

1. Clone repository.

2. Create a virtual environment with the required packages `conda env create -f <PATH-TO-ALGORTIHM>/environment.yml`.

3. Activate this environment with `conda activate <ENV_NAME>`.

4. \*\* Note: The RAD-A2C implementation requires [OpenMPI](https://www.open-mpi.org/software/ompi/v4.1/) for parallel processing.

## Debugging

In VSCode, you can run the specific algorithm and environment configuration through the Command Palette (Ctrl+Shift+P) by filtering on Debug: Select and Start Debugging or typing 'debug ' and selecting the desired configuration. An existing template has been provided.

## Distributed Evaluation Mode

The evaluation portion of this codebase has been set up to work with [Ray Clusters](https://docs.ray.io/en/latest/cluster/getting-started.html). Each episode runs as it's own [Actor](https://docs.ray.io/en/latest/ray-core/actors.html)

