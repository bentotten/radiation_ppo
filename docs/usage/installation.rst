
Installation
------------

Anaconda or other package managers recommended. The author did all development with `Micromamba <https://mamba.readthedocs.io/en/latest/installation.html>`_, 
a fast and light-weight implementation of Anaconda. To use Micromamba, simply replace `conda` commands with `micromamba`. To do a command-line install, simply
add `-c conda-forge` or `-c pytorch` or another channel to specify the correct channel.

.. contents::
   :depth:  4

Requirements
############

See algorithm environment.yml.


Installing
##########


1. Clone repository.

2. Create a virtual environment with the required packages

.. code-block:: bash

   $ conda env create -f <PATH-TO-ALGORTIHM>/environment.yml

3. Activate this environment with `conda activate <ENV_NAME>`.

4. \*\* Note: The RAD-A2C implementation requires [OpenMPI](https://www.open-mpi.org/software/ompi/v4.1/) for parallel processing.


Run it
######

During development, the easiest path is to run your application as
follow:

.. code-block:: bash

   $ python3 algos/multiagent/main.py

Replace `multiagent` with any alternative desired algorithms located in the algo file.
