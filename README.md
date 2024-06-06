# Deep Learning for Physics

A tutorial to introduce Physicists to the idea of using Neural Networks for parameterizing wave functions. In our scenario, we combine variational monte carlo approach with a neural quantum state to search for the ground state of a 2D lattice of Rydberg atoms

The following resources were consulted for this tutorial

- [Sprague and Czischek, 2024](https://www.nature.com/articles/s42005-024-01584-y)
- [Zhang and Ventra, 2023](https://physics.paperswithcode.com/paper/transformer-quantum-state-a-multi-purpose)
- [Czischek et. al., 2022](https://arxiv.org/pdf/2203.04988)
- [Hibat-Allah et. al., 2020](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.023358)
- [Deep Learning Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html)
- [QuCumber](https://github.com/PIQuIL/QuCumber)

You can consult the links for further knowledge.


## Physics of the Problem

Let us consider the physics of the problem.

- We are looking at a 2D lattice of Rydberg atoms
- We will be using the Rydberg Hamiltonian
- We are assuming all-to-all interaction between all lattice sites

## Models

- Recurrent Neural Network (Gated Recurrent Unit)
- Transformer (with Multihead Self-Attention)

## Language/Framework

- [Python3](https://www.python.org)
- [Jax](https://jax.readthedocs.io)
- [Flax](https://flax.readthedocs.io)
- [Torch](https://pytorch.org)

## Requirements

You should have both of the following installed on your local machine one way or the other

- [Python3](https://www.python.org)
- [Git](https://git-scm.com/) (You may need to download the folders as a compressed folder if you do not have git installed)


## How to Use

### Unix/Mac

1. Clone the repository:

    ```bash
    git clone https://github.com/lere01/tutorial_nqs.git
    ```

2. Change directory:

    ```bash
    cd tutorial_nqs
    ```

3. Run the setup script:

    ```bash
    bash run.sh
    ```

Remember to run `chmod +x run.sh` to make the script executable before running. Some unix based systems allow you to simply double-click on the file.

### Windows

1. Clone the repository:

    ```bash
    git clone https://github.com/lere01/tutorial_nqs.git
    ```

2. Change directory:

    ```bash
    cd tutorial_nqs
    ```

3. Run the setup script:

    ```bat
    run.bat
    ```

Note that opening the root directory in Windows Explorer and double clicking `run.bat` can also achieve the same thing.

### Advanced Users (Makefile)

The advantage of the Make commands is the fine grained control you get over running/stopping the app and cleaning your environment. So if you already have `make` setup on your PC/Mac, then using the following commands would serve better.

1. Run the setup and start the application:

    ```bash
    make run
    ```

2. To stop the application:

    ```bash
    make stop
    ```

3. To clean up the environment:

    ```bash
    make clean
    ```

These steps will ensure that the virtual environment is created, dependencies are installed, and the application is run, all with a single command, making it easier for you to get started.
