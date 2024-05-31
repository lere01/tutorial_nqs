# Tutorial at Quantum Algorithm Summer School

A tutorial to introduce Physicists to the idea of using Neural Networks for parameterizing wave functions. In our scenario, we combine variational monte carlo approach with a neural quantum state to search for the ground state of a 2D lattice of Rydberg atoms

Inspiration (and some code) for this tutorial was drawn from 

- [Sprague and Czischek, 2024](https://www.nature.com/articles/s42005-024-01584-y)
- [Zhang and Ventra, 2023](https://physics.paperswithcode.com/paper/transformer-quantum-state-a-multi-purpose)
- [Czischek et. al., 2022](https://arxiv.org/pdf/2203.04988)
- [Hibat-Allah et. al., 2020](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.023358)
- [Deep Learning Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html)

You can consult the links for further knowledge.


## Physics of the Problem

Let us consider the physics of the problem.

- We are looking at a 2D lattice of Rydberg atoms
- We will be using the Ising Model
- We are assuming all-to-all interaction between all lattice sites
- The Hamiltonian is as follows

$$
\begin{equation}
\tilde{H} = - \frac{\Omega}{2} \sum_{i = 1}^N \left( \hat{\sigma}_i^x \right) - \delta \sum_{i = 1}^N \left ( \hat{n}_i \right ) + \sum_{i,j} \left ( V_{ij} \hat{n}_i \hat{n}_j \right )
\end{equation}
$$

where $V_{ij} = \frac{7}{| \textbf{r}_i - \textbf{r}_j |^6}$.

## Models

- Recurrent Neural Network (Gated Recurrent Unit)
- Transfomer (with Multihead Self-Attention)

## Language/Framework

- Python  remember to add links
- Jax
- Flax

## Requirements

You should have both of the following installed on your local machine one way or the other

- Python
- Git (You may need to download the folders as a compressed folder if you do not have git installed)


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
    bash setup.sh
    ```

Remember to run `chmod +x setup.sh` to make the script executable.

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

If you are comfortable with using `makefile` then you can use the following commands

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


These steps will ensure that the virtual environment is created, dependencies are installed, and the application is run, all with a single command, making it easier for students to get started.

### Web Access

If you do not want to go through the hastle of downloading/cloning the repo, you can access the web app [here](https://streamlit.io). But note that it may be slow to respond depending on whether the app has hibernated or not. Also, it may no longer be availabe for access after some time.