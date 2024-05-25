# Tutorial at Quantum Algorithm Summer School

A tutorial to introduce Physicists to the idea of using Neural Networks for parameterizing wave functions. In our scenario, we combine variational monte carlo approach with a neural quantum state to search for the ground state of a 2D lattice of Rydberg atoms

## Inspiration and Materials for the Tutorial

## Physics of the Problem


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