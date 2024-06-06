import startup ## this has to be the first import
import os
import streamlit as st


st.set_page_config(
    page_title="Welcome - NQS Tutorial",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown('<link href="static/css/styles.css" rel="stylesheet">', unsafe_allow_html=True)
st.title("Neural Networks for Wave Functions Parameterization")

cwd = os.getcwd()
image_path = os.path.join(cwd, "static", "nn_models.png")
# training_image = os.path.join(cwd, "static", "training.png")
st.image(image_path)

# Body Section
st.markdown(
    r"""
        ## Welcome
        This app allows you to explore the parameterization of wave functions using neural networks. This tutorial will introduce 
        you to the idea of using Neural Networks for parameterizing wave functions. In our scenario, we combine variational 
        monte carlo approach with a neural quantum state to search for the ground state of a 2D lattice of Rydberg atoms.

        ## Acknowledgements

        - The above image and other images in other pages were taken from [Sprague and Czischek](https://www.nature.com/articles/s42005-024-01584-y/figures/1). 
        - The following resources were consulted for this tutorial
            - [Sprague and Czischek, 2024](https://www.nature.com/articles/s42005-024-01584-y)
            - [Zhang and Ventra, 2023](https://physics.paperswithcode.com/paper/transformer-quantum-state-a-multi-purpose)
            - [Czischek et. al., 2022](https://arxiv.org/pdf/2203.04988)
            - [Hibat-Allah et. al., 2020](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.023358)
            - [Deep Learning Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html)

        - With permission, code in the following repository was used for patched transformer and large patched transformer models:
            - https://github.com/APRIQuOt/VMC_with_LPTF

        ### Physics of the Problem

        Let us consider the physics of the problem.

        - We are looking at a 2D lattice of Rydberg atoms
        - We are assuming all-to-all interaction among lattice sites
        - The Hamiltonian is as follows

        $$
        \begin{equation}
            \tilde{H} = - \frac{\Omega}{2} \sum_{i = 1}^N \left( \hat{\sigma}_i^x \right) - \delta \sum_{i = 1}^N \left ( \hat{n}_i \right ) + \sum_{i,j} \left ( V_{ij} \hat{n}_i \hat{n}_j \right )
        \end{equation} 
        $$

        where $V_{ij} = \frac{\Omega R_b^6}{| \textbf{r}_i - \textbf{r}_j |^6}$ and $R_b$ is the Rydberg blockade radius.

        - $\Omega$ is the Rabi frequency
        - $\delta$ is the detuning
        - $\hat{\sigma}_i^x$ is the Pauli-X matrix
        - $\hat{n}_i$ is the number operator
        - Atoms at positions $\textbf{r}_i$ and $\textbf{r}_j$ interact through the van der Waals potential, $V_{ij}$
        - $N$ is the number of lattice sites
        
        Note that we set $\Omega = \delta = 1$ and $R_b = 7^{\frac{1}{2}}$. This is to put the system in the vicinity of transition between the ordered 
        and striated phase.



        ### A Bird's Eyeview of the Approach
        - Step 1: Take some arbitrary parameterized wave function (neural network)
        - Step 2: Sample from it
        - Step 3: Compute the expectation value of the energy
        - Step 4: Vary your parameters using some optimization function
        - Repeat Steps 2-4 until you reach the ground state
        - Our training metric is the energy density of the system
    """
)

# Next Pages
# st.image(training_image)
st.markdown(
    r"""
        ---

        There are two parts of this exercise

        - Train a model to search for the ground state energy. Your result would look like the picture above.
        - Sample from a trained network and compute observables
        
        Let's Get Started. Click the button below to get started.
    """
)

# Initialize Session State
if "model_config" not in st.session_state:
    st.session_state.model_config = None
if "model_type" not in st.session_state:
    st.session_state.model_type = None
if "vmc_config" not in st.session_state:
    st.session_state.vmc_config = None

# Footer Navigation
# Add vertical space
for _ in range(5):
    st.write("")

_, _, _, _, _, _, _, col1 = st.columns(8)

# go to home page if clicked
with col1:
    st.page_link("pages/configuration.py", label="Get Started", icon=":material/arrow_forward:")


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p><a style='display: block; text-align: center;' href="#" target="_blank">Neural Network Parameterization of Wave Functions</a></p>
</div>
"""
# st.markdown(footer,unsafe_allow_html=True)



