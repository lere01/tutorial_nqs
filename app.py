import startup ## this has to be the first import
import streamlit as st


st.set_page_config(
    page_title="Welcome - NQS Tutorial",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown('<link href="static/css/styles.css" rel="stylesheet">', unsafe_allow_html=True)
st.title("Neural Networks for Wave Functions Parameterization")

# Body Section
st.markdown(
    r"""
        ## Welcome
        This app allows you to explore the parameterization of wave functions using neural networks. This tutorial will introduce 
        you to the idea of using Neural Networks for parameterizing wave functions. In our scenario, we combine variational 
        monte carlo approach with a neural quantum state to search for the ground state of a 2D lattice of Rydberg atoms.

        ### Physics of the Problem

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

        - $\Omega$ is the Rabi frequency
        - $\delta$ is the detuning
        - $\hat{\sigma}_i^x$ is the Pauli-X matrix
        - $\hat{n}_i$ is the number operator
        - $V_{ij}$ is the interaction potential
        - $N$ is the number of lattice sites
        
        Note that we set $\Omega$ = $\delta$ = 1. This is to put the system near the critical point.


        ### A Bird's Eyeview of the Approach
        - Step 1: Take some arbitrary parameterized wave function (neural network)
        - Step 2: Sample from it
        - Step 3: Compute the expectation value of the energy
        - Step 4: Vary your parameters using some optimization function
        - Repeat Steps 2-4 until you reach the ground state
    """
)

# Next Page
st.write(
    """
        ---
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
_, _, _, _, _, _, _, col1 = st.columns(8)

# go to home page if clicked
with col1:
    st.page_link("pages/configuration.py", label="Get Started", icon=":material/arrow_forward:")
