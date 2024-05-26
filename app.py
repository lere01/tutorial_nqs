import startup ## this has to be the first import
import streamlit as st


st.markdown('<link href="static/css/styles.css" rel="stylesheet">', unsafe_allow_html=True)
st.title("Neural Networks for Wave Functions Parameterization")

# Body Section
st.write(
    """
        ## Welcome
        This app allows you to explore the parameterization of wave functions using neural networks. A tutorial to introduce 
        Physicists to the idea of using Neural Networks for parameterizing wave functions. In our scenario, we combine variational 
        monte carlo approach with a neural quantum state to search for the ground state of a 2D lattice of Rydberg atoms.
    """
)

# Next Page
st.write(
    """
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
