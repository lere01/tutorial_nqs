import streamlit as st
from models import get_model

st.markdown('<link href="../static/css/styles.css" rel="stylesheet">', unsafe_allow_html=True)
st.title("Neural Quantum State")

st.write(
    """
        Now that you have set your parameters, let's confirm your choices. 
        Click the button below to confirm that you are getting the expected model.
    """
)  

# Confirm Model
if st.button("Confirm Model"):
    try:
        model_config = st.session_state.model_config
        model_type = st.session_state.model_type
        vmc_config = st.session_state.vmc_config
    except AttributeError:
        st.write("Something is wrong. Please, close the app and restart it.")

    config_not_set = model_config is None or model_type is None or vmc_config is None
    
    if config_not_set:
        st.write("Oops! No available configuration. Did you forget to save your configuration on the previous page?")
    else:
        model = get_model(model_type)(*model_config)
        st.session_state.model = model
        
        st.markdown(f"""
            Welldone! You are ready to use {model_type}. Remember that you can 
            always change your configuration by going back to the previous page. Remember to save your configuration.
        """)


        st.markdown("""
            Now, let us look at the architecture of our approach. All the following have been 
            implemented except for the `local_energy`.

            - :blue[Wave Function] - This comes from our model
            - :blue[Sampling] - Monte Carlo sampling
            - :blue[Logpsi] - This is the logarithm of the wavefunction
            - :red[Local Energy] - Part of the Hamiltonian has been implemented; you will complete it
            - :blue[Interactions] - We are using all-to-all interactions
            - :blue[Loss Function]: - This is the expectation value of the local energy
        """)







# Footer Navigation
col1, _, _, _, _, _, _, col2 = st.columns(8)

with col1:
    st.page_link("pages/configuration.py", label="Back", icon=":material/arrow_back:")


if st.session_state.get("model") is not None:
    with col2:
        st.page_link("pages/energy_function.py", label="Next", icon=":material/arrow_forward:")