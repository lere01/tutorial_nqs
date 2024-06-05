import streamlit as st
from src.rnn_model.models import get_model
from streamlit_extras.switch_page_button import switch_page
from src.helpers import ModelType, RydbergConfig, TrainConfig, load_config, extract_args, get_tf_model
# from src.tf_models.model_builder import *
from src.run import run


# Set Page Configuration
st.set_page_config(
    page_title="Model Confirmation - NQS Tutorial",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "model_config" not in st.session_state or "model_type" not in st.session_state or "vmc_config" not in st.session_state:
    switch_page("app")
st.session_state.model_confirmed = False

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
    config_set = st.session_state.get("configuration_saved", False)
    if config_set is False:
        st.write("Oops! No available configuration. Did you forget to save your configuration on the previous page?")
    else:
        try:
            if st.session_state.model_type.name == ModelType.RNN.name:
                model = get_model(st.session_state.model_type)(*st.session_state.model_config)
                st.session_state.model = model
            else:
                cnfgs = extract_args(st.session_state.model_type)
                model, full_opt, opt_dict = run(cnfgs)
                st.session_state.model = model
                st.session_state.full_opt = full_opt
                st.session_state.opt_dict = opt_dict

            # Let session state know that the model has been confirmed
            st.session_state.model_confirmed = True
            
        
            st.markdown(f"""
                Welldone! You are ready to use {st.session_state.model_type}. Remember that you can 
                always change your configuration by going back to the previous page. Remember to save your configuration.
            """)

            st.write(f"{st.session_state.model_type}: Click to expand the brackets and see the configuration you saved.")
            record = load_config(st.session_state.model_type)
            st.json(record, expanded=False)


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
        except Exception as e:
            st.text({e})
            st.write(f"Something went wrong. {e}\n\n")
            st.write("Please, close the app and restart it. Or call the attention of the developer.")







# Footer Navigation
col1, _, _, _, _, _, _, col2 = st.columns(8)

with col1:
    st.page_link("pages/configuration.py", label="Back", icon=":material/arrow_back:")


if st.session_state.get("model_confirmed", False) is True:
    with col2:
        st.page_link("pages/energy_function.py", label="Next", icon=":material/arrow_forward:")