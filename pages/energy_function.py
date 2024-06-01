import io
import sys
import os
import json
import streamlit as st
import textwrap
from code_editor import code_editor
from streamlit_extras.let_it_rain import rain
from streamlit_extras.switch_page_button import switch_page

from src.helpers import ModelType, run_tf_model
from src.rnn_model.vmc import VMC
from jax import lax, random
import jax.numpy as jnp
from typing import List
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Energy Function - NQS Tutorial",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "model" not in st.session_state:
    switch_page("app")


cwd = os.getcwd()
button_file_path = os.path.join(cwd, "static", "buttons.json")
with open(button_file_path, "r") as file:
    buttons = json.load(file)

st.markdown(
    '<link href="../static/css/styles.css" rel="stylesheet">', unsafe_allow_html=True
)
st.title("Energy Function")



# Provide a code editor for the user to edit the code
st.markdown(
    """
    ### Compute the Local Energy
    Are you ready for some tasks? Replace all the `CHANGE_ME` in the code below with the appropriate code to compute the local energy. When you 
    are done, click the `Run Code` button or press `Ctrl + Enter` (`command + return` for Mac) to run the code.

    - :red[**WARNING**]: Only change the places where you find `CHANGE_ME`. Changing any other part of the code may result in an error.
    - :blue[**HINT**]: Remember that `delta` and `Omega` are the detuning and Rabi frequency respectively. They play crucial roles in the computation of the local energy.
    """
)

boilerplate_code = textwrap.dedent(
    """
    # Import Libraries
    import numpy as np

    # Define the Local Energy
    def local_energy():
        '''Compute the local energy of the wave function.'''
        print('Hello World!')

    local_energy()

    print('Completed!')
    """
)

runner_code_file = os.path.join(cwd, "src", "rnn_model", "runner.py")
with open(runner_code_file, "r") as file:
    runner_code = file.read()
run_model_code = textwrap.dedent(runner_code)
run_model_globals = {
    "vmc_config": st.session_state.vmc_config,
    "model": st.session_state.model,
    "VMC": VMC,
    "jnp": jnp,
    "lax": lax,
    "List": List,
    "rng_key": random.PRNGKey(1234),
    "st": st,
    "io": io,
    "sys": sys,
    "delta": 1.0,
    "Omega": 1.0,
    "state": st.session_state,
}

# Display the code editor
completed_code = code_editor(
    run_model_code,
    lang="python",
    theme="contrast",
    height=300,
    buttons=buttons,
    key="run_model_code",
)

def energy_plot(densities):
    fig, ax = plt.subplots()

    ax.plot(densities, label='Energy Density', color='g')
    ax.set_ylabel('Energy Density')
    ax.set_xlabel('Training Epoch')
    ax.set_title('Energy Density Over Training Epoch')
    ax.hlines(-0.45776822, 0, 1000, colors='r', linestyles='dashed', label='Target Energy Density')
    ax.legend()

    st.pyplot(fig, use_container_width=False)


if completed_code["type"] == "submit":
    with st.spinner("Checking your solution..."):
        # initialize task statuses
        todo_1_correct = False
        todo_2_correct = False
        todo_3_correct = False

        # extract the code from the code editor
        code = completed_code["text"]
        code_lines = code.strip().split('\n')

        # Check if the user has completed the first todo
        todo_1 = code_lines[10].strip().split()

        if todo_1[3] != 'delta':
            st.write("You might want to check on ToDo_1")
        else:
            todo_1_correct = True
            st.write("Well done! You have successfully completed the todo 1.")

        # Check if the user has completed the second todo
        todo_2 = code_lines[25].strip().split()

        if todo_2[5] != 'Omega':
            st.write("You might want to check on ToDo_2")
        else:
            todo_2_correct = True
            st.write("Well done! You have successfully completed the todo 2.")

        # Check if the user has completed the third todo
        todo_3 = code_lines[39].strip().split()
        
        todo_terms = set(todo_3)
        # condition one: each of ['interaction_term', 'transverse_field', 'chemical_potential', 'loc_e'] must be present in the code
        condition_one = all(term in todo_terms for term in ['interaction_term', 'transverse_field', 'chemical_potential', 'loc_e'])

        # condition two: '=' must be the second item in todo_3, and '+' must be the fourth and sixth items
        condition_two = todo_3[1] == '=' and all(todo_3[i] == '+' for i in (3, 5))

        if not condition_one or not condition_two:
            st.write("You might want to check on ToDo_3")
        else:
            todo_3_correct = True
            st.write("Well done! You have successfully completed the todo 3.")

        # Display checkboxes
        st.subheader("Checklist")
        st.checkbox("TODO_1: step_fn_chemical correctly implemented", value=todo_1_correct, disabled=True)
        st.checkbox("TODO_2: step_fn_transverse correctly implemented", value=todo_2_correct, disabled=True)
        st.checkbox("TODO_3: Total energy computation correctly implemented", value=todo_3_correct, disabled=True)


    # Execute the code
    if all([todo_1_correct, todo_2_correct, todo_3_correct]):
        rain(
            emoji="🎈",
            font_size=54,
            falling_speed=5,
            animation_length="5s",
        )
        if st.session_state.model_type.name == ModelType.RNN.name:
            try:
                with st.spinner("Your RNN model is training..."):
                    exec(completed_code["text"], run_model_globals)

                energy_plot(st.session_state.densities)
            except Exception as e:
                st.error(f"Error executing code: {e}")
        else:
            try:
                with st.spinner("Your Transformer model is training..."):
                    run_tf_model(st.session_state.model, st.session_state.full_opt, st.session_state.opt_dict)
            except Exception as e:
                st.error(f"Error executing code: {e}")


# Footer Navigation
col1, _, _, _, _, _, _, col2 = st.columns(8)

with col1:
    st.page_link(
        "pages/model_confirmation.py", label="Back", icon=":material/arrow_back:"
    )


with col2:
    st.page_link("app.py", label="Restart", icon=":material/refresh:")
