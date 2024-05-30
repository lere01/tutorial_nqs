import io
import sys
import os
import json
import tempfile
import subprocess
import streamlit as st
import textwrap
from code_editor import code_editor

from vmc import VMC
from jax import lax, random
import jax.numpy as jnp
from typing import List

cwd = os.getcwd()
button_file_path = os.path.join(cwd, "static", "buttons.json")
with open(button_file_path, "r") as file:
    buttons = json.load(file)

st.markdown(
    '<link href="../static/css/styles.css" rel="stylesheet">', unsafe_allow_html=True
)
st.title("Energy Function")


def run_user_code(user_code):
    """Execute the user code in a temporary file and return the output or errors."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(user_code)
        tmp_file.flush()

        try:
            result = subprocess.run(
                ["python", tmp_file.name], capture_output=True, text=True, check=True
            )
            return result.stdout, None
        except subprocess.CalledProcessError as e:
            return e.stdout, e.stderr
        finally:
            tmp_file.close()


# Read boilerplate code from a file
boilerplate_code_path = "./boilerplate.py"
try:
    with open(boilerplate_code_path, "r") as file:
        boilerplate_code = file.read()
except FileNotFoundError:
    boilerplate_code = "# Boilerplate code file not found."


# Provide a code editor for the user to edit the code
st.write(
    """
    ### Compute the Local Energy
    Replace all the `ToDo` in the code below with the appropriate code to compute the local energy.
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

runner_code_file = os.path.join(cwd, "src", "runner.py")
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
}

# Display the code editor
completed_code = code_editor(
    run_model_code,
    lang="python",
    theme="contrast",
    height=500,
    buttons=buttons,
    key="run_model_code",
)

# print("Completed Code: \n", type(completed_code), "\n\n", completed_code.get("text"))

if completed_code["type"] == "submit":
    # Redirect stdout to a string buffer
    # old_stdout = sys.stdout
    # new_stdout = io.StringIO()
    # sys.stdout = new_stdout

    # try:
    #     exec(completed_code["text"], run_model_globals)
    # except Exception as e:
    #     st.error(f"Error executing code: {e}")
    # finally:
    #     sys.stdout = old_stdout

    # Placeholder for updating loss in real-time
    # output = new_stdout.getvalue()
    # st.write("Output: ")
    # st.text(output)
    # output_placeholder.text(output)
    

    # Reset stdout
    # sys.stdout = old_stdout
    try:
        exec(completed_code["text"], run_model_globals)
    except Exception as e:
        st.error(f"Error executing code: {e}")

# if st.button("Run Code"):
#         print("Running code...", completed_code['text'])
#         exec_globals = {}
#         try:
#             st.code(completed_code, language="python")
#             exec(completed_code['text'], exec_globals)
#         except Exception as e:
#             st.error(f"Error executing code: {e}")

# if st.button("Run Code"):
#     if completed_code:
#         with st.spinner("Running your code..."):
#             output, error = run_user_code(completed_code)

#         if error:
#             st.error(f"Error:\n{error}")
#         else:
#             st.success("Code ran successfully!")
#             st.text(f"Output:\n{output}")
#     else:
#         st.warning("Please enter some code before running.")


# Footer Navigation
col1, _, _, _, _, _, _, col2 = st.columns(8)

with col1:
    st.page_link(
        "pages/model_confirmation.py", label="Back", icon=":material/arrow_back:"
    )
