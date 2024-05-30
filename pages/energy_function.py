import os
import json
import tempfile
import subprocess
import streamlit as st
import textwrap
from code_editor import code_editor

cwd = os.getcwd()
button_file_path = os.path.join(cwd, "static", "buttons.json")
with open(button_file_path, "r") as file:
    buttons = json.load(file)

st.markdown('<link href="../static/css/styles.css" rel="stylesheet">', unsafe_allow_html=True)
st.title("Energy Function")



def run_user_code(user_code):
    """Execute the user code in a temporary file and return the output or errors."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
        tmp_file.write(user_code)
        tmp_file.flush()

        try:
            result = subprocess.run(['python', tmp_file.name], capture_output=True, text=True, check=True)
            return result.stdout, None
        except subprocess.CalledProcessError as e:
            return e.stdout, e.stderr
        finally:
            tmp_file.close()

# Read boilerplate code from a file
boilerplate_code_path = './boilerplate.py'
try:
    with open(boilerplate_code_path, 'r') as file:
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

# Display the code editor
completed_code = code_editor(
    boilerplate_code,
    lang="python",
    theme="contrast",
    height=300,
    buttons=buttons,
)

# print("Completed Code: \n", type(completed_code), "\n\n", completed_code.get("text"))

if completed_code['type'] == 'submit':
    st.write("Response type: ", completed_code['type'])
    st.code(completed_code['text'], language=completed_code['lang'])
    
    exec_globals = {}
    exec(completed_code['text'], exec_globals)

if st.button("Run Code"):
        print("Running code...", completed_code['text'])
        exec_globals = {}
        try:
            st.code(completed_code, language="python")
            exec(completed_code['text'], exec_globals)
        except Exception as e:
            st.error(f"Error executing code: {e}")

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
    st.page_link("pages/model_confirmation.py", label="Back", icon=":material/arrow_back:")