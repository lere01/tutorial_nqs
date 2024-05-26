import tempfile
import subprocess
import streamlit as st

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

# Provide a text area for code entry
user_code = st.text_area("Complete the task by replacing all the ToDos:", value=boilerplate_code, height=300)

if st.button("Run Code"):
    if user_code:
        with st.spinner("Running your code..."):
            output, error = run_user_code(user_code)

        if error:
            st.error(f"Error:\n{error}")
        else:
            st.success("Code ran successfully!")
            st.text(f"Output:\n{output}")
    else:
        st.warning("Please enter some code before running.")


# Footer Navigation
col1, _, _, _, _, _, _, col2 = st.columns(8)

with col1:
    st.page_link("pages/model_confirmation.py", label="Back", icon=":material/arrow_back:")