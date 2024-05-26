import streamlit as st
from startup import prepare_file_system
from definitions.enums import ModelType
from definitions.configs import RNNConfig, VMCConfig, TransformerConfig
from typing import NamedTuple, get_type_hints, List, Dict

def get_widget(description, field_type, default_value, disabled=False):
        if field_type == int:
            return st.number_input(description, min_value=0, value=default_value, step=1, disabled=disabled)
        elif field_type == float:
            return st.number_input(description, min_value=0.00005, value=default_value, step=0.001, disabled=disabled)

def get_sidebar_widget(description, field_type, default_value, disabled=False):
        if field_type == int:
            return st.sidebar.number_input(description, min_value=0, value=default_value, step=1, disabled=disabled)
        elif field_type == float:
            return st.sidebar.number_input(description, min_value=0.00005, value=default_value, step=0.001, disabled=disabled)
    

def get_widget_group(config: NamedTuple, exclude_list: List[str], sidebar=False) -> Dict:  
    widget_group = {}
    field_defaults = config._field_defaults
    
    if sidebar:
        for field_name, field_type in get_type_hints(config).items():
            default_value = field_defaults.get(field_name, None)
            description = field_name.replace("_", " ").title()
            
            widget_group[field_name] = get_sidebar_widget(description, field_type, default_value) if field_name not in exclude_list else get_sidebar_widget(description, field_type, default_value, True)
    else:
        for field_name, field_type in get_type_hints(config).items():
            default_value = field_defaults.get(field_name, None)
            description = field_name.replace("_", " ").title()
            
            widget_group[field_name] = get_widget(description, field_type, default_value) if field_name not in exclude_list else get_widget(description, field_type, default_value, True)
    
    return widget_group

def main():
    st.markdown('<link href="../static/css/styles.css" rel="stylesheet">', unsafe_allow_html=True)
    st.title("Configuration Parameters")

    # Sidebar
    st.sidebar.title(
        """
        Set the Parameters
        """
    )

    # Body Section
    st.write(
        """
            In the sidebar, you will be able to select different models and play around with hyperparameters. 
            You should see the controls on your left. If not click the `>` button. 
            
            **Note**: Disabled inputs are not configurable as they are set to default values.
        """
    )

    # Model selection
    options=[(model.name) for model in ModelType]
    model_type = st.sidebar.selectbox("Choose Model", options)
    

    # VMC Configuration
    exclude_list = ["nx", "output_dim", "sequence_length"]
    vmc_config = get_widget_group(VMCConfig, exclude_list, sidebar=True)


    
    if model_type == ModelType.RNN.name:
        st.write(""" ### RNN Configuration """)
        output_dim = st.number_input("Output Dimension (Only 2 is supported)", min_value=2, max_value=2, value=2, disabled=True)
        num_hidden_units = st.number_input("Output Dimension", min_value=4, max_value=64, value=16, step=4)
            
        model_config = RNNConfig(output_dim, num_hidden_units)
        

    elif model_type == ModelType.TRANSFORMER.name:
        st.write(""" ### Transformer Configuration """)
        transformer_config = get_widget_group(TransformerConfig, [])
        model_config = TransformerConfig(**transformer_config)
        

    if st.button("Save Configuration"):
        st.session_state.model_type = ModelType[model_type]
        st.session_state.vmc_config = VMCConfig(**vmc_config)
        st.session_state.model_config = model_config

        st.write("Configuration Saved Successfully! You can now proceed to the next step.")
        st.session_state.configuration_saved = True

    # Footer Navigation
    col1, _, _, _, _, _, _, col2 = st.columns(8)

    with col1:
        st.page_link("app.py", label="Back", icon=":material/arrow_back:")

    
    if st.session_state.get("configuration_saved", False) == True:
        with col2:
            st.page_link("pages/model_confirmation.py", label="Next", icon=":material/arrow_forward:")



if __name__ == "__main__":
    main()
