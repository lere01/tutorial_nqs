import os
import streamlit as st
# from startup import prepare_file_system
# from rnn_model.definitions.enums import ModelType
# from src.rnn_model.definitions.configs import VMCConfig, TransformerConfig, VMCModel
# from typing import NamedTuple, get_type_hints, List, Dict
from streamlit_extras.add_vertical_space import add_vertical_space
from src.helpers import TransformerConfig, RNNConfig, VMCConfig
from src.helpers import get_widget_group, ModelType, RydbergConfig, TrainConfig
from src.helpers import save_rnn, save_ptf, save_lptf
from src.helpers import TransformerConfigDescription
from src.helpers import RydbergConfigDescription, TrainConfigDescription, VMCConfigDescription



def main():
    # Set the page configuration
    st.set_page_config(
    page_title="Configuration - NQS Tutorial",
    page_icon="",
    layout="wide",
    )

    # Initialize Session State
    st.session_state.configuration_saved = False

    # Page Styling
    st.markdown('<link href="../static/css/styles.css" rel="stylesheet">', unsafe_allow_html=True)
    st.title("Configuration Parameters")

    # Sidebar
    st.sidebar.title(
        """
        Set the Parameters
        """
    )


    # Body Section
    st.markdown(
        """
            In this tutorial, we will be trying three model architectures - Recurrent Neural Network (see this [paper](https://arxiv.org/pdf/2203.04988)) and, 
            Patched Transformer and Large Patched Transformer (see this [paper](https://www.nature.com/articles/s42005-024-01584-y)). For those who might be interested in the detail, we will be using the Gated Recurrent Unit (GRU) flavour of the RNN. 
            On this page, you will be able to do three things:
            
            - Select which of the two models you would like to use
            - Configure the hyperparameters of the model
            - Configure the hyperparameters of the Variational Monte Carlo (VMC) approach


            You should see the controls on in the sidebar; your left. If not click the `>` button. In the sidebar, you will be able to select one of two models and see 
            the parameters on the main page. You can also configure the hyperparameters of the model and the VMC approach. Remember to click the `Save Configuration` button 
            to save your configuration. Once saved, you can proceed to the next step.
            
            
            **Note**: Disabled inputs are not configurable as they are set to default values.
        """
    )

    add_vertical_space(3)
  
    # Model selection
    options=[(model.name) for model in ModelType]
    model_type = st.sidebar.selectbox("Choose Model", options)

    cwd = os.getcwd()
    img_dir = os.path.join(cwd, "pages", "images")

    def get_image_path(model_name):
        img_dict = {
            "RNN": "rnn.png",
            "Transformer": "tf.png",
        }
        return os.path.join(img_dir, f"{img_dict[model_name]}")

    if model_type == ModelType.RNN.name:
        st.write(""" ### RNN Configuration """)
    elif model_type == ModelType.Transformer.name:
        st.write(""" ### Transformer Configuration """)

    left_col, right_col = st.columns([2, 5])
    with left_col:
        st.image(get_image_path(model_type), caption=f"{model_type} Architecture", use_column_width=True)
    with right_col:
        if model_type == ModelType.RNN.name:
            tab1, tab2 = st.tabs(["Model Configuration", "VMC Configuration"])
            with tab1:
                output_dim = st.number_input("Output Dimension (Only 2 is supported)", min_value=2, max_value=2, value=2, disabled=True)
                num_hidden_units = st.number_input("Number of Hidden Units", min_value=4, max_value=64, value=64, step=4)
                model_config = RNNConfig(output_dim, num_hidden_units)

            with tab2:
                exclude_list = ["nx", "output_dim", "sequence_length"]
                vmc_config = get_widget_group(VMCConfig, VMCConfigDescription, exclude_list)
        
        else:
            tab1, tab2, tab3 = st.tabs(["Model Configuration", "Training Configuration", "Rydberg Configuration"])
            with tab1:
                ptf_config = get_widget_group(
                    TransformerConfig,
                    TransformerConfigDescription, 
                    []
                )
                model_config = TransformerConfig(**ptf_config)
            with tab2:
                trainconfig = get_widget_group(
                    TrainConfig,
                    TrainConfigDescription, 
                    []
                )
                train_config = TrainConfig(**trainconfig)
            with tab3:
                rydbergconfig = get_widget_group(
                    RydbergConfig,
                    RydbergConfigDescription, 
                    []
                )
                rydberg_config = RydbergConfig(**rydbergconfig)


            
    add_vertical_space(3)
    if st.button("Save Configuration"):
        st.session_state.model_type = ModelType[model_type]
        st.session_state.model_config = model_config

        if model_type == ModelType.RNN.name:
            vmc_config['num_hidden_units'] = model_config.num_hidden_units
        
            st.session_state.vmc_config = VMCConfig(
                **vmc_config
            )
            save_rnn(model_config, st.session_state.vmc_config)

        else:
            st.session_state.train_config = train_config
            st.session_state.rydberg_config = rydberg_config
            save_ptf(model_config, train_config, rydberg_config)
        
            

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
