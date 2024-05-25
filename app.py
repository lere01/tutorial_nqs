import streamlit as st


def main():
    st.markdown('<link href="static/css/styles.css" rel="stylesheet">', unsafe_allow_html=True)
    st.title("Neural Networks for Wave Functions Parameterization")
    
    # Sidebar
    st.sidebar.title(
        """
        Play Around with Parameters
        """
    )

    # Model selection
    model_type = st.sidebar.selectbox("Choose Model", ["RNN", "Transformer"])
    model_params = None
    vmc_params = None

    # Body Section
    st.write(
        """
        ## Introduction
        This app allows you to explore the parameterization of wave functions using neural networks. Here in the sidebar, you 
        will be able to select different models and play around with hyperparameters. You should see the controls on your left.
        """
    )


if __name__ == "__main__":
    main()
