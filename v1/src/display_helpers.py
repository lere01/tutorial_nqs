# src/display_helpers.py


import ipywidgets as widgets
from IPython.display import display, HTML
from definitions.enums import ModelType
from definitions.configs import RNNConfig, TransformerConfig, VMCConfig
from typing import NamedTuple, get_type_hints

def create_widgets_from_config(config_class: NamedTuple):
    widget_mapping = {
        int: widgets.IntText,
        float: widgets.FloatText,
        str: widgets.Text,
    }

    widgets_dict = {}
    field_defaults = config_class._field_defaults

    for field_name, field_type in get_type_hints(config_class).items():
        widget_class = widget_mapping.get(field_type, widgets.Text)
        default_value = field_defaults.get(field_name, None)
        widgets_dict[field_name] = widget_class(description=f'{field_name.replace("_", " ").title()}:', value=default_value, style={'description_width': 'initial'})
    
    return widgets_dict

def create_dropdown_widget():
    model_type_dropdown = widgets.Dropdown(
        options=[(model.name, model.value) for model in ModelType],
        value=ModelType.RNN.value,
        description='Model Type:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    return model_type_dropdown

def update_display(change, rnn_box, transformer_box):
    if change['type'] == 'change' and change['name'] == 'value':
        if change['new'] == ModelType.RNN.value:
            rnn_box.layout.display = 'block'
            transformer_box.layout.display = 'none'
        elif change['new'] == ModelType.TRANSFORMER.value:
            rnn_box.layout.display = 'none'
            transformer_box.layout.display = 'block'

def display_configuration_ui():
    model_type_dropdown = create_dropdown_widget()
    rnn_widgets = create_widgets_from_config(RNNConfig)
    transformer_widgets = create_widgets_from_config(TransformerConfig)

    rnn_box = widgets.VBox(list(rnn_widgets.values()), layout=widgets.Layout(display='block'))
    transformer_box = widgets.VBox(list(transformer_widgets.values()), layout=widgets.Layout(display='none'))

    model_type_dropdown.observe(lambda change: update_display(change, rnn_box, transformer_box))

    update_display({'type': 'change', 'name': 'value', 'new': model_type_dropdown.value}, rnn_box, transformer_box)

    display(HTML("<h2>Select Model Type and Configuration</h2>"))
    display(model_type_dropdown)
    display(rnn_box)
    display(transformer_box)

    return model_type_dropdown, rnn_widgets, transformer_widgets

def display_vmc_configuration_ui():
    vmc_widgets = create_widgets_from_config(VMCConfig)
    vmc_box = widgets.VBox(list(vmc_widgets.values()))

    display(HTML("<h2>VMC Configuration</h2>"))
    display(vmc_box)

    return vmc_widgets




