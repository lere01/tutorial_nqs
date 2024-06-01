import os
import json
import torch
from enum import Enum
import streamlit as st
from typing import NamedTuple, get_type_hints, List, Dict, Union
from src.tf_models.model_builder import *

class ModelType(Enum):
    RNN = 1
    PatchedTRANSFORMER = 2
    # LargePatchedTRANSFORMER = 3


class RNNConfig(NamedTuple):
    output_dim: int = 2
    num_hidden_units: int = 64

RNNConfigDescription = {
    "output_dim": "Output Dimension (Only 2 is supported)",
    "num_hidden_units": "Number of Hidden Units"
}

class VMCConfig(NamedTuple):
    n_samples: int = 1000
    nx: int = 4
    learning_rate: float = 0.01
    num_epochs: int = 1000
    output_dim: int = 2
    sequence_length: int = 16
    num_hidden_units: int = 64


VMCConfigDescription = {
    "n_samples": "Number of Samples",
    "nx": "Width/Height of the lattice",
    "learning_rate": "Learning Rate",
    "num_epochs": "Number of Epochs",
    "output_dim": "Output Dimension",
    "sequence_length": "Sequence Length",
    "num_hidden_units": "Number of Hidden Units"
}


"""
    PTF Optional arguments:
    
        L          (int)     -- The total number of atoms in your lattice.
    
        Nh         (int)     -- Transformer token size. Input patches are projected to match the token size.
    
        patch      (str)     -- Number of atoms input/predicted at once (patch size).
                                The Input sequence will have an effective length of L/prod(patch).
                                Example values: 2x2, 2x3, 2, 4
            
        dropout    (float)   -- The amount of dropout to use in the transformer layers.
        
        num_layers (int)     -- The number of transformer layers to use.
        
        nhead     (int)      -- The number of heads to use in Multi-headed Self-Attention. This should divide Nh.
    
        repeat_pre (bool)    -- Repeat the precondition (input) instead of projecting it out to match the token size.
""" 
class PatchedTransformerConfig(NamedTuple):
    L: int = 64
    # Nh: int = 128
    patch: str = "2x2"
    # dropout: float = 0.0
    # num_layers: int = 2
    nhead: int = 8

PatchedTransformerConfigDescription = {
    "L": "L - The total number of atoms in your lattice",
    "Nh": "Nh - Transformer token size. Input patches are projected to match the token size",
    "patch": "Patch - Number of atoms input/predicted at once (patch size)",
    "dropout": "Dropout - The amount of dropout to use in the transformer layers",
    "num_layers": "Num of Layers - The number of transformer layers to use",
    "nhead": "Number of Heads - The number of heads to use in Multi-headed Self-Attention"
}


"""
    LPTF Optional arguments:
    
        L          (int)     -- The total number of atoms in your lattice.
    
        Nh         (int)     -- Transformer token size. Input patches are projected to match the token size.
                                Note: When using an RNN subsampler this Nh MUST match the rnn's Nh.
    
        patch      (int)     -- Number of atoms input/predicted at once (patch size).
                                The Input sequence will have an effective length of L/patch.
            
        dropout    (float)   -- The amount of dropout to use in the transformer layers.
        
        num_layers (int)     -- The number of transformer layers to use.
        
        nhead     (int)     -- The number of heads to use in Multi-headed Self-Attention. This should divide Nh.
        
        subsampler (Sampler) -- The inner model to use for probability factorization. This is set implicitly
                                by including --rnn or --ptf arguments. 
"""
class LargePatchedTransformerConfig(NamedTuple):
    L: int = 64
    Nh: int = 128
    patch: str = "2x2"
    dropout: float = 0.0
    num_layers: int = 2
    nhead: int = 8

LargePatchedTransformerConfigDescription = {
    "L": "L - The total number of atoms in your lattice",
    "Nh": "Nh - Transformer token size. Input patches are projected to match the token size",
    "patch": "Patch - Number of atoms input/predicted at once (patch size)",
    "dropout": "Dropout - The amount of dropout to use in the transformer layers",
    "num_layers": "Num of Layers - The number of transformer layers to use",
    "nhead": "Number of Heads - The number of heads to use in Multi-headed Self-Attention"
}


"""
The following parameters can be chosen for the Rydberg Hamiltonian:

Lx                            			4
Ly                            			4
V                             			7.0
Omega                         			1.0
delta                         			1.0
"""
class RydbergConfig(NamedTuple):
    Lx: int = 4
    Ly: int = 4
    V: float = 7.0
    Omega: float = 1.0
    delta: float = 1.0

RydbergConfigDescription = {
    "Lx": "Lx - The number of lattice sites in the x-direction",
    "Ly": "Ly - The number of lattice sites in the y-direction",
    "V": "V - The Rydberg blockade radius",
    "Omega": "Omega - The Rabi frequency",
    "delta": "delta - The detuning"
}

"""
        Q          (int)     -- Number of minibatches per batch.
        K          (int)     -- size of each minibatch.
        B          (int)     -- Total batch size (should be Q*K).
        NLOOPS     (int)     -- Number of loops within the off_diag_labels function. Higher values save ram and
                                generally makes the code run faster (up to 2x). Note, you can only set this
                                as high as your effective sequence length. (Take L and divide by your patch size).
        steps      (int)     -- Number of training steps.
        lr         (float)   -- Learning rate.
        seed       (int)     -- Random seed for the run.
        sub_directory (str)  -- String to add to the end of the output directory (inside a subfolder). 
"""
class TrainConfig(NamedTuple):
    L: int = 64
    Q: int = 1
    K: int = 1024
    B: int = 1024
    NLOOPS: int = 16
    steps: int = 50000
    lr: float = 0.0005
    seed: int = 1234
    dir: str = "TF"
    sub_directory: str = "2x2"

TrainConfigDescription = {
    "L": "L - Total lattice size (8x8 would be L=64)",	
    "Q": "Q - Number of minibatches per batch",
    "K": "K - Size of each minibatch",
    "B": "B - Total batch size (should be Q*K)",
    "NLOOPS": "NLOOPS - Number of loops within the off_diag_labels function. Higher values save ram and generally makes the code run faster (up to 2x). Note, you can only set this as high as your effective sequence length. (Take L and divide by your patch size)",
    "steps": "Steps - Number of training steps",
    "lr": "Learning rate",
    "seed": "Seed - Random seed for the run",
    "dir": "Directory - Directory to save the output",
    "sub_directory": "Sub Directory - String to add to the end of the output directory (inside a subfolder)"
}


# Create dictionary that maps the config to its description

def get_widget(description, field_type, default_value, disabled=False):
        if field_type == int:
            return st.number_input(description, min_value=0, value=default_value, step=1, disabled=disabled)
        elif field_type == float:
            return st.number_input(description, min_value=0.00000, value=default_value, step=0.001, disabled=disabled)
        elif field_type == str:
            return st.text_input(description, value=default_value, disabled=disabled)

def get_sidebar_widget(description, field_type, default_value, disabled=False):
        if description != "Num Hidden Units":
            if field_type == int:
                return st.sidebar.number_input(description, min_value=0, value=default_value, step=1, disabled=disabled)
            elif field_type == float:
                return st.sidebar.number_input(description, min_value=0.00005, value=default_value, step=0.001, disabled=disabled)
            elif field_type == str:
                return st.sidebar.text_input(description, value=default_value, disabled=disabled)
    

def get_widget_group(config: NamedTuple, desc_dict, exclude_list: List[str], sidebar=False) -> Dict:  
    widget_group = {}
    field_defaults = config._field_defaults
    
    if sidebar:
        for field_name, field_type in get_type_hints(config).items():
            default_value = field_defaults.get(field_name, None)
            # description = field_name.replace("_", " ").title()
            description = desc_dict[field_name]
            
            widget_group[field_name] = get_sidebar_widget(description, field_type, default_value) if field_name not in exclude_list else get_sidebar_widget(description, field_type, default_value, True)
    else:
        for field_name, field_type in get_type_hints(config).items():
            default_value = field_defaults.get(field_name, None)
            # description = field_name.replace("_", " ").title()
            description = desc_dict[field_name]
            
            widget_group[field_name] = get_widget(description, field_type, default_value) if field_name not in exclude_list else get_widget(description, field_type, default_value, True)
    
    return widget_group

# Serializing/Deserializing the configuration
cwd = os.getcwd()
data_path = os.path.join(cwd, "src", "data")
os.makedirs(data_path, exist_ok=True)

rnn_path = os.path.join(data_path, "rnn_config.json")
ptf_path = os.path.join(data_path, "ptf_config.json")
lptf_path = os.path.join(data_path, "lptf_config.json")

def save_rnn(model_config: RNNConfig, vmc_config: VMCConfig, file_path: str = rnn_path):
    """Function to write the configuration to a json file"""
    with open(file_path, "w") as f:
        json.dump({"model_config": model_config._asdict(), "vmc_config": vmc_config._asdict()}, f, indent=4)


def save_ptf(model_config: PatchedTransformerConfig, train_config: TrainConfig, rydberg_config: RydbergConfig, file_path: str = ptf_path):
    """Function to write the configuration to a json file"""
    with open(file_path, "w") as f:
        json.dump({"model_config": model_config._asdict(), "train_config": train_config._asdict(), "hamiltonian_config": rydberg_config._asdict()}, f, indent=4)


def save_lptf(model_config: LargePatchedTransformerConfig, train_config: TrainConfig, rydberg_config: RydbergConfig, file_path: str = lptf_path):
    """Function to write the configuration to a json file"""
    with open(file_path, "w") as f:
        json.dump({"model_config": model_config._asdict(), "train_config": train_config._asdict(), "hamiltonian_config": rydberg_config._asdict()}, f, indent=4)


def load_config(record_type: ModelType):
    """Return json string for the selected model"""
    if record_type == ModelType.RNN:
        with open(rnn_path, "r") as f:
            return json.load(f)
    elif record_type == ModelType.PatchedTRANSFORMER:
        with open(ptf_path, "r") as f:
            return json.load(f)
    else:
        with open(lptf_path, "r") as f:
            return json.load(f)
        

# Extract Command Line Arguments from PatchedTransformer/LargePatchedTransformer
def extract_args(model_type: ModelType):
    # Check if the model type is PatchedTransformer or LargePatchedTransformer
    # Get configuration from the json file depending on the model type
    # Check for overlapping keys in the configuration objects
    # Break key values into a list of arguments/flags
    # Return the list of arguments/flags

    # Step 1
    config = load_config(ModelType.PatchedTRANSFORMER) if model_type == ModelType.PatchedTRANSFORMER else load_config(ModelType.LargePatchedTRANSFORMER)
    model_config = LargePatchedTransformerConfig(**config["model_config"])
    train_config = TrainConfig(**config["train_config"])
    rydberg_config = RydbergConfig(**config["hamiltonian_config"])

    # Step 2
    model_dict = model_config._asdict()
    train_dict = train_config._asdict()
    rydberg_dict = rydberg_config._asdict()

    # Step 3
    keys = set(model_dict.keys()).union(set(train_dict.keys())).union(set(rydberg_dict.keys()))

    # Step 4
    args = []
    model_flags = ["--ptf", "--lptf"]
    flags = ["--train", model_flags[model_type.value - 2], "--rydberg"] # model_flags[0] = "--ptf", model_flags[1] = "--lptf"
    config_arrange = [train_dict, model_dict, rydberg_dict]
    
    for flag, config in zip(flags, config_arrange):
        args.append(flag)
        for key in keys:
            if key in config:
                args.append(f"{key}={config[key]}")

    return args


# get transformer type model
def get_tf_model(args: List[str]):
    model, full_opt, opt_dict = build_model(args)
    return model, full_opt, opt_dict

def run_tf_model(model, full_opt, opt_dict):
    beta1=0.9
    beta2=0.999
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=opt_dict["TRAIN"].lr, 
        betas=(beta1,beta2)
    )

    mydir=setup_dir(opt_dict)
    orig_stdout = sys.stdout

    full_opt.save(mydir+"\\settings.json")
    f = open(mydir+'\\output.txt', 'w')
    sys.stdout = f
    try:
        reg_train(opt_dict,(model,optimizer),printf=True,mydir=mydir)
    except Exception as e:
        print(e)
        sys.stdout = orig_stdout
        f.close()
        1/0
    sys.stdout = orig_stdout
    f.close()