import os
import ast
import copy
import json
import torch
import numpy as np
from enum import Enum
import streamlit as st
from typing import NamedTuple, get_type_hints, List, Dict, Union
from src.tf_models.model_builder import *

class ModelType(Enum):
    RNN = 1
    Transformer = 2


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
    learning_rate: float = 0.005
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
class TransformerConfig(NamedTuple):
    L: int = 64
    # Nh: int = 128
    patch: str = "1x1"
    # dropout: float = 0.0
    num_layers: int = 2
    nhead: int = 8

TransformerConfigDescription = {
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
    sub_directory: str = "1x1"

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
        # if isinstance(description, str) and description.split()[0] == "Patch":
        #     pass
        # else:
        if field_type == int:
            return st.number_input(description, min_value=0, value=default_value, step=1, disabled=disabled)
        elif field_type == float:
            return st.number_input(description, min_value=0.00000, value=default_value, step=0.00001, disabled=disabled)
        elif field_type == str:
            return st.text_input(description, value=default_value, disabled=disabled)

def get_sidebar_widget(description, field_type, default_value, disabled=False):
        if description != "Num Hidden Units":
            if field_type == int:
                return st.sidebar.number_input(description, min_value=0, value=default_value, step=1, disabled=disabled)
            elif field_type == float:
                return st.sidebar.number_input(description, min_value=0.00005, value=default_value, disabled=disabled)
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
            if field_name != "patch":
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
tf_path = os.path.join(data_path, "tf_config.json")
lptf_path = os.path.join(data_path, "lptf_config.json")

def save_rnn(model_config: RNNConfig, vmc_config: VMCConfig, file_path: str = rnn_path):
    """Function to write the configuration to a json file"""
    with open(file_path, "w") as f:
        json.dump({"model_config": model_config._asdict(), "vmc_config": vmc_config._asdict()}, f, indent=4)


def save_ptf(model_config: TransformerConfig, train_config: TrainConfig, rydberg_config: RydbergConfig, file_path: str = tf_path):
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
    elif record_type == ModelType.Transformer:
        with open(tf_path, "r") as f:
            return json.load(f)
    # else:
    #     with open(lptf_path, "r") as f:
    #         return json.load(f)
        

# Extract Command Line Arguments from PatchedTransformer/LargePatchedTransformer
def extract_args(model_type: ModelType):
    # Check if the model type is PatchedTransformer or LargePatchedTransformer
    # Get configuration from the json file depending on the model type
    # Check for overlapping keys in the configuration objects
    # Break key values into a list of arguments/flags
    # Return the list of arguments/flags

    # Step 1
    config = load_config(ModelType.Transformer)
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


compare_arrs = lambda x, y: (x == y).all()

def state_flipper(idx, s_state):
    new_state = copy.deepcopy(s_state)
    new_state[idx] = 1 - new_state[idx]
    return new_state

state = np.random.randint(0, 2, 16)
flip_idex = np.random.randint(0, 16, 1)


def fake_logpsi(flipped_state, params, model):
    if np.array_equal(flipped_state, state):
        print("something is wrong")
        return 25
    
    lgp = np.log(np.mean(flipped_state)) + params + model
    return lgp


def fake_step_fn_transverse(i, holder, Omega, log_psi, params, model):
    f_state, output = holder
    flipped_state = state_flipper(i, f_state)
    flipped_logpsi = fake_logpsi(flipped_state, params, model)
    output += - 0.5 * Omega * np.exp(flipped_logpsi - log_psi) # Something about the Rabi frequency
    return f_state, output


def check_transverse_fn(user_func):
    print("checking to see if tranverse_fn works")
    Omega = 1.0
    log_psi = 0.00002
    params = .00005
    model = .00007
    output = 0
    i = np.random.randint(0, 16, 1)
    state_cp_2 = copy.deepcopy(state)
    state_cp_1 = copy.deepcopy(state)

    # try checker function
    check_arr, check_out = fake_step_fn_transverse(i, (state_cp_2, output), Omega, log_psi, params, model)
    user_arr, user_out = user_func(i, (state_cp_1, output))


    arrs_eq = (check_arr == user_arr).all()
    outs_eq = check_out == user_out
    return(arrs_eq and outs_eq)


def check_flip_state(user_func):
    print("checking to see if flip_state works")
    c_state = np.random.randint(0, 2, 16)

    # we have to do this because arrays are passed by reference
    # we want independent results
    c_state_cp = copy.deepcopy(c_state) 

    correct_result = state_flipper(flip_idex, c_state)
    user_result =  user_func(flip_idex, c_state_cp)

    return (user_result == correct_result).all()



def extract_loc_e(code: str) -> bool:
    # Parse the provided code into an AST
    tree = ast.parse(code)
    
    # Initialize the flag to check if the student has passed the test
    passed_test = False
    
    # Define a visitor class to visit nodes in the AST
    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            nonlocal passed_test
            # Check if the assigned variable is 'loc_e'
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'loc_e':
                print([n for n in node.targets])
                # print(node.value.left)
                # Check if the value is a BinOp (binary operation) representing a sum
                # if isinstance(node.value, ast.BinOp):
                if isinstance(node.value, ast.Assign):
                    # Check if the left and right sides are also sums
                    terms = [node.value.left, node.value.right]
                    # for l, term in enumerate(terms):
                    #     print(l)
                    #     print(term)
                    #     print(isinstance(term, ast.BinOp))
                    #     print(node.value.left.right.id)

                    if all(isinstance(term, ast.BinOp) for term in terms):
                        # Extract the actual variable names and check if they match
                        variables = [terms[0].left, terms[0].right, terms[1].right]
                        # print(f"{variables=}")
                        variable_names = {var.id for var in variables if isinstance(var, ast.Name)}
                        expected_names = {'interaction_term', 'transverse_field', 'chemical_potential'}
                        if variable_names == expected_names:
                            passed_test = True
    
    # Create an instance of the visitor and visit the parsed AST
    visitor = Visitor()
    visitor.visit(tree)
    
    return passed_test


class LineCollector(ast.NodeVisitor):
    def __init__(self):
        self.target_line = None

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'loc_e':
            self.target_line = node.lineno
        self.generic_visit(node)


def meets_cond(target_str: str):
    terms = target_str.split()
    if len(terms) < 7:
        return False
    
    terms_target = set([terms[i] for i in (2, 4, 6)])

    target_0 = "loc_e"
    target_1 = "="
    target_3_5 = "+"
    targets_2_4_6 = set(["interaction_term", "transverse_field", "chemical_potential"])

    all_met = terms[0] == target_0 and \
        terms[1] == target_1 and \
        terms[3] == target_3_5 and \
        terms[5] == target_3_5 and \
        terms_target == targets_2_4_6
    
    return all_met