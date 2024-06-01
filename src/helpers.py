from enum import Enum
import streamlit as st
from typing import NamedTuple, get_type_hints, List, Dict
from pydantic import BaseModel

class ModelType(Enum):
    RNN = 1
    PatchedTRANSFORMER = 2
    LargePatchedTRANSFORMER = 3


class RNNConfig(NamedTuple):
    output_dim: int = 2
    num_hidden_units: int = 64

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
    Nh: int = 128
    patch: str = "2x2"
    dropout: float = 0.1
    num_layers: int = 2
    nhead: int = 8
    repeat_pre: bool = False


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
    patch: str = "3x3"
    dropout: float = 0.1
    num_layers: int = 2
    nhead: int = 8
    repeat_pre: bool = False


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
"""
class TrainConfig(NamedTuple):
    Q: int = 1
    K: int = 256
    B: int = 256
    NLOOPS: int = 1
    steps: int = 50000
    lr: float = 5e-4
    seed: int = 1234


def get_widget(description, field_type, default_value, disabled=False):
        if field_type == int:
            return st.number_input(description, min_value=0, value=default_value, step=1, disabled=disabled)
        elif field_type == float:
            return st.number_input(description, min_value=0.00005, value=default_value, step=0.001, disabled=disabled)

def get_sidebar_widget(description, field_type, default_value, disabled=False):
        if description != "Num Hidden Units":
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