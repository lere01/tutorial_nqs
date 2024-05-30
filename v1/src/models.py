# src/models.py

import flax.linen as nn
from definitions import ModelType, ModelConfigType, RNNModelProtocol, TransformerModelProtocol, RNNConfigType, TransformerConfigType, QuantumModelProtocol
from attention import EncoderBlock
from errors import ModelTypeError




class RNNModel(nn.Module):
    """
    Neural Network for parameterizing our wavefunction.
    """
    output_dim: int
    num_hidden_units: int

    def setup(self):
      # Initialize the GRU cell with the specified number of hidden units
      gru_cell = nn.GRUCell(
          name='gru_cell',
          features=self.num_hidden_units,
          # kernel_init = jnn.initializers.glorot_uniform()
      )
      self.rnn = nn.RNN(gru_cell, return_carry=True)
      self.dense = nn.Dense(
          self.output_dim,
          name = 'dense_layer',
          # kernel_init = jnn.initializers.glorot_uniform()
      )


    
    def __call__(self, x, initial_carry=None):
        # Apply GRU layers
        carry, x = self.rnn(x, initial_carry = initial_carry)

        # Output layer
        x = self.dense(x)

        return carry, x




class TransformerModel(nn.Module):
    num_layers: int
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float

    def setup(self):
        self.layers = [EncoderBlock(self.input_dim, self.num_heads, self.dim_feedforward, self.dropout_prob) for _ in range(self.num_layers)]

    def __call__(self, x, mask=None, train=True):
        for layer in self.layers:
            x = layer(x, mask, train)
        return x
    
    def get_attention_maps(self, x, mask=None, train=True):
        attention_maps = []
        for layer in self.layers:
            _, attention = layer.self_attn(x, mask)
            attention_maps.append(attention)
            x = layer(x, mask=mask, train=train)
        
        return attention_maps
    


class ModelFactory:
    
    def __init__(self):
        self.models = {
            ModelType.RNN: RNNModel,
            ModelType.Transformer: TransformerModel
        }

    def set_model(self, model_type: ModelType) -> ModelConfigType:
        """
        Create a model based on the given model type.

        Args:
            model_type (ModelType): The type of model to create.

        Returns:
            ModelConfig: The created model.
        """
        self.model = self.models[model_type]
    

    def get_model(self, model_config: ModelConfigType) -> RNNModel | TransformerModel:
        """
        Initialize the model with the given configuration.

        Args:
            model_config (ModelConfig): The configuration for the model.

        Returns:
            RNN or Transformer: The initialized model.
        """

        valid_config = isinstance(self.model, RNNModelProtocol) and isinstance(model_config, RNNConfigType) or \
                        isinstance(self.model, TransformerModelProtocol) and isinstance(model_config, TransformerConfigType)
        

        if valid_config:
            return self.model(model_config)
        else:
            raise ValueError("Incompatible model configuration. Ensure that your model and configuration match.")
        


def get_model(model_type: ModelType) -> QuantumModelProtocol:
    if model_type.value == 1:
        return RNNModel
    elif model_type.value == 2:
        return TransformerModel
    else:
        raise ModelTypeError(model_type)