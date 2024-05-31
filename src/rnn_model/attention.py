import math
import flax.linen as nn
import jax.numpy as jnp


def expand_mask(mask):
    assert mask.dim >= 2, "Mask should have at least two dimensions"
    if mask.dim == 3:
        mask = mask.unsqueeze(1)
    while mask.dim < 4:
        mask = mask.unsqueeze(0)
    return mask


def scaled_dot_product(q, k, v, mask = None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)

    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)

    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)   
    return values, attention


class MultiheadAttention(nn.Module):
    embed_dim: int # Output dimension
    num_heads: int # number of parallel heads


    def setup(self):
        self.qkv_proj = nn.Dense(
            3 * self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros
        )

        self.out_proj = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros
        )

    def __call__(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape
        mask = expand_mask(mask) if mask is not None else None

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        qkv = qkv.transpose((0, 2, 1, 3)) # [batch_size, num_heads, seq_length, embed_dim]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.transpose((0, 2, 1, 3)) # [batch_size, seq_length, num_heads, embed_dim]
        values = values.reshape(batch_size, seq_length, embed_dim)
        output = self.out_proj(values)

        # clean up
        del qkv, q, k, v, values, batch_size, seq_length, embed_dim

        return output, attention
    

class EncoderBlock(nn.Module):
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float

    def setup(self):
        self.self_attn = MultiheadAttention(self.input_dim, self.num_heads)

        # MLP
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(rate=self.dropout_prob),
            nn.relu,
            nn.Dense(self.input_dim),
        ]

        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(rate=self.dropout_prob)

    def __call__(self, x, mask=None, train=True):

        # attention block
        attn_output, _ = self.self_attn(x, mask)
        x += self.dropout(attn_output, deterministic=not train)
        x = self.norm1(x)

        # feedforward block
        mlp_output = x
        for layer in self.linear:
            mlp_output = layer(mlp_output) if not isinstance(layer, nn.Dropout) else layer(mlp_output, deterministic=not train)

        x += self.dropout(mlp_output, deterministic=not train)
        x = self.norm2(x)

        return x
    

class TransformerEncoder(nn.Module):
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