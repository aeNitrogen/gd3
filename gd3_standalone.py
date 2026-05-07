"""
Casted linear and the trunc_normal initialization have been taken from https://github.com/SamsungSAILMontreal/TinyRecursiveModels
Author: Thorben Comes (Ann Comes)
"""
import einops
import torch
from torch import nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
nn = torch.nn


@dataclass
class gd3neoConfig:
    matrix_size: int
    glu_size: int
    input_size: int
    output_size: int
    inner_size: int
    matrix_bias: bool = False
    dropout: float = 0.1
    block_width: int = 4
    block_height: int = 4
    init_name: str = "width aware initialization"
    norm: str = "Pre"
    norm_type: str = "rmsnorm"

############################################### Initialization #########################################################


# Truncated LeCun normal init
def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


def initialization_constructor(tensor: torch.Tensor, in_features, out_features, config, c=1.0, uniform=False, sqrt=False, factors=None, block = None, in_factor=0.5):
    if uniform:
        tensor.uniform_(-1, 1)
    else:
        tensor.std()

    if sqrt:
        tensor = torch.sqrt(torch.abs(tensor)) * torch.sign(tensor)

    if factors is not None:
        if factors == "out":
            tensor = tensor * (1/out_features)**0.5
        if factors == "in":
            tensor = tensor * (1/in_features)**in_factor
        if factors == "both":
            tensor = tensor * (1/(out_features + in_features))**0.5

    if block is not None:
        if block == "normal":
            tensor = tensor * config.block_width
        if block == "div":
            tensor = tensor / config.block_width
        if block == "square":
            tensor = tensor * (config.block_width)**2
        if block == "div_square":
            tensor = tensor / (config.block_width)**2
        if block == "sqrt":
            tensor = tensor * (1/config.block_width)**0.5
        if block == "vanilla_combined":
            tensor = tensor * (1/(config.block_width + in_features))**0.5
        if type(block) == float:
            tensor = tensor * (1 / config.block_width) ** block

    tensor = tensor * c
    return tensor.clip_(-3, 3)


def get_init(init_name, tensor, in_features, out_features, config):
    if init_name == "lecun":
        return trunc_normal_init_(tensor, std=1.0 / (in_features ** 0.5))
    elif init_name == "lecun_w":
        return trunc_normal_init_(tensor, std=1.0 / (in_features ** 0.5)) * (1/config.block_width)**0.5
    elif init_name == "lecun_w_quadratic":
        return trunc_normal_init_(tensor, std=1.0 / (in_features ** 0.5)) * (1/config.block_width)**0.25
    elif init_name == "width aware square_root":
        return initialization_constructor(tensor, in_features, out_features, config, c=0.4, uniform=True, sqrt=True, factors="in", in_factor=0.5, block=0.25)
    elif init_name[:4] == "width aware initialization":
        return initialization_constructor(tensor, in_features, out_features, config, c=0.4, uniform=True, sqrt=False, factors="in",
                                          in_factor=0.5, block=0.25)
    else:
        raise NotImplementedError

################################################ Normalization #########################################################


def get_normalization_post(name, size):
    """normalize over last dimension (dim=-1)"""
    name = name.lower()
    if name == "layernorm":
        return nn.LayerNorm(normalized_shape=size)
    elif name == "layernorm_unbiased":
        return nn.LayerNorm(normalized_shape=size, bias=False)
    elif name == "instancenorm1d":
        return nn.InstanceNorm1d(size)
    elif name == "rmsnorm":
        return nn.RMSNorm(size)
    elif name == "rmsnorm_no_element_affine":
        return nn.RMSNorm(size, elementwise_affine=False)
    else:
        raise NotImplementedError

####################################################  Layer  ####################################################

class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool, config=None, init_name: str = "default"):
        super().__init__()
        # print("used init", init_name)
        self.weight = nn.Parameter(
            get_init(init_name, torch.empty((out_features, in_features)), in_features, out_features, config)
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype),
                        bias=self.bias.to(input.dtype) if self.bias is not None else None)


class GD3(nn.Module):
    def __init__(self, gd3_config: gd3neoConfig):
        super().__init__()
        bias = gd3_config.bias
        mlp_size = gd3_config.inner_size - gd3_config.glu_size - gd3_config.matrix_size

        block_count = math.ceil(gd3_config.matrix_size / gd3_config.block_height)
        matrix_output_element_count = block_count * gd3_config.block_height  # output size of matrix multiplication
        matrix_el_count = matrix_output_element_count * gd3_config.block_width  # all non-zero matrix entries
        matrix_input_element_count = block_count * gd3_config.block_width  # input size of matrix multiplication

        real_hidden_size = mlp_size + gd3_config.glu_size + matrix_output_element_count  # real size of inner layer

        initial_projection_out_elements = mlp_size + 2 * gd3_config.glu_size

        matrix_part_total = matrix_el_count + matrix_input_element_count

        self.in_projection = CastedLinear(gd3_config.input_size, initial_projection_out_elements, bias=bias)
        self.matrix_projection = CastedLinear(gd3_config.input_size, matrix_part_total, gd3_config.matrix_bias,
                                              config=gd3_config, init_name=gd3_config.init_name)
        self.out_projection = CastedLinear(real_hidden_size, gd3_config.output_size, bias=bias)

        self.mlp_cut = mlp_size
        self.act_cut = mlp_size + gd3_config.glu_size
        self.matrix_cut = matrix_input_element_count

        self.height = gd3_config.block_height
        self.width = gd3_config.block_width

        if gd3_config.dropout != False and gd3_config.dropout != 0.0:
            self.dropout_enabled = True
            self.dropout = nn.Dropout(gd3_config.dropout)
        else:
            self.dropout_enabled = False

        self.matrix_input_element_count = matrix_input_element_count

        if gd3_config.norm.lower() == "pre":
            self.pre_norm = get_normalization_post(gd3_config.norm_type, self.width)
            self.post_norm = None
        elif gd3_config.norm.lower() == "post":
            self.post_norm = get_normalization_post(gd3_config.norm_type, matrix_output_element_count)
            self.pre_norm = None
        else:
            self.pre_norm = None
            self.post_norm = None

    def forward(self, x):
        in_projection = self.in_projection(x)
        matrix_elements_all = self.matrix_projection(x)

        mlp_glu = F.silu(in_projection[:, :, :self.act_cut])
        glu = mlp_glu[:, :, self.mlp_cut:] * in_projection[:, :, self.act_cut:]

        matrix_in_elements = F.silu(matrix_elements_all[:, :, :self.matrix_input_element_count])
        batch_dims = matrix_in_elements.shape[:-1]

        matrix_entries = matrix_elements_all[:, :, self.matrix_input_element_count:]
        matrix_mul_vec = matrix_in_elements.view(*batch_dims, -1, self.width).repeat(1, 1, 1, self.height).view(
            *batch_dims, -1, self.width)

        if self.pre_norm is not None:
            matrix_entries = self.pre_norm(matrix_entries.view(*batch_dims, -1, self.width))
        else:
            matrix_entries = matrix_entries.view(*batch_dims, -1, self.width)

        pre_sum = matrix_mul_vec * matrix_entries
        matrix_out = torch.sum(pre_sum, dim=-1)

        if self.post_norm is not None:
            matrix_out = self.post_norm(matrix_out)
        concat = torch.cat((matrix_out, glu, mlp_glu[:, :, :self.mlp_cut]), dim=-1)

        if self.dropout_enabled:
            output = self.out_projection(self.dropout(concat))
        else:
            output = self.out_projection(concat)
        return output
