import torch
import torch.nn as nn
import math
from typing import Union, Optional
import torch.nn.functional as F
import math

# From https://github.com/rabeehk/compacter

def glorot_normal(tensor: torch.Tensor):
    return torch.nn.init.xavier_normal_(tensor, gain=math.sqrt(2))


def glorot_uniform(tensor: torch.Tensor):
    return torch.nn.init.xavier_uniform_(tensor, gain=math.sqrt(2))

def init_ones(tensor):
    return torch.nn.init.ones_(tensor)

class LowRankLinear(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int = 1,
        bias: bool = True,
        w_init: str = "glorot-uniform",
    ):
        super(LowRankLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.bias = bias
        self.w_init = w_init
        self.W_left = nn.Parameter(
            torch.Tensor(size=(input_dim, rank)), requires_grad=True
        )
        self.W_right = nn.Parameter(
            torch.Tensor(size=(rank, output_dim)), requires_grad=True
        )
        if bias:
            self.b = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias:
            self.b.data = torch.zeros_like(self.b.data)
        if self.w_init == "glorot-uniform":
            self.W_left.data = glorot_uniform(self.W_left.data)
            self.W_right.data = glorot_uniform(self.W_right.data)
        elif self.w_init == "glorot-normal":
            self.W_left.data = glorot_normal(self.W_left.data)
            self.W_right.data = glorot_normal(self.W_right.data)
        else:
            raise ValueError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.W_left * self.W_right
        output = torch.matmul(input=x, other=W)
        if self.bias:
            output += self.b
        return output


def kronecker_product(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    # return torch.stack([torch.kron(ai, bi) for ai, bi in zip(a,b)], dim=0)
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    out = res.reshape(siz0 + siz1)
    return out


def kronecker_product_einsum_batched(A: torch.Tensor, B: torch.Tensor):
    """
    Batched Version of Kronecker Products
    :param A: has shape (b, a, c)
    :param B: has shape (b, k, p)
    :return: (b, ak, cp)
    """
    assert A.dim() == 3 and B.dim() == 3
    res = torch.einsum("bac,bkp->bakcp", A, B).view(
        A.size(0), A.size(1) * B.size(1), A.size(2) * B.size(2)
    )
    return res


def matvec_product(
    W: torch.Tensor,
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    phm_rule: Union[torch.Tensor],
    kronecker_prod=False,
) -> torch.Tensor:
    """
    Functional method to compute the generalized matrix-vector product based on the paper
    "Parameterization of Hypercomplex Multiplications (2020)"
    https://openreview.net/forum?id=rcQdycl0zyk
    y = Hx + b , where W is generated through the sum of kronecker products from the Parameterlist W, i.e.
    W is a an order-3 tensor of size (phm_dim, in_features, out_features)
    x has shape (batch_size, phm_dim*in_features)
    phm_rule is an order-3 tensor of shape (phm_dim, phm_dim, phm_dim)
    H = sum_{i=0}^{d} mul_rule \otimes W[i], where \otimes is the kronecker product
    """
    if kronecker_prod:
        H = kronecker_product(phm_rule, W).sum(0)
    else:
        H = kronecker_product_einsum_batched(phm_rule, W).sum(0)

    y = torch.matmul(input=x, other=H)
    if bias is not None:
        y += bias
    return y


class PHMLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        phm_dim: int,
        phm_rule: Union[None, torch.Tensor] = None,
        bias: bool = True,
        w_init: str = "phm",
        c_init: str = "random",
        learn_phm: bool = True,
        shared_phm_rule=False,
        factorized_phm=False,
        shared_W_phm=False,
        factorized_phm_rule=False,
        phm_rank=1,
        phm_init_range=0.0001,
        kronecker_prod=False,
    ) -> None:
        super(PHMLinear, self).__init__()
        #assert w_init in ["phm", "glorot-normal", "glorot-uniform", "normal"]
        #assert c_init in ["normal", "uniform"]
        assert (
            in_features % phm_dim == 0
        ), f"Argument `in_features`={in_features} is not divisble be `phm_dim`{phm_dim}"
        assert (
            out_features % phm_dim == 0
        ), f"Argument `out_features`={out_features} is not divisble be `phm_dim`{phm_dim}"
        self.in_features = in_features
        self.out_features = out_features
        self.learn_phm = learn_phm
        self.phm_dim = phm_dim
        self._in_feats_per_axis = in_features // phm_dim
        self._out_feats_per_axis = out_features // phm_dim
        self.phm_rank = phm_rank
        self.phm_init_range = phm_init_range
        self.kronecker_prod = kronecker_prod
        self.shared_phm_rule = shared_phm_rule
        self.factorized_phm_rule = factorized_phm_rule
        self.bias_flag = bias
        self.w_init = w_init
        self.c_init = c_init
        self.shared_W_phm = shared_W_phm
        self.factorized_phm = factorized_phm
        if not self.shared_phm_rule:
            if self.factorized_phm_rule:
                self.phm_rule_left = nn.Parameter(
                    torch.FloatTensor(phm_dim, phm_dim, 1), requires_grad=learn_phm
                )
                self.phm_rule_right = nn.Parameter(
                    torch.FloatTensor(phm_dim, 1, phm_dim), requires_grad=learn_phm
                )
            else:
                self.phm_rule = nn.Parameter(
                    torch.FloatTensor(phm_dim, phm_dim, phm_dim),
                    requires_grad=learn_phm,
                )
        if not self.shared_W_phm:
            if self.factorized_phm:
                self.W_left = nn.Parameter(
                    torch.Tensor(
                        size=(phm_dim, self._in_feats_per_axis, self.phm_rank)
                    ),
                    requires_grad=True,
                )
                self.W_right = nn.Parameter(
                    torch.Tensor(
                        size=(phm_dim, self.phm_rank, self._out_feats_per_axis)
                    ),
                    requires_grad=True,
                )
            else:
                self.W = nn.Parameter(
                    torch.Tensor(
                        size=(
                            phm_dim,
                            self._in_feats_per_axis,
                            self._out_feats_per_axis,
                        )
                    ),
                    requires_grad=True,
                )
        if self.bias_flag:
            self.b = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("b", None)
        self.reset_parameters()

    def init_W(self):
        if self.w_init == "glorot-normal":
            if self.factorized_phm:
                for i in range(self.phm_dim):
                    self.W_left.data[i] = glorot_normal(self.W_left.data[i])
                    self.W_right.data[i] = glorot_normal(self.W_right.data[i])
            else:
                for i in range(self.phm_dim):
                    self.W.data[i] = glorot_normal(self.W.data[i])
        elif self.w_init == "glorot-uniform":
            if self.factorized_phm:
                for i in range(self.phm_dim):
                    self.W_left.data[i] = glorot_uniform(self.W_left.data[i])
                    self.W_right.data[i] = glorot_uniform(self.W_right.data[i])
            else:
                for i in range(self.phm_dim):
                    self.W.data[i] = glorot_uniform(self.W.data[i])
        elif self.w_init == "normal":
            if self.factorized_phm:
                for i in range(self.phm_dim):
                    self.W_left.data[i].normal_(mean=0, std=0.02)
                    self.W_right.data[i].normal_(mean=0, std=0.02)
            else:
                for i in range(self.phm_dim):
                    self.W.data[i].normal_(mean=0.0, std=0.02)
        else:
            #for i in range(self.phm_dim):
            #        self.W_left.data[i] = init_ones(self.W_left.data[i])
            #        self.W_right.data[i] = init_ones(self.W_right.data[i])
            #        #self.W_right.data[i].normal_(mean=0, std=0.02)
            for i in range(self.phm_dim):
                self.W_left.data[i] = init_ones(self.W_left.data[i]) / (self._in_feats_per_axis * self.phm_dim)
                self.W_right.data[i] = init_ones(self.W_right.data[i])
            #raise ValueError

    def reset_parameters(self):
        if not self.shared_W_phm:
            self.init_W()

        if self.bias_flag:
            self.b.data = torch.zeros_like(self.b.data)

        if not self.shared_phm_rule:
            if self.factorized_phm_rule:
                if self.c_init == "uniform":
                    self.phm_rule_left.data.uniform_(-1, 1)
                    self.phm_rule_right.data.uniform_(-1, 1)
                elif self.c_init == "normal":
                    self.phm_rule_left.data.normal_(std=self.phm_init_range)
                    self.phm_rule_right.data.normal_(std=self.phm_init_range)
                else:
                    raise NotImplementedError
            else:
                if self.c_init == "uniform":
                    self.phm_rule.data.uniform_(-1, 1)
                elif self.c_init == "normal":
                    self.phm_rule.data.normal_(mean=0, std=0.02)
                else:
                    self.phm_rule.data = init_ones(self.phm_rule.data) / (self.phm_dim * self.phm_dim * self.phm_dim)
                    #raise NotImplementedError

    def set_phm_rule(self, phm_rule=None, phm_rule_left=None, phm_rule_right=None):
        """If factorized_phm_rules is set, phm_rule is a tuple, showing the left and right
        phm rules, and if this is not set, this is showing  the phm_rule."""
        if self.factorized_phm_rule:
            self.phm_rule_left = phm_rule_left
            self.phm_rule_right = phm_rule_right
        else:
            self.phm_rule = phm_rule

    def set_W(self, W=None, W_left=None, W_right=None):
        if self.factorized_phm:
            self.W_left = W_left
            self.W_right = W_right
        else:
            self.W = W

    def forward(
        self, x: torch.Tensor, phm_rule: Union[None, nn.ParameterList] = None
    ) -> torch.Tensor:

        if self.factorized_phm:
            W = torch.bmm(self.W_left, self.W_right)
        if self.factorized_phm_rule:
            phm_rule = torch.bmm(self.phm_rule_left, self.phm_rule_right)
        y = matvec_product(
            W=W if self.factorized_phm else self.W,
            x=x,
            bias=self.b,
            phm_rule=phm_rule if self.factorized_phm_rule else self.phm_rule,
            kronecker_prod=self.kronecker_prod,
        )
        #print("Input", x)
        #print("Output", y)
        return y
