import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_prod_attention(queries, keys, values, mask=None):
    r"""
    :param queries: :math:`... \times l_q  \times d_k`
    :param keys:    :math:`... \times l_kv \times d_k`
    :param values:  :math:`... \times l_kv \times d_v`
    :param mask:    :math:`... \times l_q  \times l_kv`
    :type queries:  torch.Tensor(dtype=torch.float32)
    :type keys:     torch.Tensor(dtype=torch.float32)
    :type values:   torch.Tensor(dtype=torch.float32)
    :type mask:     torch.Tensor(dtype=torch.uint8)
    :return:        :math:`... \times l_q  \times d_v`
    :rtype:         torch.Tensor(dtype=torch.float32)
    """
    weights = queries.matmul(keys.transpose(-1, -2))  # ... x l_q x l_kv
    weights *= keys.size(-1) ** -0.5
    if mask is not None:
        if mask.dim() == weights.dim() - 1:
            mask = mask.unsqueeze(-2).expand_as(weights)
        weights[mask] = -float('inf')
    weights = F.softmax(weights, dim=-1)
    return weights.matmul(values)


class BigBirdMultiHeadAttention(nn.Module):
    def __init__(self, head_count=None, query_size=None, key_size=None, value_size=None,
                 key_size_per_head=None, value_size_per_head=None, global_size=None):
        super().__init__()
        self.head_count = head_count

        self.query_size = query_size
        self.key_size = query_size if key_size is None else key_size
        self.value_size = self.key_size if value_size is None else value_size

        self.key_size_per_head = self.key_size // self.head_count if key_size_per_head is None \
            else key_size_per_head
        self.value_size_per_head = self.value_size // self.head_count if value_size_per_head is None \
            else value_size_per_head

        ###
        self.global_size = global_size  # Size of the global context

        self.query_project = nn.Linear(self.query_size, self.head_count * self.key_size_per_head, bias=False)
        self.key_project = nn.Linear(self.key_size, self.head_count * self.key_size_per_head, bias=False)
        self.value_project = nn.Linear(self.value_size, self.head_count * self.value_size_per_head, bias=False)
        self.recombine = nn.Linear(self.head_count * self.value_size_per_head, self.value_size, bias=False)

        #### Global attention parameters
        self.global_query_project = nn.Linear(self.query_size, self.head_count * self.key_size_per_head, bias=False)
        self.global_key_project = nn.Linear(self.global_size, self.head_count * self.key_size_per_head, bias=False)
        self.global_value_project = nn.Linear(self.global_size, self.head_count * self.value_size_per_head, bias=False)

    def forward(self, queries, keys, values, global_context, mask=None):
        r"""
        :param queries: :math:`... \times l_q  \times d_q`
        :param keys:    :math:`... \times l_kv \times d_k`
        :param values:  :math:`... \times l_kv \times d_v`
        :param global_context: :math:`... \times l_global \times d_global`
        :param mask:    :math:`... \times l_q  \times l_kv`
        :type queries:  torch.Tensor(dtype=torch.float32)
        :type keys:     torch.Tensor(dtype=torch.float32)
        :type values:   torch.Tensor(dtype=torch.float32)
        :type global_context: torch.Tensor(dtype=torch.float32)
        :type mask:     torch.Tensor(dtype=torch.uint8)
        :return:        :math:`... \times l_q  \times d_v`
        :rtype:         torch.Tensor(dtype=torch.float32)
        """
        q_proj = self.query_project(queries).chunk(self.head_count, dim=-1)
        k_proj = self.key_project(keys).chunk(self.head_count, dim=-1)
        v_proj = self.value_project(values).chunk(self.head_count, dim=-1)

        global_q_proj = self.global_query_project(queries).chunk(self.head_count, dim=-1)
        global_k_proj = self.global_key_project(global_context).chunk(self.head_count, dim=-1)
        global_v_proj = self.global_value_project(global_context).chunk(self.head_count, dim=-1)

        #### Local attention
        local_att_applied = tuple(map(scaled_dot_prod_attention,
                                     q_proj, k_proj, v_proj, (mask for _ in range(self.head_count))))

        #### Global attention
        global_att_applied = tuple(map(scaled_dot_prod_attention,
                                      global_q_proj, global_k_proj, global_v_proj, (None for _ in range(self.head_count))))

        # Combine local and global attention outputs
        combined_attention = [local + global_attention for local, global_attention in zip(local_att_applied, global_att_applied)]
        combined_attention = torch.cat(combined_attention, dim=-1)

        return self.recombine(combined_attention)
