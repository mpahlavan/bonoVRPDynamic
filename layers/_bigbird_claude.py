import torch
import torch.nn as nn
from bigbird.core import SparseAttention

class BigBirdMHAttention(nn.Module):
    def __init__(self, heads, d_model, dropout):
        super().__init__()
        self.heads = heads  
        self.d_model = d_model
        
        # Linear transformations for query, key, and value
        self.query_lin = nn.Linear(d_model, d_model)
        self.key_lin = nn.Linear(d_model, d_model)  
        self.value_lin = nn.Linear(d_model, d_model)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Create BigBird sparse attention 
        self.attn = SparseAttention(
            num_heads=heads, 
            block=32, 
            window=128,
            global_block_indices=[1, 3, 5]
        )
        
        # Linear transformation for the output
        self.out_lin = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        # Input shape: (batch_size, sequence_length, d_model)
        batch, seq_len, _ = x.shape
        
        # Linear transformations for query, key, and value
        x = self.query_lin(x)
        x = x.view(batch, seq_len, self.heads, -1) 
        q = x.permute(0, 2, 1, 3)
        
        x = self.key_lin(x)
        k = x.view(batch, seq_len, self.heads, -1).permute(0, 2, 1, 3)       

        x = self.value_lin(x)
        v = x.view(batch, seq_len, self.heads, -1).permute(0, 2, 1, 3)                
        
        # Apply BigBird sparse attention 
        x = self.attn(q, k, v, attention_mask=mask)
        
        # Reshape and linear transformation for the output
        x = x.permute(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)        
        return self.out_lin(self.dropout(x))
