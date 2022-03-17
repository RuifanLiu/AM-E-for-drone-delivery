import torch
import numpy as np
from torch import nn
import math
from nets.graph_encoder import SkipConnection

class SkipConnection_EDGE(nn.Module):

    def __init__(self, module):
        super(SkipConnection_EDGE, self).__init__()
        self.module = module

    def forward(self, input):
        return input[0] + self.module(input)
    
    
    
        
class MultiHeadAttention_EDGE(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention_EDGE, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        
        self.E_val1 = nn.Parameter(torch.Tensor(n_heads, input_dim, 1))
        # self.E_val2 = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        
        # self.E_weight = nn.Parameter(torch.Tensor(1, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))
        

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        q, e = input
        if h is None:
            h = q  # compute self-attention
            

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)
        
        if e is not None:
            _, _, _, input_dim_e = e.size()
            eflat = e.contiguous().view(-1,input_dim)
            
            shp_e = (self.n_heads, batch_size, graph_size, graph_size, -1)
            E = torch.matmul(eflat,self.E_val1).view(shp_e).squeeze(-1)
            # V_e = torch.matmul(eflat,self.E_val2).view(shp_e).squeeze(-1)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * (torch.matmul(Q, K.transpose(2, 3)) ) + E

        # Optionally apply mask to prevent attention
        # if e is not None:
        #     mask = -10e9 * (1.0 - e)
        #     mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
        #     compatibility += mask
            
            # mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            # compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        # if mask is not None:
        #     attnc = attn.clone()
        #     attnc[mask] = 0
        #     attn = attnc
            
        # edge_feat = torch.matmul(attn, E.squeeze(-1))
        # edge_diag = torch.diagonal(edge_feat,dim1=2,dim2=3)[:,:,:,None]
        # edge = torch.matmul(edge_diag, self.E_weight)

        heads = torch.matmul(attn, V) #+ torch.sum(torch.mul(attn.unsqueeze(-1), V_e),3)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out

class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input

class MultiHeadAttentionLayer_EDGE(nn.Sequential):
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer_EDGE, self).__init__(
            SkipConnection_EDGE(
                MultiHeadAttention_EDGE(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )

class GraphAttentionEncoder_EDGE(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder_EDGE, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer_EDGE(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        # h = self.layers(h)
        
        node_feat, edge_feat = h
        
        for layer in self.layers:
            node_feat = layer([node_feat, edge_feat])

        return (
            node_feat,  # (batch_size, graph_size, embed_dim)
            node_feat.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


