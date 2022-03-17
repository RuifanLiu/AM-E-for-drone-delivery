import torch
from torch import nn

from nets.graph_encoder import GraphAttentionEncoder
from nets.graph_encoder_edge import GraphAttentionEncoder_EDGE
from nets.graph_encoder_egat import GraphAttentionEncoder_EGAT
from nets.graph_encoder_hetero import GraphAttentionEncoder_HETERO


class CriticNetwork(nn.Module):
    
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        n_layers,
        encoder_normalization,
        attention_type='Kool'
    ):
        super(CriticNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        
        self.attention_type = attention_type


        if self.attention_type == 'Kool':
            self.encoder = GraphAttentionEncoder(
                n_heads=8,
                embed_dim=embedding_dim,
                n_layers=n_layers,
                normalization=encoder_normalization
            )
            
        elif self.attention_type == 'withedge1':
            self.init_embed_edge = nn.Linear(1, embedding_dim)
            self.encoder = GraphAttentionEncoder_EDGE(
                n_heads=8,
                embed_dim=embedding_dim,
                n_layers=n_layers,
                normalization=encoder_normalization
            )
            
        elif self.attention_type == 'withedge2':
            self.init_embed_edge = nn.Linear(1, embedding_dim)
            self.encoder = GraphAttentionEncoder_EGAT(
                n_heads=8,
                embed_dim=embedding_dim,
                n_layers=n_layers,
                normalization=encoder_normalization
            )
            
        elif self.attention_type == 'heterogeneous':
            self.encoder = GraphAttentionEncoder_HETERO(
                n_heads=8,
                embed_dim=embedding_dim,
                n_layers=n_layers,
                normalization=encoder_normalization
            )
        # self.encoder = GraphAttentionEncoder(
        #     n_heads=8,
        #     embed_dim=embedding_dim,
        #     n_layers=n_layers,
        #     normalization=encoder_normalization
        # )

        self.init_embed_pick = nn.Linear(2, embedding_dim)
        self.init_embed_delivery = nn.Linear(2, embedding_dim)
        self.init_embed_station = nn.Linear(2, embedding_dim)
        self.init_embed_depot = nn.Linear(2, embedding_dim)


        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """

        :param inputs: (batch_size, graph_size, input_dim)
        :return:
        """
        

        _, graph_embeddings = self.encoder(self._init_embed(inputs))
        return self.value_head(graph_embeddings)
    
    
    def _init_embed(self, input):

        batchsize = input['loc'].size(0)

        n_loc = input['loc'].size(1)  # input['loc']: [batch_size, graph_size, 2]
        feature_pick = input['loc'][:, :n_loc // 2, :]
        feature_delivery = input['loc'][:, n_loc // 2:, :]
        feature_station = input['station']
        feature_depot = input['depot'][:, None, :]
        
        embed_pick = self.init_embed_pick(feature_pick)
        embed_delivery = self.init_embed_delivery(feature_delivery)
        embed_station = self.init_embed_station(feature_station)
        embed_depot = self.init_embed_depot(feature_depot)
        
        if self.attention_type == 'Kool' or self.attention_type == 'heterogeneous':
            return torch.cat([embed_pick, embed_delivery, embed_station, embed_depot], 1)
        else:
            
            loc_all =torch.cat([input['loc'],input['station'],input['depot'][:, None, :]],1)
            
            n_node = loc_all.size(1)
            
            x_loc = loc_all[:,:,None,:].expand(-1,-1,n_node,-1)
            y_loc = loc_all[:,None,:,:].expand(-1,n_node,-1,-1)
            
            dis_matrix = (x_loc - y_loc).norm(p=2, dim=-1)
            
            adj_matrix = torch.ones([batchsize,n_node,n_node],device=loc_all.device)
            adj_matrix[:, :n_loc, :n_loc] = 0
            adj_matrix[:, :n_loc//2, n_loc//2:n_loc] = torch.eye(n_loc//2)
            adj_matrix[:, n_loc//2:n_loc, :n_loc//2] = 1 - torch.eye(n_loc//2)
            
            
            edge_feat = torch.cat([dis_matrix[:,:,:,None], adj_matrix[:,:,:,None]], -1)
            edge_embedding = self.init_embed_edge(adj_matrix[:,:,:,None])
            # edge_embedding = self.init_embed_edge(edge_feat)
            # adj_mat1 = torch.zeros([batchsize,n_loc//2,n_loc//2])
            # adj_mat2 = torch.eye([batchsize,n_loc//2])
            # adj_mat2 = torch.zeros([batchsize,n_loc//2,n_loc//2])
            
            return torch.cat([embed_pick, embed_delivery, embed_station, embed_depot], 1), edge_embedding
        
        # return torch.cat([embed_pick, embed_delivery, embed_depot, embed_station], 1)        
