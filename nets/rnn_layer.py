# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:57:47 2022

@author: s313488
"""
import torch
from torch import nn

class rnn_layer(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_dim, 
                 n_layers
    ):
        super(rnn_layer, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Initializing hidden state for first input using method defined below
    
    def forward(self, x):
        # Passing in the input and hidden state into the model and obtaining outputs
        out, self.hidden = self.rnn(x, self.hidden)
        
        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        self.hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to('cuda')
            
