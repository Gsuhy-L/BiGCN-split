import torch
import torch.nn as nn
from Process.ggnn_model.layers import *
class GNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, gru_step, dropout_p):
        super(GNN,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.gru_step = gru_step
        self.GraphLayer = GraphLayer(
            input_dim = self.input_dim,
            output_dim = self.hidden_dim,
            act = torch.nn.Tanh(),
            dropout_p = self.dropout_p,
            gru_step = self.gru_step
        )
        self.ReadoutLayer = ReadoutLayer(
            input_dim = self.hidden_dim,
            output_dim = self.output_dim,
            act = torch.nn.Tanh(),
            dropout_p = self.dropout_p
        )
        self.layers = [self.GraphLayer, self.ReadoutLayer]


    def forward(self, data):

        activations = data.x

        adj = data.edge_index
        return_data = []
        return_data.append(activations)
        for layer in self.layers:
            print('acti_shape',activations.shape)
            hidden = layer(return_data[-1], adj)
            return_data.append(hidden)
            # hidden = layer(activations[-1], adj)
            # activations.append(hidden)
        embeddings = return_data[-2]
        outputs = return_data[-1]
        return outputs

