import torch
import torch.nn as nn
from Process.ggnn_model_1.layers import *
import torch.nn.functional as F
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
            # output_dim = self.output_dim,
            act = torch.nn.Tanh(),
            dropout_p = self.dropout_p
        )
        self.layers = [self.GraphLayer, self.ReadoutLayer]

        #self.fc=torch.nn.Linear((output_dim+hid_feats)*2,4)



    def forward(self, feature, root, rootindex, support, mask):

        activations = [feature]


        for layer in self.layers:

            hidden = layer(activations[-1], root, rootindex, support, mask)

            activations.append(hidden)
        embeddings = activations[-2]
        outputs = activations[-1]
        #x = F.log_softmax(outputs, dim=1)

        return outputs,embeddings

