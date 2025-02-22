import torch
import torch.nn as nn
from Process.ggnn_model_1.inits import glorot,xavier
import torch as th
import copy

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
class gru_unit(nn.Module):
    def __init__(self, output_dim, act, dropout_p):
        super(gru_unit,self).__init__()
        self.output_dim = output_dim
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(1-self.dropout_p)
        self.act = act
        self.z0_weight = glorot([self.output_dim, self.output_dim]) # nn.Parameter(torch.randn(self.output_dim, self.output_dim))
        self.z1_weight = glorot([self.output_dim, self.output_dim])
        self.r0_weight = glorot([self.output_dim, self.output_dim])
        self.r1_weight = glorot([self.output_dim, self.output_dim])
        self.h0_weight = glorot([self.output_dim, self.output_dim])
        self.h1_weight = glorot([self.output_dim, self.output_dim])
        self.z0_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.z1_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.r0_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.r1_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.h0_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.h1_bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self,support, x, mask):
        support = self.dropout(support)

        a = torch.matmul(support, x)
        # updata gate
        z0 = torch.matmul(a, self.z0_weight) + self.z0_bias
        z1 = torch.matmul(x, self.z1_weight) + self.z1_bias
        z = torch.sigmoid(z0+z1)
        # reset gate
        r0 = torch.matmul(a, self.r0_weight) + self.r0_bias
        r1 = torch.matmul(x, self.r1_weight) + self.r1_bias
        r = torch.sigmoid(r0+r1)
        # update embeddings
        h0 = torch.matmul(a, self.h0_weight) + self.h0_bias
        h1 = torch.matmul(r*x, self.h1_weight) + self.h1_bias
        h = self.act(mask * (h0 + h1))
        output = h*z + x*(1-z)




        return output


class GraphLayer(nn.Module):
    """Graph layer."""
    def __init__(self,
                      input_dim,
                      output_dim,
                      act=nn.Tanh(),
                      dropout_p = 0.,
                      gru_step = 2):
        super(GraphLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(1-self.dropout_p)
        self.gru_step = gru_step
        self.gru_unit = gru_unit(output_dim = self.output_dim,
                                 act = self.act,
                                 dropout_p = self.dropout_p)

        # self.dropout
        self.encode_weight = glorot([self.input_dim, self.output_dim])
        self.encode_bias = nn.Parameter(torch.zeros(self.output_dim))


    def forward(self, feature, root, rootindex, support, mask):
        feature = self.dropout(feature)
        x1 = copy.copy(root)
        # encode inputs
        # print('feature_shape',feature.shape)
        # print('weight_shape',self.encode_weight.shape)
        #xw
        encoded_feature = torch.matmul(feature, self.encode_weight) + self.encode_bias
        # print('mask_shape',mask.shape)
        # print('encode_shape',encoded_feature.shape)

        output = mask * self.act(encoded_feature)
        # convolve
        #for _ in range(self.gru_step):

        #output = self.gru_unit(support, output, mask)

        # print(root)
        #
        # root_extend = th.zeros(output.shape[0], output.shape[1], root.shape[2]).to(device)
        #
        # for num_batch in range(output.shape[0]):
        #     tmp_data = th.zeros(output.shape[1], root.shape[2])
        #     for tmp_idx in range(len(tmp_data.shape[0])):
        #         tmp_data[tmp_idx] = x1[num_batch]
        #     root_extend[num_batch] = tmp_data
        #
        # output = th.cat((output, root_extend), 2)
        # output = self.gru_unit(support, output, mask)
        # convolve
        for _ in range(self.gru_step):

            output = self.gru_unit(support, output, mask)
        return output

        return output

class ReadoutLayer(nn.Module):
    """Graph Readout Layer."""
    def __init__(self,
                 input_dim,
                 # output_dim,
                 act=nn.ReLU(),
                 dropout_p=0.):
        super(ReadoutLayer, self).__init__()

        self.input_dim = input_dim
        # self.output_dim = output_dim
        self.act = act
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(1-self.dropout_p)
        self.att_weight = glorot([self.input_dim, 1])
        self.emb_weight = glorot([self.input_dim, self.input_dim])
        # self.mlp_weight = glorot([self.input_dim, self.output_dim])
        self.att_bias = nn.Parameter(torch.zeros(1))
        self.emb_bias = nn.Parameter(torch.zeros(self.input_dim))
        # self.mlp_bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self,x,root,rootindex,_,mask):  # _ not used
        # soft attention
        att = torch.sigmoid(torch.matmul(x, self.att_weight)+self.att_bias)
        emb = self.act(torch.matmul(x, self.emb_weight)+self.emb_bias)

        N = torch.sum(mask, dim=1)

        M = (mask - 1) * 1e9

        # graph summation
        #有数据的地方才有值
        g = mask * att * emb

        g = torch.sum(g, dim=1)/N+ torch.max(g+M,dim=1)[0]
        g = self.dropout(g)
        # classification
        # output = torch.matmul(g,self.mlp_weight)+self.mlp_bias
        # return output
        return g