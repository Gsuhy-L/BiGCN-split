import os
import numpy as np
import torch
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import random
import re
from tqdm import tqdm
import torch
import logging

def get_dataset_data(fold_x,treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,data_path = os.path.join('..', '..', 'data', 'Twitter15graph')):

    fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
    x = []
    edge_index = []
    BU_edge_index = []
    y = []
    root = []
    rootindex = []

    for i in range(len(fold_x)):

        id =fold_x[i]

        data=np.load(os.path.join(data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        data_x = data['x']

        if tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        max_num = data_x.shape[0]
        # max_num = 0
        # try:
        #     for i1 in new_edgeindex:
        #         temp = max(i1)
        #         if temp>max_num:
        #             max_num = temp
        # except:
        #     pass

        weight = [1 for i in range(len(new_edgeindex[0]))]


        new_edgeindex = sp.csr_matrix((weight, (new_edgeindex[0], new_edgeindex[1])), shape= (max_num,max_num))

        new_edgeindex = new_edgeindex.toarray()



        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])

        if budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]


        #max_num = 0
        max_num = data_x.shape[0]
        # try:
        #     for i1 in bunew_edgeindex:
        #         temp = max(i1)
        #         if temp>max_num:
        #             max_num = temp
        # except:
        #     pass


        weight = [1 for i in range(len(bunew_edgeindex[0]))]




        bunew_edgeindex = sp.csr_matrix((weight, (bunew_edgeindex[0], bunew_edgeindex[1])), shape = (max_num,max_num))
        bunew_edgeindex = bunew_edgeindex.toarray()




        # tem_x=torch.tensor(data['x'],dtype=torch.float32)
        # tem_edge_index=torch.LongTensor(new_edgeindex)
        # tem_BU_edge_index=torch.LongTensor(bunew_edgeindex)
        # tem_y=torch.LongTensor([int(data['y'])])
        # tem_root=torch.LongTensor(data['root'])
        # tem_rootindex=torch.LongTensor([int(data['rootindex'])])

        tem_x=data['x']
        tem_edge_index=new_edgeindex

        tem_BU_edge_index=bunew_edgeindex
        tem_y=[int(data['y'])]
        tem_root=data['root']
        tem_rootindex=[int(data['rootindex'])]

        x.append(tem_x)
        edge_index.append(tem_edge_index)
        x_shape = tem_x.shape[0]
        edge_shape = tem_edge_index.shape[0]
        # print('x_shape',x_shape)
        # print('edge_shape',edge_shape)

        BU_edge_index.append(tem_BU_edge_index)
        y.append(tem_y)
        root.append(tem_root)
        rootindex.append(tem_rootindex)

    graph_data = {}

    graph_data['x'] = np.array(x)


    graph_data['edge_index'] = np.array(edge_index)
    graph_data['Bu_edge_index'] = np.array(BU_edge_index)
    graph_data['y'] = np.array(y)
    graph_data['root'] = np.array(root)
    graph_data['rootindex'] = np.array(rootindex)


    return graph_data
def texting_data():
    # return Data(x=torch.tensor(data['x'],dtype=torch.float32),
    #             edge_index=torch.FloatTensor(new_edgeindex),BU_edge_index=torch.FloatTensor(bunew_edgeindex),
    #      y=torch.FloatTensor([int(data['y'])]), root=torch.FloatTensor(data['root']),
    #      rootindex=torch.FloatTensor([int(data['rootindex'])]))

    names = ['x_adj', 'x_embed', 'y', 'tx_adj', 'tx_embed', 'ty', 'allx_adj', 'allx_embed', 'ally']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x_adj, x_embed, y, tx_adj, tx_embed, ty, allx_adj, allx_embed, ally = tuple(objects)
    # train_idx_ori = parse_index_file("data/{}.train.index".format(dataset_str))
    # train_size = len(train_idx_ori)

    train_adj = []
    train_embed = []
    val_adj = []
    val_embed = []
    test_adj = []
    test_embed = []

    for i in range(len(y)):
        # print(x_adj[i])
        # print('---------------------------------------')
        adj = x_adj[i].toarray()

        embed = np.array(x_embed[i])
        train_adj.append(adj)
        train_embed.append(embed)

    for i in range(len(y), len(ally)):  # train_size):
        adj = allx_adj[i].toarray()
        embed = np.array(allx_embed[i])
        val_adj.append(adj)
        val_embed.append(embed)

    for i in range(len(ty)):
        adj = tx_adj[i].toarray()
        embed = np.array(tx_embed[i])
        test_adj.append(adj)
        test_embed.append(embed)

    train_adj = np.array(train_adj)
    val_adj = np.array(val_adj)
    test_adj = np.array(test_adj)
    train_embed = np.array(train_embed)
    val_embed = np.array(val_embed)
    test_embed = np.array(test_embed)
    train_y = np.array(y)
    val_y = np.array(ally[len(y):len(ally)])  # train_size])
    test_y = np.array(ty)

    return train_adj, train_embed, train_y, val_adj, val_embed, val_y, test_adj, test_embed, test_y