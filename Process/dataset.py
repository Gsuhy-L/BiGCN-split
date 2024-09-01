import os
import numpy as np
import torch
import random
import scipy.sparse as sp
from torch.utils.data import Dataset
from torch_geometric.data import Data

class GraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.droprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))

def collate_fn(data):
    return data


class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        # print()
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        # print(len(self.fold_x))
        # print(e)
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        # print(os.path.join(self.data_path, id + ".npz"))
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        # try:
        #     max_num = new_edgeindex[0][0]
        #     for i1 in new_edgeindex:
        #         temp = max(i1)
        #         if temp>max_num:
        #             max_num = temp
        #
        #     weight = [1 for i in range(len(new_edgeindex[0]))]
        #
        #
        #     new_edgeindex = sp.csr_matrix((weight, (new_edgeindex[0], new_edgeindex[1])), shape= (max_num+1,max_num+1))
        #     new_edgeindex = new_edgeindex.toarray()
        # except:
        #     pass

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])

        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        # try:
        #     max_num = bunew_edgeindex[0][0]
        #     for i1 in bunew_edgeindex:
        #         temp = max(i1)
        #         if temp>max_num:
        #             max_num = temp
        #
        #     weight = [1 for i in range(len(bunew_edgeindex[0]))]
        #1A
        #     bunew_edgeindex = sp.csr_matrix((weight, (bunew_edgeindex[0], bunew_edgeindex[1])), shape = (max_num+1,max_num+1))
        #     bunew_edgeindex = bunew_edgeindex.toarray()
        # except:
        #     pass
        # print(id)
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))
        # return Data(x=torch.tensor(data['x'],dtype=torch.float32),
        #             edge_index=torch.FloatTensor(new_edgeindex),BU_edge_index=torch.FloatTensor(bunew_edgeindex),
        #      y=torch.FloatTensor([int(data['y'])]), root=torch.FloatTensor(data['root']),
        #      rootindex=torch.FloatTensor([int(data['rootindex'])]))

def get_Node_num(data_path):
    data_path=data_path+'/'
    file_node_len = {}
    bigcn_files = sorted([data_path + f for f in os.listdir(data_path)],
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))  # 用idx.pkl中的idx排序
    for bigcn_file in bigcn_files:
        bigcn_file_data = np.load(os.path.join(bigcn_file), allow_pickle=True)
        id = bigcn_file.split('/')[-1].split('.')[0]
        #KeyError: 'x is not a file in the archive'
        file_node_len[id] = len(bigcn_file_data['x'])
    # print('len',len(file_node_len))
    return file_node_len


class BisplitGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph'),split_num = 0):
        # print()

        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate
        self.file_node_num = get_Node_num(data_path)
        # print(da)
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))

        # self.fold_x = list(filter(lambda id: id in treeDic and self.file_node_num[id] >= lower and self.file_node_num[id] <= upper, self.fold_x_0))


    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        # print(len(self.fold_x))
        # print(e)
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".txt.npz"), allow_pickle=True)
        # print(os.path.join(self.data_path, id + ".npz"))
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        # try:
        #     max_num = new_edgeindex[0][0]
        #     for i1 in new_edgeindex:
        #         temp = max(i1)
        #         if temp>max_num:
        #             max_num = temp
        #
        #     weight = [1 for i in range(len(new_edgeindex[0]))]
        #
        #
        #     new_edgeindex = sp.csr_matrix((weight, (new_edgeindex[0], new_edgeindex[1])), shape= (max_num+1,max_num+1))
        #     new_edgeindex = new_edgeindex.toarray()
        # except:
        #     pass

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])

        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        # try:
        #     max_num = bunew_edgeindex[0][0]
        #     for i1 in bunew_edgeindex:
        #         temp = max(i1)
        #         if temp>max_num:
        #             max_num = temp
        #
        #     weight = [1 for i in range(len(bunew_edgeindex[0]))]
        #
        #     bunew_edgeindex = sp.csr_matrix((weight, (bunew_edgeindex[0], bunew_edgeindex[1])), shape = (max_num+1,max_num+1))
        #     bunew_edgeindex = bunew_edgeindex.toarray()
        # except:
        #     pass
        # print(id)
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]),
             rootindex=torch.LongTensor([int(data['rootindex'])]))
        # return Data(x=torch.tensor(data['x'],dtype=torch.float32),
        #             edge_index=torch.FloatTensor(new_edgeindex),BU_edge_index=torch.FloatTensor(bunew_edgeindex),
        #      y=torch.FloatTensor([int(data['y'])]), root=torch.FloatTensor(data['root']),
        #      rootindex=torch.FloatTensor([int(data['rootindex'])]))
class BisplitFTGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph'),split_num = 0):
        # print()

        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate
        self.file_node_num = get_Node_num(data_path)
        # print(da)
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        # print(list(filter(lambda id: id in treeDic and len(treeDic[id]) < lower or len(treeDic[id]) > upper, fold_x)))

        # self.fold_x = list(filter(lambda id: id in treeDic and self.file_node_num[id] >= lower and self.file_node_num[id] <= upper, self.fold_x_0))
        if "Pheme" in data_path:
            self.ego_files_name = self.get_ego_name()
            self.fold_x = list(filter(lambda id: id in self.ego_files_name, self.fold_x))
            # print(self.fold_x)
            # if "553540824768991233" in self.fold_x:
            #
            #     self.fold_x.remove("553540824768991233")
            # if "580352540316946432" in self.fold_x:
            #     self.fold_x.remove("580352540316946432")
            # if "525068253341970432" in self.fold_x:
            #     self.fold_x.remove("525068253341970432")
            # if "498300832648273920" in self.fold_x:
            #     self.fold_x.remove("498300832648273920")

            # self.fold_x.remove("553540824768991233")
            # print(len(fold_x))

    def get_ego_name(self):
        ego_files_name = []
        ego_files = os.listdir("/home/ubuntu/PyProjects_gsuhyl/PyProjects/RDMSC-main_1/data/Pheme_Ego/")
        for file in ego_files:
            ego_files_name.append(file.split(".")[0])
        return ego_files_name

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):

        id =self.fold_x[index]
        if "Pheme" in self.data_path:
            if ("count_split" or "data_split")  in self.data_path:
                data=np.load(os.path.join(self.data_path, id + ".txt.npz"), allow_pickle=True)
            else:
                data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        else:
            if ("count_split" or "data_split")  in self.data_path:
                data=np.load(os.path.join(self.data_path, id + ".txt.npz"), allow_pickle=True)
            else:
                data=np.load(os.path.join(self.data_path, id + ".txt.npz"), allow_pickle=True)
        # print(os.path.join(self.data_path, id + ".npz"))
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        # try:
        #     max_num = new_edgeindex[0][0]
        #     for i1 in new_edgeindex:
        #         temp = max(i1)
        #         if temp>max_num:
        #             max_num = temp
        #
        #     weight = [1 for i in range(len(new_edgeindex[0]))]
        #
        #
        #     new_edgeindex = sp.csr_matrix((weight, (new_edgeindex[0], new_edgeindex[1])), shape= (max_num+1,max_num+1))
        #     new_edgeindex = new_edgeindex.toarray()
        # except:
        #     pass

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])

        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        # try:
        #     max_num = bunew_edgeindex[0][0]
        #     for i1 in bunew_edgeindex:
        #         temp = max(i1)
        #         if temp>max_num:
        #             max_num = temp
        #
        #     weight = [1 for i in range(len(bunew_edgeindex[0]))]
        #
        #     bunew_edgeindex = sp.csr_matrix((weight, (bunew_edgeindex[0], bunew_edgeindex[1])), shape = (max_num+1,max_num+1))
        #     bunew_edgeindex = bunew_edgeindex.toarray()
        # except:
        #     pass
        # print(id)
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]),
             rootindex=torch.LongTensor([int(data['rootindex'])]))
        # return Data(x=torch.tensor(data['x'],dtype=torch.float32),
        #             edge_index=torch.FloatTensor(new_edgeindex),BU_edge_index=torch.FloatTensor(bunew_edgeindex),
        #      y=torch.FloatTensor([int(data['y'])]), root=torch.FloatTensor(data['root']),
        #      rootindex=torch.FloatTensor([int(data['rootindex'])]))


class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..','data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]

        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))
