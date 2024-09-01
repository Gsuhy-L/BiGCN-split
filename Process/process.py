import os
from Process.dataset import GraphDataset,BiGraphDataset,UdGraphDataset,BisplitFTGraphDataset
cwd=os.getcwd()
from Process.new_utils import get_dataset_data


################################### load tree#####################################
def loadTree(dataname):
    if 'Twitter' in dataname:
        treePath = os.path.join(cwd,'data/'+dataname+'/data.TD_RvNN.vol_5000.txt')
        print("reading twitter tree")
        treeDic = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
        print('tree no:', len(treeDic))

    if dataname == "Weibo":
        treePath = os.path.join(cwd,'data/Weibo/weibotree.txt')
        print("reading Weibo tree")
        treeDic = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC,Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]),line.split('\t')[3]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
        print('tree no:', len(treeDic))
    if dataname == "Pheme":
        treePath = os.path.join(cwd, 'data/Pheme/data.TD.RvNN.vol_5000.txt')
        print("reading Weibo tree")
        treeDic = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC, Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]), \
                                       line.split('\t')[-1]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
        print('tree no:', len(treeDic))
    return treeDic

################################# load data ###################################
def loadData(dataname, treeDic,fold_x_train,fold_x_test,droprate):
    data_path=os.path.join(cwd, 'data', dataname+'graph')
    print("loading train set", )
    traindata_list = GraphDataset(fold_x_train, treeDic, droprate=droprate,data_path= data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = GraphDataset(fold_x_test, treeDic,data_path= data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadUdData(dataname, treeDic,fold_x_train,fold_x_test,droprate):
    data_path=os.path.join(cwd, 'data',dataname+'graph')
    print("loading train set", )
    traindata_list = UdGraphDataset(fold_x_train, treeDic, droprate=droprate,data_path= data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = UdGraphDataset(fold_x_test, treeDic,data_path= data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadBiData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate,BUdroprate):
    data_path = os.path.join(cwd,'data', dataname + 'graph')
    print("loading train set", )
    traindata_list = BiGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = BiGraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list


def loadBisplitData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate,BUdroprate,split_num):

    split_path = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/'
    data_path = os.path.join(split_path, dataname+'fulltime')
    split_data_path = os.path.join(split_path,'data_split', dataname ,split_num)
    print("loading train set", )
    traindata_list = BiGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = BiGraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadBisplitFTData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate,BUdroprate,split_num):
    split_path = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-master/Process/'
    data_path = os.path.join(split_path, 'data/'+dataname+'fulltime/')
    # split_data_path = os.path.join(split_path,'data_split', dataname ,split_num)
    split_data_path = os.path.join(split_path,'count_split', dataname ,split_num)

    print("loading train set", )
    traindata_list = BisplitFTGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = BisplitFTGraphDataset(fold_x_test, treeDic, data_path=split_data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadBiData_new(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate,BUdroprate):
    data_path = os.path.join(cwd,'data', dataname + 'graph')
    print("loading train set", )
    traindata_list = get_dataset_data(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)

    print("train no:", len(traindata_list['y']))
    print("loading test set", )
    testdata_list = get_dataset_data(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list['y']))
    return traindata_list, testdata_list

