import sys,os
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
import copy
from Process.ggnn_model_1.models import GNN
from Process.ggnn_model_1.utils import preprocess_adj,preprocess_features
import torch.nn as nn
from Process.ggnn_model_1.inits import glorot,xavier

class TDrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1=copy.copy(x.float())

        x = self.conv1(x, edge_index)
        x2=copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)
        x= scatter_mean(x, data.batch, dim=0)

        return x

class BUrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index

        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)

        x2 = copy.copy(x)

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x= scatter_mean(x, data.batch, dim=0)
        return x

class Net(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.fc=th.nn.Linear((out_feats+hid_feats)*2,4)

    def forward(self, data):

        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = th.cat((BU_x,TD_x), 1)
        x=self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

class Ggnn_Net(th.nn.Module):
    def __init__(self,input_dim, output_dim, hidden_dim, gru_step, dropout_p):
        super(Ggnn_Net, self).__init__()

        #def __init__(self, input_dim, output_dim, hidden_dim, gru_step, dropout_p):
        '''model_func(args=args,
                   input_dim=args.input_dim,
                   output_dim=train_y.shape[1],
                   hidden_dim=args.hidden_dim,
                   gru_step = args.steps,
                   dropout_p=args.dropout)'''

        self.TdGGNN = GNN(input_dim, output_dim, hidden_dim, gru_step, dropout_p)
        self.BuGGNN = GNN(input_dim, output_dim, hidden_dim, gru_step, dropout_p)
        self.mlp_weight1 = glorot([hidden_dim*2, output_dim])
        self.mlp_bias1 = nn.Parameter(th.zeros(output_dim))
        # self.mlp_weight2 = glorot([hidden_dim, output_dim])
        # self.mlp_bias2 = nn.Parameter(th.zeros(output_dim))

        #self.fc=th.nn.Linear( output_dim ,4)

#def forward(self, feature, adj):
    def forward(self, x, root, rootindex, Tdadj, Tdmask, Buadj, Bumask):

        # print(rootindex)

        x1, _  = self.TdGGNN(x, root, rootindex, Tdadj, Tdmask)
        x2, _  = self.BuGGNN(x, root, rootindex, Buadj, Bumask)
        output = th.cat((x1,x2), dim=1)

        output = th.matmul(output,self.mlp_weight1)+self.mlp_bias1
        #output = th.matmul(output,self.mlp_weight2)+self.mlp_bias2


        #x=self.fc(x)

        x = F.log_softmax(output, dim=1)

        return x, _

def train_GCN(treeDic, x_test, x_train,TDdroprate,BUdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter):
    #model = Net(5000,64,64).to(device)
    '''model_func(
                   input_dim=args.input_dim,
                   output_dim=train_y.shape[1],
                   hidden_dim=args.hidden_dim,
                   gru_step = args.steps,
                   dropout_p=args.dropout)'''
    model = Ggnn_Net(5000,4,96,2,0.5).to(device)

    # BU_params=list(map(id,model..conv1.parameters()))
    # BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    # base_params=filter(lambda p:id(p) not in BU_params,model.parameters())
    # optimizer = th.optim.Adam([
    #     {'params':base_params},
    #     {'params':model.BUrumorGCN.conv1.parameters(),'lr':lr/5},
    #     {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr/5}
    # ], lr=lr, weight_decay=weight_decay)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    # print('x_train',len(x_train))
    for epoch in range(n_epochs):
        #traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate,BUdroprate)
        traindata_list, testdata_list = loadBiData_new(dataname, treeDic, x_train, x_test, TDdroprate,BUdroprate)

        # print('--------------------')
        # print(traindata_list['edge_index'])
        # print('----------------')



        # train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=0)
        # test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=0)
        avg_loss = []
        avg_acc = []
        batch_idx = 0

        #tqdm_train_loader = tqdm(train_loader)
        #for Batch_data in tqdm_train_loader:
        indices = np.arange(0, len(traindata_list['y']))


        for start in range(0, len(traindata_list['y']), batchsize):


            end = start + batchsize
            idx = indices[start:end]

            batch_train_data_list = {}
            batch_train_data_list['edge_index'] = traindata_list['edge_index'][idx]
            batch_train_data_list['Bu_edge_index'] = traindata_list['Bu_edge_index'][idx]
            batch_train_data_list['x'] = traindata_list['x'][idx]
            batch_train_data_list['y'] = traindata_list['y'][idx]
            batch_train_data_list['root'] = traindata_list['root'][idx]
            batch_train_data_list['rootindex'] = traindata_list['rootindex'][idx]
            '''    graph_data['x'] = np.array(x)

    graph_data['edge_index'] = np.array(edge_index)
    graph_data['Bu_edge_index'] = np.array(BU_edge_index)
    graph_data['y'] = np.array(y)
    graph_data['root'] = np.array(root)
    graph_data['rootindex'] = np.array(rootindex)'''



            tem_train_feature = preprocess_features(batch_train_data_list['x'])
            tem_train_Tdadj, tem_train_Tdmask = preprocess_adj(batch_train_data_list['edge_index'])
            tem_train_Buadj, tem_train_Bumask = preprocess_adj(batch_train_data_list['Bu_edge_index'])


            batch_train_data_list['x'] = th.tensor(tem_train_feature).to(device).float()
            batch_train_data_list['edge_index'] = th.tensor(tem_train_Tdadj).to(device).float()
            batch_train_data_list['Tdmask'] = th.tensor(tem_train_Tdmask).to(device).float()
            batch_train_data_list['Bu_edge_index'] = th.tensor(tem_train_Buadj).to(device).float()
            batch_train_data_list['Bumask'] = th.tensor(tem_train_Bumask).to(device).float()
            batch_train_data_list['root'] = th.tensor(batch_train_data_list['root']).to(device).float()
            batch_train_data_list['rootindex'] = th.tensor(batch_train_data_list['rootindex']).to(device).float()

            batch_train_data_list['y'] = th.tensor(batch_train_data_list['y']).to(device)
            batch_train_data_list['y'] = batch_train_data_list['y'].flatten(0)

            outputs, _ = model(batch_train_data_list['x'], batch_train_data_list['root'],batch_train_data_list['rootindex'], batch_train_data_list['edge_index'], batch_train_data_list['Tdmask'],batch_train_data_list['Bu_edge_index'], batch_train_data_list['Bumask'])  # embeddings not used

            #loss = softmax_cross_entropy(loss_fn, outputs, train_y[idx])

            #Batch_data.to(device)
            #out_labels= model(Batch_data)

            finalloss=F.nll_loss(outputs,batch_train_data_list['y'])
            loss=finalloss
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = outputs.max(dim=-1)
            correct = pred.eq(batch_train_data_list['y']).sum().item()
            train_acc = correct / len(batch_train_data_list['y'])
            avg_acc.append(train_acc)

            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            batch_idx = batch_idx + 1


        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        #tqdm_test_loader = tqdm(test_loader)
        #for Batch_data in tqdm_test_loader:
        #    Batch_data.to(device)
        #    val_out = model(Batch_data)

        indices = np.arange(0, len(testdata_list['y']))
        for start in range(0, len(testdata_list), batchsize):
            end = start + batchsize
            idx = indices[start:end]
            batch_test_data_list={}
            batch_test_data_list['edge_index'] = testdata_list['edge_index'][idx]
            batch_test_data_list['Bu_edge_index'] = testdata_list['Bu_edge_index'][idx]
            batch_test_data_list['x'] = testdata_list['x'][idx]
            batch_test_data_list['y'] = testdata_list['y'][idx]
            batch_test_data_list['root'] = testdata_list['root'][idx]
            batch_test_data_list['rootindex'] = testdata_list['rootindex'][idx]
            '''    graph_data['x'] = np.array(x)

    graph_data['edge_index'] = np.array(edge_index)
    graph_data['Bu_edge_index'] = np.array(BU_edge_index)
    graph_data['y'] = np.array(y)
    graph_data['root'] = np.array(root)
    graph_data['rootindex'] = np.array(rootindex)'''

            tem_test_feature = preprocess_features(batch_test_data_list['x'])
            tem_test_Tdadj, tem_test_Tdmask = preprocess_adj(batch_test_data_list['edge_index'])
            tem_test_Buadj, tem_test_Bumask = preprocess_adj(batch_test_data_list['Bu_edge_index'])


            batch_test_data_list['x'] = th.tensor(tem_test_feature).to(device).float()
            batch_test_data_list['root'] = th.tensor(batch_test_data_list['root']).to(device).float()
            batch_test_data_list['rootindex'] = th.tensor(batch_test_data_list['rootindex']).to(device).float()
            batch_test_data_list['edge_index'] = th.tensor(tem_test_Tdadj).to(device).float()
            batch_test_data_list['Tdmask'] = th.tensor(tem_test_Tdmask).to(device).float()
            batch_test_data_list['Bu_edge_index'] = th.tensor(tem_test_Buadj).to(device).float()
            batch_test_data_list['Bumask'] = th.tensor(tem_test_Bumask).to(device).float()


            batch_test_data_list['y'] = th.tensor(batch_test_data_list['y']).to(device)
            batch_test_data_list['y'] = batch_test_data_list['y'].flatten(0)

            val_outputs, _ = model(batch_test_data_list['x'], batch_test_data_list['root'], batch_test_data_list['rootindex'], batch_test_data_list['edge_index'], batch_test_data_list['Tdmask'],batch_test_data_list['Bu_edge_index'], batch_test_data_list['Bumask'])  # embeddings not used

            val_loss  = F.nll_loss(val_outputs, batch_test_data_list['y'])
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_outputs.max(dim=1)
            correct = val_pred.eq(batch_test_data_list['y']).sum().item()
            val_acc = correct / len(batch_test_data_list['y'])
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, batch_test_data_list['y'])
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'BiGCN', dataname)
        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return train_losses , val_losses ,train_accs, val_accs,accs,F1,F2,F3,F4
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    print("seed:", seed)
lr=0.0005
weight_decay=1e-4
patience=10
n_epochs=200
batchsize=1
TDdroprate=0.2
BUdroprate=0.2
datasetname=sys.argv[1] #"Twitter15"、"Twitter16"
iterations=int(sys.argv[2])
model="GCN"
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
test_accs = []
NR_F1 = []
FR_F1 = []
TR_F1 = []
UR_F1 = []
for iter in range(iterations):
    fold0_x_test, fold0_x_train, \
    fold1_x_test,  fold1_x_train,  \
    fold2_x_test, fold2_x_train, \
    fold3_x_test, fold3_x_train, \
    fold4_x_test,fold4_x_train = load5foldData(datasetname)
    treeDic=loadTree(datasetname)
    train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(treeDic,
                                                                                               fold0_x_test,
                                                                                               fold0_x_train,
                                                                                               TDdroprate,BUdroprate,
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(treeDic,
                                                                                               fold1_x_test,
                                                                                               fold1_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(treeDic,
                                                                                               fold2_x_test,
                                                                                               fold2_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(treeDic,
                                                                                               fold3_x_test,
                                                                                               fold3_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(treeDic,
                                                                                               fold4_x_test,
                                                                                               fold4_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)
    NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
    FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
    UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
    sum(test_accs) / iterations, sum(NR_F1) /iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations))


