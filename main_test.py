# -*- coding: utf-8 -*-

import argparse
import glob
import os
import time
import torch
import torch.nn.functional as F
from models import ClassificationModel
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from sklearn.metrics import auc, precision_recall_curve, roc_curve, f1_score, roc_auc_score
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import random
from torch_geometric.utils import dense_to_sparse,to_dense_batch, to_dense_adj, to_networkx
from torch_geometric.data import Data
import networkx as nx
import json
import shutil
from utils import get_dataset
from sklearn.preprocessing import normalize, StandardScaler
parser = argparse.ArgumentParser()


parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--nhid', type=int, default=256, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--dataset', type=str, default=None, help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('--pooling_ratio', type=float, default=0.25, help='pooling ratio')
parser.add_argument('--thredhold', type=float, default=0.85, help='')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic=True
	

def train(model, train_loader, val_loader, test_loader, sample_set, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    
    min_loss = 1e10
    max_acc = 0
    patience_cnt = 0
    val_prc_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    thredhold_list = []

    for epoch in range(args.epochs):
	    model.train()
	    loss_train = 0.0
	    acc_train = 0.0
	    loss_train_MI = 0.0
	    thred_epoch = 0
	    for i, data in enumerate(train_loader):
	        data = data.to(args.device)


	        
	        optimizer.zero_grad()
	        loss_, pred, logits, acc_tr = model(data)
	        loss = loss_ 
	        loss.backward()
	        optimizer.step()
	        acc_train += acc_tr.item()
	        loss_train += loss.item() * num_graphs(data)
	    scheduler.step()
	    loss_val, f1_val, roc_val, _, acc_val = test(model, val_loader, sample_set, args)        
	    print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(loss_train/len(train_loader.dataset)), 'acc_train:{:.4f}'.format(acc_train/len(train_loader.dataset)), 'loss_val: {:.4f}'.format(loss_val),'acc_val: {:.4f}'.format(acc_val),'time: {:.4f}s'.format(time.time() - t))
	    torch.save(model.state_dict(), '{}.pth'.format(epoch))
	    if acc_val >= max_acc:
	        max_acc = acc_val
	        best_epoch = epoch
	        patience_cnt = 0
	    else:
	        patience_cnt += 1

	    if patience_cnt == args.patience and epoch > 150:
	        break

	    files = glob.glob('*.pth')
	    for f in files:
	        epoch_nb = int(f.split('.')[0])
	        if epoch_nb < best_epoch:
	            os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))
    return best_epoch

	    
	    
def test(model, test_loader, sample_set, args):
    label = []
    pred_label = []
    pred_logits = []
    model.eval()
    loss_test = 0.0
    acc = 0.0
    with torch.no_grad():  
        for i, data in enumerate(test_loader):
            data = data.to(args.device)
            loss, pred, pred_logit, _ = model(data, sample_set)
            loss_test += loss * num_graphs(data)
            if len(label) != 0:
                label = np.concatenate((label,data.y[:,0].cpu().detach().numpy()),axis=0)
                pred_label = np.concatenate((pred_label,pred.cpu().detach().numpy()),axis=0)
                pred_logits = np.concatenate((pred_logits,pred_logit.cpu().detach().numpy()), axis=0)
            else:
                label = data.y[:,0].cpu().detach().numpy()
                pred_label = pred.cpu().detach().numpy()
                pred_logits = pred_logit.cpu().detach().numpy()
  
        f1 = []        
        f1.append(f1_score(label, pred_label, average='weighted'))
        f1.append(0)
        f1.append(f1_score(label, pred_label, average='micro'))
        f1.append(f1_score(label, pred_label, average='macro'))

        

        if pred_logits.shape[1] > 2:
            auroc = roc_auc_score(label, pred_logits, multi_class='ovo')
        else:
            auroc = roc_auc_score(label, pred_logits[:,1])
        
        if args.num_classes > 2:
            auprc = 0
        else:
            precision_ab, recall_ab, _ = precision_recall_curve(label, pred_logits[:,1])
            auprc = auc(recall_ab, precision_ab)
        acc = sum(np.equal(label, pred_label))
        torch.cuda.empty_cache()
    return loss_test.item() / len(test_loader.dataset) , f1, auroc, auprc, acc/len(label)  


def dataprepross(args, dataset):
    feature_view2 = []
    feature_graphstruc = []

    for i, data in enumerate(dataset):
        x, edge_index, label = data.x, data.edge_index, data.y  
         
        data_ = Data(data=x, edge_index=edge_index, num_nodes=x.shape[0])
        data_nx = to_networkx(data_, to_undirected=True)          
        degree = torch.tensor(list(dict(data_nx.degree()).values()),device=args.device).reshape(-1,1)
        num_triangle = torch.tensor(list(nx.triangles(data_nx).values()),device=args.device).reshape(-1,1)
        num_core = torch.tensor(list(nx.core_number(data_nx).values()),device=args.device).reshape(-1,1)
        size_clique = torch.tensor(list(nx.node_clique_number(data_nx).values()),device=args.device).reshape(-1,1)
        num_clique = torch.tensor(list(nx.number_of_cliques(data_nx).values()),device=args.device).reshape(-1,1)
        cluster = torch.tensor(list(nx.clustering(data_nx).values()),device=args.device).reshape(-1,1)
        square = torch.tensor(list(nx.square_clustering(data_nx).values()),device=args.device).reshape(-1,1)
         
        
        feature_struc = torch.cat((degree, num_triangle), dim=1)
        feature_struc_dem1 = torch.cat((num_core, size_clique), dim=1)      
        feature_struc_dem2 = torch.cat((num_clique, cluster), dim=1)    
        feature_struc = torch.cat((feature_struc, feature_struc_dem1), dim=1)
        feature_struc = torch.cat((feature_struc, feature_struc_dem2), dim=1)
        feature_struc = torch.cat((feature_struc, square), dim=1)                
        feature_view2.append(feature_struc.reshape(1,-1).tolist())
        triangle = (sum(num_triangle) / 3).reshape(1,1)
        clique_num = torch.tensor(nx.graph_clique_number(data_nx),device=args.device).reshape(1,1) #The *clique number* of a graph is the size of the largest clique in the graph.
        cluster_g = torch.tensor(nx.average_clustering(data_nx),device=args.device).reshape(1,1)#Compute the average clustering coefficient for the graph G.
        global_eff = torch.tensor(nx.global_efficiency(data_nx),device=args.device).reshape(1,1)#Returns the average global efficiency of the graph.The *efficiency* of a pair of nodes in a graph is the multiplicative 
                                 #inverse of the shortest path distance between the nodes. The *average global efficiency* of a graph is the average
                                 #efficiency of all pairs of nodes
        local_eff = torch.tensor(nx.local_efficiency(data_nx),device=args.device).reshape(1,1)#Returns the average local efficiency of the graph. The *efficiency* of a pair of nodes in a graph is the multiplicative
                            # inverse of the shortest path distance between the nodes. The *local efficiency* of a node in the graph is the average 
                            #global efficiency of the subgraph induced by the neighbors of the node. The *average local efficiency* is the average 
                            #of the local efficiencies of each node.
        bridge = torch.tensor(int(nx.has_bridges(data_nx)),device=args.device).reshape(1,1)#a bridge is an edge whose remove will cause the number of connected components of the graph to increase.
        s_metric = torch.tensor(nx.s_metric(data_nx, normalized=False),device=args.device).reshape(1,1)
        
        graph_str = torch.cat((triangle, clique_num),dim=1)                 
        graph_str_dem1 = torch.cat((cluster_g, global_eff), dim=1)
        graph_str_dem2 = torch.cat((local_eff, bridge), dim=1)
        graph_str = torch.cat((graph_str, graph_str_dem1),dim=1)   
        graph_str = torch.cat((graph_str, graph_str_dem2),dim=1)   
        graph_str = torch.cat((graph_str, s_metric),dim=1)          
    
        feature_graphstruc.append(graph_str.squeeze().tolist())     
    print('Data proprossing done!')
    
    return feature_view2, feature_graphstruc


def k_fold(dataset, folds, args):
    skf = StratifiedKFold(folds, shuffle=True, random_state=args.seed)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), [int(data.y.squeeze()[0].cpu()) for data in dataset]):
            test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

    
if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    for dataname in ['DD', 'BZR','COX2','PROTEINS','NCI1','DHFR','NCI109', 'PROTEINS_full','IMDB-BINARY','MUTAG','REDDIT-BINARY']:#'IMDB-MULTI',,'REDDIT-MULTI-5K'
        args.dataset = dataname
        
        if args.dataset in ["IMDB-BINARY", "REDDIT-BINARY", "COLLAB", "IMDB-MULTI",'REDDIT-MULTI-5K']:
            dataset = get_dataset(args.dataset)
        else:
            dataset = TUDataset('data/', name=args.dataset, use_node_attr=True)
        args.num_classes = dataset.num_classes
        args.num_features = dataset.num_features
        
        filedata = '/home/roma/our/dataprocess/'+args.dataset+'view2.txt'
        filedata2 = '/home/roma/our/dataprocess/'+args.dataset+'view3.txt'
        
        if os.path.exists(filedata):
            view2 = [] 
            with open(filedata, 'r') as f:
                view2 = f.read()              
                view2 = eval(view2) 
            view_graph = []
            with open(filedata2, 'r') as f2:
                view_graph = f2.read()
                view_graph = eval(view_graph)              
              
        else:	
            view2, view_graph = dataprepross(args, dataset)
            
            file_view2 = open(filedata, 'a')
            file_view2.write('{}'.format(view2))
            file_view2.close()
            file_view3 = open(filedata2, 'a')
            file_view3.write('{}'.format(view_graph))
            file_view3.close()   
        
        
        dataset = dataset
        dataset_ = []
        for i, data in enumerate(dataset):
            x = torch.cat((torch.tensor(data.x, device=args.device), torch.tensor(view2[i], device=args.device).reshape(-1,7)), dim=1)
            y = torch.cat((torch.tensor(data.y, device=args.device), torch.tensor(view_graph[i], device=args.device).to(torch.float32)),dim=0).reshape(1,-1)
            dataset_.append(Data(x=x,edge_index=data.edge_index, edge_attr=data.edge_attr,y=y))
        

        graph_label = [int(data.y.squeeze()[0].cpu()) for data in dataset_]
        kfd=StratifiedKFold(n_splits=10, random_state=args.seed, shuffle=True)#######n_splits  

        final_test_acc = []
        final_test_loss = []
        final_test_f1 = []
        final_test_auc = []
        final_test_prc = []
        final_test_acc = []
        best_model_ = []
        for fold_number, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset_, 10, args))):
            graphs_train = [dataset_[int(i)] for i in train_idx]
            graphs_test = [dataset_[int(i)] for i in test_idx] 
            graphs_val = [dataset_[int(i)] for i in val_idx] 

            sample_size = int(min(args.batch_size-1, len(train_idx)))
            sample_data = random.sample(graphs_train, k=sample_size)             


            train_loader = DataLoader(graphs_train, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(graphs_val, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(graphs_test, batch_size=args.batch_size, shuffle=False)            
            sample_set = DataLoader(sample_data, batch_size=sample_size, shuffle=False)        
    
            model = ClassificationModel(args).to(args.device)
            model.reset_parameters()
            path = './'+args.dataset+'/'+str(fold_number)+'fold/'
            os.makedirs(path, exist_ok=True)
            if len(os.listdir(path)) == 0:
                best_model = train(model, train_loader, val_loader, test_loader, sample_set, args)
                shutil.copy(str(best_model)+'.pth', path)
            else:
                for root, dirs, files in os.walk(path):
                    for name in files:
                        name = os.path.splitext(name)[0] 
                        if name.isnumeric():
                            best_model = name

            best_model_.append(best_model)
            
            model.load_state_dict(torch.load(path+'{}.pth'.format(best_model)))
            test_loss, test_f1, test_auc, test_prc, accuracy = test(model, test_loader, sample_set, args)
            print('Test set results, loss = {:.6f}, f1 score ={}, auc = {:.6f}, prc = {:.6f}, accuracy = {:.4f}'.format(test_loss, test_f1, test_auc, test_prc, accuracy))
            
            final_test_loss.append(test_loss)
            final_test_f1.append(test_f1)            
            final_test_auc.append(test_auc)
            final_test_prc.append(test_prc)
            final_test_acc.append(accuracy)
            
        final_test_loss = np.array(final_test_loss)
        final_test_f1 = np.array(final_test_f1)     
        final_test_auc = np.array(final_test_auc)
        final_test_prc = np.array(final_test_prc)
        final_test_acc = np.array(final_test_acc)
        loss_mean = final_test_loss.mean()
        loss_std = final_test_loss.std() 
        f1_1_mean = final_test_f1[:,0].mean()
        f1_1_std = final_test_f1[:,0].std() 
        f1_0_mean = final_test_f1[:,1].mean()
        f1_0_std = final_test_f1[:,1].std() 
        f1_micro_mean = final_test_f1[:,2].mean()
        f1_micro_std = final_test_f1[:,2].std() 
        f1_macro_mean = final_test_f1[:,3].mean()
        f1_macro_std = final_test_f1[:,3].std() 
        auc_mean = final_test_auc.mean()
        auc_std = final_test_auc.std() 
        prc_mean = final_test_prc.mean()
        prc_std = final_test_prc.std() 
        acc_mean = final_test_acc.mean()
        acc_std = final_test_acc.std()
        print('Final test set results, {}, loss: {:.4f}, {:.4f}, f1: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, auc: {:.4f}, {:.4f}, prc: {:.4f}, {:.4f}, accuracy: {:.4f}, {:.4f}'.format(args.dataset,loss_mean,loss_std, f1_1_mean, f1_1_std, f1_0_mean, f1_0_std, f1_micro_mean, f1_micro_std, f1_macro_mean,f1_macro_std,auc_mean,auc_std, prc_mean, prc_std, acc_mean, acc_std))
        file_result = open('./result/'+args.dataset+'.txt', 'a')
        file_result.write('Final test set results, {}, lr: {}, loss: {:.4f}, {:.4f}, f1: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, auc: {:.4f}, {:.4f}, prc: {:.4f}, {:.4f}, accuracy: {:.4f}, {:.4f}\n'.format(args.dataset, args.lr, loss_mean,loss_std, f1_1_mean, f1_1_std, f1_0_mean, f1_0_std, f1_micro_mean, f1_micro_std, f1_macro_mean,f1_macro_std,auc_mean,auc_std, prc_mean, prc_std,  acc_mean, acc_std))
        file_result.close()
