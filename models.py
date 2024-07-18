# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import math
import torch.nn as nn
from torch.nn import init
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap
from torch_geometric.nn import GAE
from torch_geometric.utils import dense_to_sparse,to_dense_batch, to_dense_adj, to_networkx
import random
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from collections import Counter
from torch.nn.parameter import Parameter
from torch_geometric.nn import global_sort_pool



class ClassifierO(torch.nn.Module):
     def __init__(self, args):
        super(ClassifierO, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        
        self.lin1 = torch.nn.Linear(self.nhid * 5, self.nhid*2)
        self.lin2 = torch.nn.Linear(self.nhid*2, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self.num_classes)
        
        #self.lin4 = torch.nn.Linear(7, 7)
        #self.lin5 = torch.nn.Linear(7, 3)
        #self.lin6 = torch.nn.Linear(3, self.num_classes)
        
        self.focalloss = FocalLoss()
        self.reset_parameters()
        
     def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

     def forward(self, x, y):  
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        
        #z = F.relu(self.lin4(z))
        #z = F.dropout(z, p=self.dropout_ratio, training=self.training)
        #z = F.relu(self.lin5(z))
        #z = F.dropout(z, p=self.dropout_ratio, training=self.training)
        #z = self.lin6(z) 
        #z = torch.mul(z, torch.argmax(z, dim=1).reshape(-1,1)) 
            
        #x = x + z
        x_ = F.log_softmax(x, dim=-1)
        
        if self.args.imbalance == True:
            loss = self.focalloss(x_, y.long())
        else:  
            loss = F.nll_loss(x_, y.long())
        pred = x_.max(dim=1)[1]
        
        return loss, pred, x

class Classifier1(torch.nn.Module):
     def __init__(self, args):
        super(Classifier1, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        
        self.lin1 = torch.nn.Linear(self.nhid * 6, self.nhid*2)
        self.lin2 = torch.nn.Linear(self.nhid*2, self.nhid//4)
        self.lin3 = torch.nn.Linear(self.nhid//4, self.num_classes)
        
        #self.lin4 = torch.nn.Linear(7, 7)
        #self.lin5 = torch.nn.Linear(7, 3)
        #self.lin6 = torch.nn.Linear(3, self.num_classes)
        
        self.focalloss = FocalLoss()
        self.reset_parameters()
        
     def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        
     def forward(self, x, y):  
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        
        #z = F.relu(self.lin4(z))
        #z = F.dropout(z, p=self.dropout_ratio, training=self.training)
        #z = F.relu(self.lin5(z))
        #z = F.dropout(z, p=self.dropout_ratio, training=self.training)
        #z = self.lin6(z) 
        #z = torch.mul(z, torch.argmax(z, dim=1).reshape(-1,1)) 
            
        #x = x + z
        x_ = F.log_softmax(x, dim=-1)
        
        if self.args.imbalance == True:
            loss = self.focalloss(x_, y.long())
        else:  
            loss = F.nll_loss(x_, y.long())
        pred = x_.max(dim=1)[1]
        
        return loss, pred, x

class GCN(torch.nn.Module):
    def __init__(self, args, n_features, n_graphfea, n_hid):
        super(GCN, self).__init__()
        self.n_features = n_features
        self.n_hid = n_hid
        self.args = args
        self.dropout = self.args.dropout_ratio
        self.conv1 = GCNConv(self.n_features+n_graphfea, self.n_hid)
        self.conv2 = GCNConv(self.n_hid, self.n_hid)
        self.conv3 = GCNConv(self.n_hid, self.n_hid)
        
        self.conv4 = GCNConv(self.n_features+n_graphfea, self.n_hid)
        self.conv5 = GCNConv(self.n_hid, self.n_hid)
        self.conv6 = GCNConv(self.n_hid, self.n_hid)
        self.relu = nn.ReLU()
        self.reset_parameters()
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.conv5.reset_parameters()
        self.conv6.reset_parameters()
        
    def forward(self, x, view2, edge_index, D_inv):
        #x = F.dropout(x, p=self.dropout, training=self.training)
        #view2 = F.dropout(view2, p=self.dropout, training=self.training)        
        x1_view1, x1_view2 = self.conv1(x, edge_index)
        x2_view1, x2_view2 = self.conv4(view2, edge_index)
        x1 = self.relu(x1_view1 + torch.mul(x2_view2,D_inv))
        x2 = self.relu(x2_view1 + torch.mul(x1_view2,D_inv))

       
        #x1 = F.dropout(x1, p=self.dropout, training=self.training)
        #x2 = F.dropout(x2, p=self.dropout, training=self.training)      
        z1_view1, z1_view2 = self.conv2(x1, edge_index)
        z2_view1, z2_view2 = self.conv5(x2, edge_index)
        z1 = self.relu(z1_view1 + torch.mul(z2_view2,D_inv))
        z2 = self.relu(z2_view1 + torch.mul(z1_view2,D_inv))

        #z1 = F.dropout(z1, p=self.dropout, training=self.training)
        #z2 = F.dropout(z2, p=self.dropout, training=self.training)              
        v1_view1, v1_view2 = self.conv3(z1, edge_index)
        v2_view1, v2_view2 = self.conv6(z2, edge_index)
        v1 = self.relu(v1_view1 + torch.mul(v2_view2,D_inv))
        v2 = self.relu(v2_view1 + torch.mul(v1_view2,D_inv))
        #v1 = F.dropout(v1, p=self.args.dropout_ratio, training=self.training)
        #v2 = F.dropout(v2, p=self.args.dropout_ratio, training=self.training)
       
        q = torch.cat((x1, z1), dim=1)
        q = torch.cat((q, v1), dim=1)
        
       
        p = torch.cat((x2, z2), dim=1)
        p = torch.cat((p, v2), dim=1)        

        return q, p

class GCN2(torch.nn.Module):
    def __init__(self, args, n_features, n_graphfea, n_hid):
        super(GCN2, self).__init__()
        self.n_features = n_features
        self.n_hid = n_hid
        self.args = args
        self.dropout = self.args.dropout_ratio
        self.conv1 = GCNConv(self.n_features+n_graphfea, self.n_hid)
        self.conv2 = GCNConv(self.n_hid, self.n_hid)
        self.conv3 = GCNConv(self.n_hid, self.n_hid)
        
        self.conv4 = GCNConv(self.n_features+n_graphfea, self.n_hid)
        self.conv5 = GCNConv(self.n_hid, self.n_hid)
        self.conv6 = GCNConv(self.n_hid, self.n_hid)
        self.relu = nn.ReLU()
        self.reset_parameters()
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.conv5.reset_parameters()
        self.conv6.reset_parameters()
        
    def forward(self, x, view2, edge_index, D_inv):
        #x = F.dropout(x, p=self.dropout, training=self.training)
        #view2 = F.dropout(view2, p=self.dropout, training=self.training)        
        x1_view1, x1_view2 = self.conv1(x, edge_index)
        x2_view1, x2_view2 = self.conv4(view2, edge_index)
        x1 = self.relu(x1_view1 + torch.mul(x2_view2,D_inv))
        x2 = self.relu(x2_view1 + torch.mul(x1_view2,D_inv))

       
        #x1 = F.dropout(x1, p=self.dropout, training=self.training)
        #x2 = F.dropout(x2, p=self.dropout, training=self.training)      
        z1_view1, z1_view2 = self.conv2(x1, edge_index)
        z2_view1, z2_view2 = self.conv5(x2, edge_index)
        z1 = self.relu(z1_view1 + torch.mul(z2_view2,D_inv))
        z2 = self.relu(z2_view1 + torch.mul(z1_view2,D_inv))

        #z1 = F.dropout(z1, p=self.dropout, training=self.training)
        #z2 = F.dropout(z2, p=self.dropout, training=self.training)              
        v1_view1, v1_view2 = self.conv3(z1, edge_index)
        v2_view1, v2_view2 = self.conv6(z2, edge_index)
        v1 = self.relu(v1_view1 + torch.mul(v2_view2,D_inv))
        v2 = self.relu(v2_view1 + torch.mul(v1_view2,D_inv))
        #v1 = F.dropout(v1, p=self.args.dropout_ratio, training=self.training)
        #v2 = F.dropout(v2, p=self.args.dropout_ratio, training=self.training)
       
        q = torch.cat((x1, z1), dim=1)
        q = torch.cat((q, v1), dim=1)
        
       
        p = torch.cat((x2, z2), dim=1)
        p = torch.cat((p, v2), dim=1)        

        return x1, x2
        #return v1,v2


class GCN3(torch.nn.Module):
    def __init__(self, args, n_features, n_graphfea, n_hid):
        super(GCN3, self).__init__()
        self.n_features = n_features
        self.n_hid = n_hid
        self.args = args
        self.dropout = self.args.dropout_ratio
        self.conv1 = GCNConv(self.n_features, self.n_hid)
        self.conv2 = GCNConv(self.n_hid, self.n_hid)
        self.conv3 = GCNConv(self.n_hid, self.n_hid)
        
        self.conv4 = GCNConv(n_graphfea, self.n_hid)
        self.conv5 = GCNConv(self.n_hid, self.n_hid)
        self.conv6 = GCNConv(self.n_hid, self.n_hid)
        self.relu = nn.ReLU()
        self.reset_parameters()
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.conv5.reset_parameters()
        self.conv6.reset_parameters()
        
    def forward(self, x, view2, edge_index, D_inv):
        #x = F.dropout(x, p=self.dropout, training=self.training)
        #view2 = F.dropout(view2, p=self.dropout, training=self.training)        
        x1_view1, _ = self.conv1(x, edge_index)
        x2_view1, _ = self.conv4(view2, edge_index)
        x1 = self.relu(x1_view1)
        x2 = self.relu(x2_view1)

       
        #x1 = F.dropout(x1, p=self.dropout, training=self.training)
        #x2 = F.dropout(x2, p=self.dropout, training=self.training)      
        z1_view1, z1_view2 = self.conv2(x1, edge_index)
        z2_view1, z2_view2 = self.conv5(x2, edge_index)
        z1 = self.relu(z1_view1)
        z2 = self.relu(z2_view1)

        #z1 = F.dropout(z1, p=self.dropout, training=self.training)
        #z2 = F.dropout(z2, p=self.dropout, training=self.training)              
        v1_view1, v1_view2 = self.conv3(z1, edge_index)
        v2_view1, v2_view2 = self.conv6(z2, edge_index)
        v1 = self.relu(v1_view1)
        v2 = self.relu(v2_view1)
        #v1 = F.dropout(v1, p=self.args.dropout_ratio, training=self.training)
        #v2 = F.dropout(v2, p=self.args.dropout_ratio, training=self.training)
       
        q = torch.cat((x1, z1), dim=1)
        q = torch.cat((q, v1), dim=1)
        
       
        p = torch.cat((x2, z2), dim=1)
        p = torch.cat((p, v2), dim=1)        

        return q, p


class Full_layer(torch.nn.Module):
     def __init__(self, args, n_features, n_graphfea, n_hid):
        super(Full_layer, self).__init__()
        self.args = args
        self.num_features = n_features
        self.n_graphfea = n_graphfea
        self.nhid = n_hid
        
        self.lin1 = torch.nn.Linear(self.nhid * 6, self.nhid*2)
        self.lin2 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin3 = torch.nn.Linear(self.nhid, self.nhid)
        
        
        self.lin4 = torch.nn.Linear(6, self.nhid//2)
        self.lin5 = torch.nn.Linear(self.nhid//2, self.nhid//4)
        self.lin6 = torch.nn.Linear(self.nhid//4, self.nhid//4)
        
        self.reset_parameters()
        
     def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()
        self.lin5.reset_parameters()
        self.lin6.reset_parameters()
        
     def forward(self, node_fea, graph_fea):  
        x1 = self.lin1(node_fea)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x2 = self.lin2(x1)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x3 = self.lin3(x2)
        x3 = F.relu(x3)
        
        x = torch.cat((x1, x2), dim=1)
        x = torch.cat((x, x3), dim=1)
        
        z1 = self.lin4(graph_fea)
        z1 = F.relu(z1)
        z1 = F.dropout(z1, p=0.5, training=self.training)
        z2 = self.lin5(z1)
        z2 = F.relu(z2)
        z2 = F.dropout(z2, p=0.5, training=self.training)
        z3 = self.lin6(z2)
        z3 = F.relu(z3) 
        #z = torch.mul(z, torch.argmax(z, dim=1).reshape(-1,1)) 
        z = torch.cat((z1,z2), dim=1)
        z = torch.cat((z,z3), dim=1)
        
        #fea = x + z
        fea = torch.cat((x,z), dim=1)
        
        return fea




        
class ClassificationModel(torch.nn.Module):
    def __init__(self, args):
        super(ClassificationModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.nlat = args.nlat
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.thred = 0

        self.ClassifierO = ClassifierO(self.args)
        self.Classifier1 = Classifier1(self.args)

        self.relu = nn.ReLU()
        self.GCN = GCN(self.args, self.num_features, 7, self.nhid)
        #self.GCN2 = GCN3(self.args, self.nhid*6, 6, self.nhid)
        self.full_layer = Full_layer(self.args, self.nhid*6, 6, self.nhid)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.GCN.reset_parameters()
        #self.GCN2.reset_parameters()
        self.ClassifierO.reset_parameters()
        self.Classifier1.reset_parameters()
        self.full_layer.reset_parameters()

    def loss_MI(self, z1, z2):
        l = z1.shape[0]
        res = torch.mm(z1, z2.t())
        pos = torch.diag(res)
        pos_JSD = (math.log(2.) - F.softplus(-pos)).sum()
        neg = (F.softplus(-res) + res - math.log(2.)).sum() - (F.softplus(-pos) + pos - math.log(2.)).sum()
        loss = neg/(l*(l-1)) - pos_JSD/l
        return loss	

    def DotProductSimilarity(tensor_1, tensor_2, scale_output=False):
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if scale_output:
            result /= math.sqrt(tensor_1.size(-1))
        return result



    def forward(self, data, training_sample=None, best_thred=None):
        x_, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y  
        
        x = x_[:,:self.args.num_features]
        view2 = x_[:,self.args.num_features:]
        D = (1./(view2[:,0]+1)).reshape(-1,1)
        OneM = torch.ones((x.shape[0],1), device=self.args.device)
        D_inv = torch.where(D>OneM, OneM, D)
        x_temp = torch.zeros((x.shape[0], 7), device=self.args.device)
        view2_temp = torch.zeros((x.shape[0], self.num_features), device=self.args.device)
        x = torch.cat((x,x_temp), dim=1)
        view2 = torch.cat((view2_temp, view2), dim=1)
        
        label = y[:,0]
        feature_graphstruc = y[:,1:]
        _, num = batch.unique(return_counts=True)#nodes num
        num = 1./num
        feature_graphstruc_ = torch.cat((feature_graphstruc[:,0:1]* num.reshape(-1,1),feature_graphstruc[:,1:-1]), dim=1)
        #feature_graphstruc_ = torch.cat((feature_graphstruc_,feature_graphstruc[:,-1].reshape(-1,1)* num.reshape(-1,1)), dim=1)
        feature_graphstruc = feature_graphstruc_
        #feature_graphstruc = self.BN1(feature_graphstruc)
        #x1_view1, x1_view2 = self.conv1(x, edge_index)
        #x2_view1, x2_view2 = self.conv4(view2, edge_index)
        #x1 = self.relu(x1_view1 + torch.mul(x2_view2,D_inv))
        #x2 = self.relu(x2_view1 + torch.mul(x1_view2,D_inv))
        #x1 = F.dropout(x1, p=self.args.dropout_ratio, training=self.training)
        #x2 = F.dropout(x2, p=self.args.dropout_ratio, training=self.training)
        
        
        #z1_view1, z1_view2 = self.conv2(x1, edge_index)
        #z2_view1, z2_view2 = self.conv5(x2, edge_index)
        #z1 = self.relu(z1_view1 + torch.mul(z2_view2,D_inv))
        #z2 = self.relu(z2_view1 + torch.mul(z1_view2,D_inv))
        #z1 = F.dropout(z1, p=self.args.dropout_ratio, training=self.training)
        #z2 = F.dropout(z2, p=self.args.dropout_ratio, training=self.training)
        
        #v1_view1, v1_view2 = self.conv3(z1, edge_index)
        #v2_view1, v2_view2 = self.conv6(z2, edge_index)
        #v1 = self.relu(v1_view1 + torch.mul(v2_view2,D_inv))
        #v2 = self.relu(v2_view1 + torch.mul(v1_view2,D_inv))
        ##v1 = F.dropout(v1, p=self.args.dropout_ratio, training=self.training)
        ##v2 = F.dropout(v2, p=self.args.dropout_ratio, training=self.training)
        
        #q = torch.cat((x1, z1), dim=1)
        #q = torch.cat((q, v1), dim=1)
        #p = torch.cat((x2, z2), dim=1)
        #p = torch.cat((p, v2), dim=1)    
        
        q, p = self.GCN(x, view2, edge_index, D_inv)
        
        q_final =gmp(q, batch)            
        p_final =gmp(p, batch)
        
        loss_MI1 = self.loss_MI(q_final, p_final)
        
        #z_final = torch.maximum(q_final, p_final)
        #z_final = (q_final + p_final)/2
        z_final = torch.cat((q_final, p_final), dim=1)
        loss_1, pred1, logit1 = self.Classifier1(z_final, label)
     #   graph_rep = torch.cat((z_final, feature_graphstruc), dim=1)
        #loss_rep, pred_rep, logit_rep = self.Classifier0(graph_rep, label)
        out_f = self.full_layer(z_final, feature_graphstruc)
        loss, pred, logits = self.ClassifierO(out_f, label)
        logits_f = F.softmax(logits, dim=-1) 
        pred_f = logits_f.max(1)[1]
        acc = pred_f.eq(label).sum()
        return loss, pred_f, logits_f, acc
        #if self.training:
            #print(graph_rep, label)
            #graph_base = torch.cat((gmp(x_,batch), feature_graphstruc), dim=1)
            #sim = self.DotProductSimilarity(graph_base.unsqueeze(1), graph_base.unsqueeze(0))
            #sim = F.cosine_similarity(graph_base.unsqueeze(1), graph_base.unsqueeze(0), dim=-1)
            #sim = F.cosine_similarity(z_final.unsqueeze(1), z_final.unsqueeze(0), dim=-1)
        #    sim = torch.mm(z_final, z_final.t()) #要不要sigmoid
            #sim = torch.mm(graph_rep, graph_rep.t()) 
        #    zero_m = torch.zeros((graph_rep.shape[0], graph_rep.shape[0]), device=self.args.device)
        #    one_m = torch.ones((graph_rep.shape[0], graph_rep.shape[0]), device=self.args.device)
        #    self.thred = torch.quantile(sim, self.args.thredhold)
        #    A = torch.where(sim>self.thred, one_m, zero_m)
        #    A_ = torch.maximum(A, A.t())
        #    edge_index_graph, edge_attr_graph = dense_to_sparse(A_)
        #    D_g = A_.sum(1)
        #    D_g = (1./(D_g+1)).reshape(-1,1)
        #    One = torch.ones((A_.shape[0],1), device=self.args.device)
        #    D_g_inv = torch.where(D_g>One, One, D_g)
        #    graph_rep1_temp = torch.zeros((graph_rep.shape[0],6), device=self.args.device)
        #    graph_rep2_temp = torch.zeros((graph_rep.shape[0],graph_rep.shape[1]-6), device=self.args.device)
        #    graph_rep1 = torch.cat((graph_rep[:, :-6], graph_rep1_temp), dim=1)
        #    graph_rep2 = torch.cat((graph_rep2_temp,graph_rep[:, -6:]), dim=1)
            
            
            
            #out1, out2 = self.GCN2(graph_rep[:, :-6], graph_rep[:, -6:], edge_index_graph, D_g_inv)
            #out1, out2 = self.GCN2(graph_rep1, graph_rep2, edge_index_graph, D_g_inv)
            #out_f = out1 + out2
            #out_f = torch.cat((out1, out2), dim=1)
        #    out_f = self.full_layer(graph_rep[:, :-6], graph_rep[:, -6:])
        #    loss, pred, logits = self.ClassifierO(out_f, label)
            #loss_MI2 = self.loss_MI(out1, out2)
         #   logits_f = F.softmax(logits, dim=-1) 
            #logits_f = F.softmax(logits, dim=-1) 
         #   pred_f = logits_f.max(1)[1]
        #    acc = pred_f.eq(label).sum()
        #    return loss, pred_f, logits_f, acc, self.thred 
        #else:
        #    for data_ in training_sample:
        #        data_ = data_.to(self.args.device)
        #        sample_x_, sample_edge_index, sample_batch, sample_y = data_.x, data_.edge_index, data_.batch, data_.y 
        #    sample_x = sample_x_[:,:self.args.num_features]
        #    sample_view2 = sample_x_[:,self.args.num_features:]
        #    sample_D = (1./(sample_view2[:,0]+1)).reshape(-1,1)
        #    sample_OneM = torch.ones((sample_x.shape[0],1), device=self.args.device)
        #    sample_D_inv = torch.where(sample_D>sample_OneM, sample_OneM, sample_D)
        #    sample_x_temp = torch.zeros((sample_x.shape[0], 7), device=self.args.device)
       #     sample_view2_temp = torch.zeros((sample_x.shape[0], self.num_features), device=self.args.device)
       #     sample_x = torch.cat((sample_x,sample_x_temp), dim=1)
       #     sample_view2 = torch.cat((sample_view2_temp, sample_view2), dim=1)
       #     sample_feature_graphstruc = sample_y[:,1:]
       #     _, sample_num = sample_batch.unique(return_counts=True)
       #     sample_num = 1./sample_num
       #     #sample_feature_graphstruc = sample_feature_graphstruc * sample_num.reshape(-1,1)
       #     sample_feature_graphstruc_ = torch.cat((sample_feature_graphstruc[:,0:1]* sample_num.reshape(-1,1),sample_feature_graphstruc[:,1:-1]), dim=1)
      #      #sample_feature_graphstruc_ = torch.cat((sample_feature_graphstruc_,sample_feature_graphstruc[:,-1].reshape(-1,1)* sample_num.reshape(-1,1)), dim=1)
       #     sample_feature_graphstruc = sample_feature_graphstruc_
       #     #sample_feature_graphstruc = self.BN2(sample_feature_graphstruc)            
       #     sample_q, sample_p = self.GCN(sample_x, sample_view2, sample_edge_index, sample_D_inv)
       #     sample_q_final =gmp(sample_q, sample_batch)            
       #     sample_p_final =gmp(sample_p, sample_batch)
                        
       #     sample_z_final = torch.cat((sample_q_final, sample_p_final), dim=1)
       #     sample_graph_rep = torch.cat((sample_z_final, sample_feature_graphstruc), dim=1)
      #      graph_rep_test = torch.cat((sample_graph_rep, graph_rep), dim=0)
               

            #graph_test_base = torch.cat((gmp(sample_x_,sample_batch), sample_feature_graphstruc), dim=1)
            #graph_test_base_t = torch.cat((gmp(x_,batch), feature_graphstruc), dim=1)
            #graph_test_base = torch.cat((graph_test_base, graph_test_base_t), dim=0)
            #sim = self.DotProductSimilarity(graph_test_base.unsqueeze(1), graph_test_base.unsqueeze(0))
            #sim = F.cosine_similarity(graph_test_base.unsqueeze(1), graph_test_base.unsqueeze(0), dim=-1)
            #sim = F.cosine_similarity(graph_rep_test[:,0:sample_z_final.shape[1]].unsqueeze(1), graph_rep_test[:,0:sample_z_final.shape[1]].unsqueeze(0), dim=-1)
       #     sim = torch.mm(graph_rep_test[:,0:sample_z_final.shape[1]], graph_rep_test[:,0:sample_z_final.shape[1]].t())
            #sim = torch.mm(graph_test_base, graph_test_base.t())
            #sim = torch.mm(graph_rep_test, graph_rep_test.t())
        #    mask = torch.zeros((graph_rep.shape[0],graph_rep.shape[0]), device=self.args.device)
        #    sim[sample_graph_rep.shape[0]:,sample_graph_rep.shape[0]:] = mask
                        
        #    zero_m = torch.zeros((graph_rep_test.shape[0], graph_rep_test.shape[0]), device=self.args.device)
       #     one_m = torch.ones((graph_rep_test.shape[0], graph_rep_test.shape[0]), device=self.args.device)
                        #A = sim/(torch.max(sim))
                        #thred = torch.quantile(sim, 0.7, dim=1).reshape(-1,1)
       #     if best_thred == None:
       #         thred_test = self.thred
       #     else:
       #         thred_test = best_thred
       #     A = torch.where(sim>thred_test, one_m, zero_m)
            #A = torch.where(sim>self.thred, one_m, zero_m)
                        #A_ = torch.maximum(A, A.t())
       #     A_ = torch.maximum(A, A.t())
      #      edge_index_graph, edge_attr_graph = dense_to_sparse(A_)
      #      D_g = A_.sum(1)
      #      D_g = (1./(D_g+1)).reshape(-1,1)
      #      One = torch.ones((A_.shape[0],1), device=self.args.device)
      #      D_g_inv = torch.where(D_g>One, One, D_g)
      #      graph_rep1_temp = torch.zeros((graph_rep_test.shape[0],6), device=self.args.device)
      #      graph_rep2_temp = torch.zeros((graph_rep_test.shape[0],graph_rep_test.shape[1]-6), device=self.args.device)
      #      graph_rep1 = torch.cat((graph_rep_test[:, :-6], graph_rep1_temp), dim=1)
      #      graph_rep2 = torch.cat((graph_rep2_temp,graph_rep_test[:, -6:]), dim=1)
            
            #out1, out2 = self.GCN2(graph_rep_test[:, :-6], graph_rep_test[:, -6:], edge_index_graph, D_g_inv)
            #out1, out2 = self.GCN2(graph_rep1, graph_rep2, edge_index_graph, D_g_inv)
            #out_f = (out1 + out2)[sample_graph_rep.shape[0]:]
            #out_f = torch.cat((out1, out2), dim=1)[sample_graph_rep.shape[0]:]
       #     out_f = self.full_layer(graph_rep_test[:, :-6], graph_rep_test[:, -6:])[sample_graph_rep.shape[0]:]
       #     loss, pred, logits = self.ClassifierO(out_f, label)
            #loss_MI2 = self.loss_MI(out1, out2)
       #     logits_f = F.softmax(logits, dim=-1) 
            #logits_f = F.softmax(logits, dim=-1) 
       #     pred_f = logits_f.max(1)[1]
       #     return loss, pred_f, logits_f
            
            
#        if self.training:
            #graph_rep = self.BN(graph_rep)
#            sim = torch.mm(graph_rep, graph_rep.t()) 
            #print(sim)
#            zero_m = torch.zeros((graph_rep.shape[0], graph_rep.shape[0]), device=self.args.device)
#            one_m = torch.zeros((graph_rep.shape[0], graph_rep.shape[0]), device=self.args.device)
            #A = sim/(torch.max(sim))
#            self.thred = torch.quantile(sim, 0.7)
#            A = torch.where(sim>self.thred, one_m, zero_m)
#            A_ = torch.maximum(A, A.t())
#            edge_index_graph, edge_attr_graph = dense_to_sparse(A_)    
#            out1, _ = self.conv_graph(graph_rep, edge_index_graph)#, edge_weight=edge_attr_graph
#            out1 = self.relu(out1)
#            out1 = F.dropout(out1, p=self.args.dropout_ratio, training=self.training)
#            out2, _ = self.conv_graph2(out1, edge_index_graph)
#            out2 = self.relu(out2)
#            out2 = F.dropout(out2, p=self.args.dropout_ratio, training=self.training)
#            out3, _ = self.conv_graph3(out2, edge_index_graph)
#            out_ = torch.cat((out1, out2), dim=1)
#            out_ = torch.cat((out_, out3), dim=1)
#            loss, pred, logits = self.ClassifierO(out_, label)
            
#            return loss+MIloss+loss_1, pred, logits
#        else:
#            for data_ in training_sample:
#                data_ = data_.to(self.args.device)
#                sample_x_, sample_edge_index, sample_batch, sample_y = data_.x, data_.edge_index, data_.batch, data_.y  	
            #sample_x_, sample_edge_index, sample_batch, sample_y = training_sample.x, training_sample.edge_index, training_sample.batch, training_sample.y  
#            sample_x = sample_x_[:,:self.args.num_features]
#            sample_view2 = sample_x_[:,self.args.num_features:]
#            sample_D = (1./(sample_view2[:,0]+1)).reshape(-1,1)
#            sample_OneM = torch.ones((sample_x.shape[0],1), device=self.args.device)
#            sample_D_inv = torch.where(sample_D>sample_OneM, sample_OneM, sample_D)
#            sample_x_temp = torch.zeros((sample_x.shape[0], 7), device=self.args.device)
#            sample_view2_temp = torch.zeros((sample_x.shape[0], self.num_features), device=self.args.device)
#            sample_x = torch.cat((sample_x,sample_x_temp), dim=1)
#            sample_view2 = torch.cat((sample_view2_temp, sample_view2), dim=1)
            
            #sample_label = sample_y[:,0]
#            sample_feature_graphstruc = sample_y[:,1:]
#            _, sample_num = sample_batch.unique(return_counts=True)
#            sample_num = 1/sample_num
#            sample_feature_graphstruc = sample_feature_graphstruc * sample_num.reshape(-1,1)
#            sample_feature_graphstruc = self.BN(sample_feature_graphstruc)            
#            sample_q, sample_p = self.GCN(sample_x, sample_view2, sample_edge_index, sample_D_inv)
 #           sample_q_final =gmp(sample_q, sample_batch)            
#            sample_p_final =gmp(sample_p, sample_batch)
            
#            sample_z_final = torch.cat((sample_q_final, sample_p_final), dim=1)
#            sample_graph_rep = torch.cat((sample_z_final, sample_feature_graphstruc), dim=1)
            

#            graph_rep_test = torch.cat((sample_graph_rep, graph_rep), dim=0)
            #graph_rep_test = self.BN(graph_rep_test)
#            sim = torch.mm(graph_rep_test, graph_rep_test.t())
#            mask = torch.zeros((graph_rep.shape[0],graph_rep.shape[0]), device=self.args.device)
#            sim[sample_graph_rep.shape[0]:,sample_graph_rep.shape[0]:] = mask
            
#            zero_m = torch.zeros((graph_rep_test.shape[0], graph_rep_test.shape[0]), device=self.args.device)
#            one_m = torch.zeros((graph_rep_test.shape[0], graph_rep_test.shape[0]), device=self.args.device)
            #A = sim/(torch.max(sim))
            #thred = torch.quantile(sim, 0.7, dim=1).reshape(-1,1)
#            A = torch.where(sim>self.thred, one_m, zero_m)
            #A_ = torch.maximum(A, A.t())
#            edge_index_graph, edge_attr_graph = dense_to_sparse(A)
#            out1, _ = self.conv_graph(graph_rep_test, edge_index_graph)#, edge_weight=edge_attr_graph
#            out1 = self.relu(out1)
#            out1 = F.dropout(out1, p=self.args.dropout_ratio, training=self.training)
#            out2, _ = self.conv_graph2(out1, edge_index_graph)
#            out2 = self.relu(out2)
#            out2 = F.dropout(out2, p=self.args.dropout_ratio, training=self.training)
#            out3, _ = self.conv_graph3(out2, edge_index_graph)
#            out_ = torch.cat((out1, out2), dim=1)
#            out_ = torch.cat((out_, out3), dim=1)
#            out_ = out_[sample_graph_rep.shape[0]:]
#            loss, pred, logits = self.ClassifierO(out_, label)
            
            
            
        #loss_MI = self.MIloss(z_1, z_2) + self.MIloss(z_final,feature_graphstruc)  
        #loss_MI = self.MIloss(z, z_1) + self.MIloss(z, z_2)
        #logits_f = F.softmax(logit1 + logits, dim=-1)
        #pred_f = 
#        return loss, MIloss, pred, logits
        #print('train:{},loss_1:{}, MIloss:{}, loss:{}'.format(self.training, loss_1, MIloss, loss))

            #return loss+loss_1, pred, logits