# %matplotlib inline

import json
import collections
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit as masking
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import GCNConv

import networkx as nx

with open("data/musae_git_features.json") as json_data:
    data_raw = json.load(json_data)
    feats=data_raw
edges=pd.read_csv("data/musae_git_edges.csv")
target_df=pd.read_csv("data/musae_git_target.csv")

# print("5 top nodes labels")
# print(target_df.head(5).to_markdown())
# print()
# print("5 last nodes")
# print(target_df.tail(5).to_markdown())
#
# print(edges.head(5))

# plt.hist(target_df.ml_target,bins=4);
# plt.title("Classes distribution")
# plt.show()

# print(target_df.ml_target.head(5))

# plt.hist(feat_counts,bins=20)
# plt.title("Number of features per graph distribution")
# plt.show()
#
plt.hist(feats,bins=50)
plt.title("Features distribution")
plt.show()

# def encode_data(light=False,n=60):
#     if light==True:
#         nodes_included=n
#     elif light==False:
#         nodes_included=len(data_raw)
#
#     data_encoded={}
#     for i in range(nodes_included):#
#         one_hot_feat=np.array([0]*(max(feats)+1))
#         this_feat=data_raw[str(i)]
#         one_hot_feat[this_feat]=1
#         data_encoded[str(i)]=list(one_hot_feat)
#
#     if light==True:
#         sparse_feat_matrix=np.zeros((1,max(feats)+1))
#         for j in range(nodes_included):
#             temp=np.array(data_encoded[str(j)]).reshape(1,-1)
#             sparse_feat_matrix=np.concatenate((sparse_feat_matrix,temp),axis=0)
#         sparse_feat_matrix=sparse_feat_matrix[1:,:]
#         return(data_encoded,sparse_feat_matrix)
#     elif light==False:
#         return(data_encoded, None)