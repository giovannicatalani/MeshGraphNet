import torch
import numpy as np
import pandas as pd
from tqdm import trange
import copy
import os
from torch_geometric.data import Data
import torch_scatter
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt

from utils import *
from model import *


# Directories
data_directory = r'C:\Users\giova\Python_Projects\MeshGraphNet\data'
data = np.load(data_directory + '\db_reduced_total4000_compressible.npy', allow_pickle = True).item()

#Options
train_model = True
predict_test = False

#Pre process dataset 
dataset = create_dataset(data)
stats_list = get_stats(dataset)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# build model
num_node_features = dataset[0].x.shape[1]
num_edge_features = dataset[0].edge_attr.shape[1]
num_classes = 1 # the dynamic variables have the shape of 1 (pressure)

for args in [
        {'model_type': 'meshgraphnet', 
         'graph_size': 2000, 
         'num_layers': 15,
         'batch_size': 4, 
         'hidden_dim': 32, 
         'epochs': 500,
         'opt': 'adam', 
         'opt_scheduler': 'none', 
         'opt_restart': 0, 
         'weight_decay': 5e-4, 
         'lr': 0.001,
         'train_size': 500, 
         'test_size': 139,
         'compr' : 'yes',
         'device':device,
         'shuffle': True, 
         'save_velo_val': True,
         'save_best_model': True, 
         'checkpoint_dir': data_directory + '/best_models/',
         'postprocess_dir': data_directory + '/2d_loss_plots/'},
    ]:
        args = objectview(args)

model_name= '\model_nl' + str(args.num_layers) + '_bs' + str(args.batch_size) + '_hd' + str(args.hidden_dim) + \
             + '_ep'+ '_wd0.0005_lr0.001_nodes' + str(args.graph_size)+ '_shuff_True_tr' + str(args.train_size) + \
             '_te' + str(args.test_size) +  '_compryes.pt'

model_path = data_directory + r'\best_models' +  model_name 
model = MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes,
                            args).to(device)

#Predict 
def predict(loader, model, model_path, stats_list, args):
    args.device = device

    model.load_state_dict(torch.load(model_path))
    model = model.to(args.device)
    model.eval()

    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y] = stats_list
    (mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y)=(mean_vec_x.to(device),
        std_vec_x.to(device),mean_vec_edge.to(device),std_vec_edge.to(device),mean_vec_y.to(device),std_vec_y.to(device))
   
    
    # Create empty lists to store the predictions and true labels
    preds = []
    labels = []
    inputs = []

    for data in loader :
        data=data.to(args.device) 
        with torch.no_grad():
            pred = model(data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge)
            pred = unnormalize( pred, mean_vec_y, std_vec_y ).cpu().numpy()
            true = data.y.cpu().numpy()
            input = data.x.cpu().numpy()[0]
            inputs.append(input)
            preds.append(pred)
            labels.append(true)

    # Concatenate the predictions and true labels into NumPy arrays
    preds = np.concatenate(preds, axis=1)
    labels = np.concatenate(labels, axis=1)
    #inputs = np.concatenate(inputs, axis=0)

    return preds, labels, inputs





preds, labels, inputs = predict(dataset[args.train_size:], data['nodes'], model, model_path, stats_list, args)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.tricontourf(data['nodes'][:,0], data['nodes'][:,1],preds[:,60], cmap='viridis')
ax1.fill(data['airfoil'][:, 0], data['airfoil'][:, 1], facecolor='white', edgecolor=None)
ax1.set_title('Predictions')
ax2.tricontourf(data['nodes'][:,0], data['nodes'][:,1],labels[:,60], cmap='viridis')
ax2.fill(data['airfoil'][:, 0], data['airfoil'][:, 1], facecolor='white', edgecolor=None)
ax2.set_title('True Values')
plt.show() 