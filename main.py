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
data_test = np.load(data_directory + '\db_reduced_random2000_compressible.npy', allow_pickle = True).item()

#Options
train_model = True
predict_test = False

#Pre process dataset 
dataset = create_dataset(data)
dataset_test = create_dataset(data_test)

def train(dataset, device, stats_list, args):
    '''
    Performs a training loop on the dataset for MeshGraphNets. Also calls
    test and validation functions.
    '''

    df = pd.DataFrame(columns=['epoch','train_loss','test_loss'])

    #Define the model name for saving 
    model_name='model_nl'+str(args.num_layers)+'_bs'+str(args.batch_size) + \
               '_hd'+str(args.hidden_dim)+'_ep'+str(args.epochs)+'_wd'+str(args.weight_decay) + \
               '_lr'+str(args.lr)+'_nodes' + str(args.graph_size) + '_shuff_'+str(args.shuffle)+'_tr'+str(args.train_size)+'_te'+str(args.test_size) + \
               '_compr' + str(args.compr)


    #torch_geometric DataLoaders are used for handling the data of lists of graphs
    loader = DataLoader(dataset[:args.train_size], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[args.train_size:], batch_size=args.batch_size, shuffle=False)

    #The statistics of the data are decomposed
    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y] = stats_list
    (mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y)=(mean_vec_x.to(device),
        std_vec_x.to(device),mean_vec_edge.to(device),std_vec_edge.to(device),mean_vec_y.to(device),std_vec_y.to(device))

    # build model
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    num_classes = 1 # the dynamic variables have the shape of 1 (pressure)

    model = MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes,
                            args).to(device)
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    losses = []
    test_losses = []
    best_test_loss = np.inf
    best_model = None
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        num_loops=0
        for batch in loader:
            #Note that normalization must be done before it's called. The unnormalized
            #data needs to be preserved in order to correctly calculate the loss
            batch=batch.to(device)
            opt.zero_grad()         #zero gradients each time
            pred = model(batch,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge)
            loss = model.loss(pred,batch,mean_vec_y,std_vec_y)
            loss.backward()         #backpropagate loss
            opt.step()
            total_loss += loss.item()
            num_loops+=1
        total_loss /= num_loops
        losses.append(total_loss)

        #Every tenth epoch, calculate acceleration test loss and velocity validation loss
        if epoch % 10 == 0:
            
            test_loss = test(test_loader,device,model,mean_vec_x,std_vec_x,mean_vec_edge,
                                 std_vec_edge,mean_vec_y,std_vec_y)

            test_losses.append(test_loss.item())

            # saving model
            if not os.path.isdir( args.checkpoint_dir ):
                os.mkdir(args.checkpoint_dir)

            PATH = args.checkpoint_dir + model_name+'.csv'
            df.to_csv(PATH,index=False)

            #save the model if the current one is better than the previous best
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = copy.deepcopy(model)

        else:
              test_losses.append(test_losses[-1])
              
        
        df = pd.concat([df,pd.DataFrame([{'epoch': epoch, 'train_loss': losses[-1], 'test_loss': test_losses[-1]}])]) 

        if(epoch%50==0):
            
            print("train loss", str(round(total_loss,3)), "test loss", str(round(test_loss.item(),3)))


            if(args.save_best_model):

                PATH = args.checkpoint_dir + model_name+'.pt'
                torch.save(best_model.state_dict(), PATH )

    return test_losses, losses, best_model, best_test_loss, test_loader

def test(loader,device,test_model,
         mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y,
           save_model_preds=False, model_type=None):
  
    '''
    Calculates test set losses and validation set errors.
    '''

    loss=0
    velo_rmse = 0
    num_loops=0

    for data in loader:
        data=data.to(device)
        with torch.no_grad():

            #calculate the loss for the model given the test set
            pred = test_model(data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge)
            loss += test_model.loss(pred, data,mean_vec_y,std_vec_y)

        num_loops+=1
        # if velocity is evaluated, return velo_rmse as 0

    return loss/num_loops

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

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
         'device':'cuda',
         'shuffle': True, 
         'save_velo_val': True,
         'save_best_model': True, 
         'checkpoint_dir': data_directory + '/best_models/',
         'postprocess_dir': data_directory + '/2d_loss_plots/'},
    ]:
        args = objectview(args)

#To ensure reproducibility the best we can, here we control the sources of
#randomness by seeding the various random number generators used in this Colab
#For more information, see: https://pytorch.org/docs/stable/notes/randomness.html
import random
torch.manual_seed(5)  #Torch
random.seed(5)        #Python
np.random.seed(5)     #NumPy

if(args.shuffle):
  random.shuffle(dataset)

stats_list = get_stats(dataset)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device
    
if train_model:
    test_losses, losses, best_model, best_test_loss, test_loader = train(dataset, device, stats_list, args)

    print("Min test set loss: {0}".format(min(test_losses)))
    print("Minimum loss: {0}".format(min(losses)))


#Predict 
def predict(loader, nodes, model, model_path, stats_list, args):
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

if predict_test:

    # build model
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    num_classes = 1 # the dynamic variables have the shape of 1 (pressure)

    model = MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes,
                                args).to(device)
    model_name= '\model_nl15_bs4_hd32_ep500_wd0.0005_lr0.001_nodes2000_shuff_True_tr500_te139_compryes.pt'
    model_path = data_directory + r'\best_models' +  model_name 

    preds, labels, inputs = predict(dataset[args.train_size:], data['nodes'], model, model_path, stats_list, args)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.tricontourf(data['nodes'][:,0], data['nodes'][:,1],preds[:,60], cmap='viridis')
    ax1.fill(data['airfoil'][:, 0], data['airfoil'][:, 1], facecolor='white', edgecolor=None)
    ax1.set_title('Predictions')
    ax2.tricontourf(data['nodes'][:,0], data['nodes'][:,1],labels[:,60], cmap='viridis')
    ax2.fill(data['airfoil'][:, 0], data['airfoil'][:, 1], facecolor='white', edgecolor=None)
    ax2.set_title('True Values')
    plt.show() 

