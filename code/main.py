import numpy as np
import pandas as pd
import argparse, sys, json, random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from utils import *
from model import *



def init_argparse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--type', type=str, default="feature", help="one of these: feature, comparison, test")
    parser.add_argument('--data_source', type=str, default="K562", help="which cell line to train and predict")
    parser.add_argument('--threshold', type=float, default=-3, help="threshold of SL determination")
    parser.add_argument('--specific_graph', type=lambda s:[item for item in s.split("%") if item != ""], default=["SL"], help="lists of cell-specific graphs to use.")
    parser.add_argument('--indep_graph', type=lambda s:[item for item in s.split("%") if item != ""], 
                    default=['PPI-genetic','PPI-physical','co-ess','co-exp'], help="lists of cell-independent graphs to use.")
    parser.add_argument('--node_feats', type=str, default="raw_omics", help="gene node features")

    parser.add_argument('--balanced', type=int, default=0, help="whether the negative and positive samples are balanced")
    parser.add_argument('--pos_weight', type=float, default=5, help="weight for positive samples in loss function")
    parser.add_argument('--CCLE', type=int, default=0, help="whether or not include CCLE features into node features")
    parser.add_argument('--CCLE_dim', type=int, default=64, help="dimension of embeddings for each type of CCLE omics data")
    parser.add_argument('--node2vec_feats', type=int, default=0, help="whether or not using node2vec embeddings")

    parser.add_argument('--model', type=str, default="GCN_pool", help="model type")
    parser.add_argument('--pooling', type=str, default="max", help="type of pooling operations")
    parser.add_argument('--LR', type=float, default=0.001, help="learning rate")
    parser.add_argument('--epochs', type=int, default=1000, help="number of maximum training epochs")
    parser.add_argument('--out_channels', type=int, default=64, help="dimension of output channels")
    parser.add_argument('--patience', type=int, default=200, help="patience in early stopping")
    parser.add_argument('--training_percent', type=float, default=0.70, help="proportion of the SL data as training set")
    
    parser.add_argument('--save_results', type=int, default=1, help="whether to save test results into json")
    parser.add_argument('--predict_novel_genes', type=int, default=0, help="whether to predict on novel out of samples")
    parser.add_argument('--novel_cellline', type=str, default="Jurkat", help="name of novel celllines")

    args = parser.parse_args()

    return args


def train_model(model, optimizer, data, device, train_pos_edge_index, train_neg_edge_index):
    model.train()
    optimizer.zero_grad()
    x = data.x.to(device)

    temp_z_list = []
    for i, edge_index in enumerate(data.edge_index_list):
        edge_index = edge_index.to(device)
        if args.model == 'GCN_multi' or args.model == 'GAT_multi':
            i = torch.tensor(i).to(device)
            temp_z = model.encode(x, edge_index, i)
        else:
            temp_z = model.encode(x, edge_index)
        temp_z_list.append(temp_z)
    
    z = torch.cat(temp_z_list,1)
    # transpose is used to transform the data from (batch, # graphs, # features) into (batch, # features, # graphs)
    # the pooling operation is performed on the third dimension (graphs)
    z = z.unsqueeze(1).reshape(z.shape[0],len(data.edge_index_list),-1).transpose(1,2)
    
    if args.pooling == "max":
        z = F.max_pool2d(z, (1,len(data.edge_index_list))).squeeze(2)
    elif args.pooling == "mean":
        z = F.avg_pool2d(z, (1,len(data.edge_index_list))).squeeze(2)

    link_logits = model.decode(z, train_pos_edge_index, train_neg_edge_index)
    #link_probs = link_logits.sigmoid()
    link_labels = get_link_labels(train_pos_edge_index, train_neg_edge_index, device)

    if args.balanced:
        pos_weight = torch.tensor(1)
    else:
        pos_weight = torch.tensor(args.pos_weight)

    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels, pos_weight=pos_weight)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test_model(model, optimizer, data, device, pos_edge_index, neg_edge_index):
    model.eval()
    results = {}
    x = data.x.to(device)

    temp_z_list = []
    for i, edge_index in enumerate(data.edge_index_list):
        edge_index = edge_index.to(device)
        if args.model == 'GCN_multi' or args.model == 'GAT_multi':
            i = torch.tensor(i).to(device)
            temp_z = model.encode(x, edge_index, i)
        else:
            temp_z = model.encode(x, edge_index)
        temp_z_list.append(temp_z)
    
    z = torch.cat(temp_z_list,1)
    z = z.unsqueeze(1).reshape(z.shape[0],len(data.edge_index_list),-1).transpose(1,2)
    
    if args.pooling == "max":
        z = F.max_pool2d(z, (1,len(data.edge_index_list))).squeeze(2)
    elif args.pooling == "mean":
        z = F.avg_pool2d(z, (1,len(data.edge_index_list))).squeeze(2)
    
    link_logits = model.decode(z, pos_edge_index, neg_edge_index)
    link_probs = link_logits.sigmoid()
    link_labels = get_link_labels(pos_edge_index, neg_edge_index, device)

    if args.balanced:
        pos_weight = torch.tensor(1)
    else:
        pos_weight = torch.tensor(args.pos_weight)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels, pos_weight=pos_weight)

    results['auc'] = roc_auc_score(link_labels.cpu(), link_probs.cpu())
    results['aupr'] = average_precision_score(link_labels.cpu(), link_probs.cpu())
    link_pred_labels = (link_probs > 0.5).long()
    results['accuracy'] = accuracy_score(link_labels.cpu(), link_pred_labels.cpu())
    results['f1'], results['precision'], results['recall'], results['thres_f1'] = compute_fmax(link_labels.cpu().numpy(), link_probs.cpu().numpy())
    results['ranking_metrics'] = ranking_metrics(link_labels.cpu().numpy(), link_probs.cpu().numpy())
    results['true_label'] = link_labels.cpu().numpy().tolist()
    results['pred'] = link_probs.cpu().numpy().tolist()
    return float(loss), results


@torch.no_grad()
def predict_oos(model, optimizer, data, device, pos_edge_index, neg_edge_index):
    model.eval()
    x = data.x.to(device)

    temp_z_list = []
    for i, edge_index in enumerate(data.edge_index_list):
        edge_index = edge_index.to(device)
        if args.model == 'GCN_multi' or args.model == 'GAT_multi':
            i = torch.tensor(i).to(device)
            temp_z = model.encode(x, edge_index, i)
        else:
            temp_z = model.encode(x, edge_index)
        temp_z_list.append(temp_z)
    
    z = torch.cat(temp_z_list,1)
    z = z.unsqueeze(1).reshape(z.shape[0],len(data.edge_index_list),-1).transpose(1,2)

    if args.pooling == "max":
        z = F.max_pool2d(z, (1,len(data.edge_index_list))).squeeze(2)
    elif args.pooling == "mean":
        z = F.avg_pool2d(z, (1,len(data.edge_index_list))).squeeze(2)
    
    # due to the huge size of the input data, split them into 100 batches
    batch_num = 100
    step_size_neg = int(neg_edge_index.shape[1]/batch_num) + 1
    link_probs = []
    for j in tqdm(range(batch_num)):
        temp_link_logits = model.decode(z, pos_edge_index, neg_edge_index[:,(j*step_size_neg):((j+1)*step_size_neg)])
        temp_link_probs = temp_link_logits.sigmoid()
        link_probs.extend(temp_link_probs.cpu().numpy().tolist())

    return link_probs


if __name__ == "__main__":
    args = init_argparse()
    print(args)
    graph_input = args.specific_graph + args.indep_graph
    print("Number of input graphs: {}".format(len(graph_input)))
    if len(graph_input) == 0:
        print("Please specify input graph features...")
        sys.exit(0)
    # load data
    data, SL_data_train, SL_data_val, SL_data_test, SL_data_oos, gene_mapping = generate_torch_geo_data(args.data_source, args.CCLE, args.CCLE_dim, args.node2vec_feats, 
                                    args.threshold, graph_input, args.node_feats, args.predict_novel_genes, args.training_percent)

    num_features = data.x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    if args.model == "GCN_pool":
        model = GCN_pool(num_features, args.out_channels, len(data.edge_index_list)).to(device)
    elif args.model == 'GCN_conv':
        model = GCN_conv(num_features, args.out_channels, len(data.edge_index_list)).to(device)
    elif args.model == 'GCN_multi':
        model = GCN_multi(num_features, args.out_channels, len(data.edge_index_list)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR)

    # generate SL torch data
    train_pos_edge_index, train_neg_edge_index = generate_torch_edges(SL_data_train, args.balanced, True, device)
    val_pos_edge_index, val_neg_edge_index = generate_torch_edges(SL_data_val, True, False, device)
    test_pos_edge_index, test_neg_edge_index = generate_torch_edges(SL_data_test, True, False, device)
    if args.predict_novel_genes:
        oos_pos_edge_index, oos_neg_edge_index = generate_torch_edges(SL_data_oos, False, False, device)

    print("Number of training samples:", train_pos_edge_index.size())

    train_losses = []
    valid_losses = []
    
    # initialize the early_stopping object
    random_key = random.randint(1,100000000)
    checkpoint_path = "checkpoint/{}.pt".format(str(random_key))
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, reverse=False, path=checkpoint_path)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_model(model, optimizer, data, device, train_pos_edge_index, train_neg_edge_index)
        train_losses.append(train_loss)
        val_loss, results = test_model(model, optimizer, data, device, val_pos_edge_index, val_neg_edge_index)
        valid_losses.append(val_loss)
        print('Epoch: {:03d}, loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}, val_loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, thres: {:.4f}, Ranking: {:.4f}'.format(epoch, 
                                        train_loss, results['auc'], results['aupr'], val_loss, results['precision'], results['recall'], results['f1'], results['thres_f1'], results['ranking_metrics']))
        
        early_stopping(results['f1'], model)
        #early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early Stopping!!!")
            break
    

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_path))

    test_loss, results = test_model(model, optimizer, data, device, test_pos_edge_index, test_neg_edge_index)
    print("\ntest result:")
    print('AUC: {:.4f}, AP: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, thres: {:.4f}, Ranking: {:.4f}'.format(results['auc'], 
                                                            results['aupr'], results['accuracy'], results['precision'], results['recall'], results['f1'], results['thres_f1'], results['ranking_metrics']))
    
    save_dict = {**vars(args), **results}
    
    
    if args.predict_novel_genes:
        print("Predicting on novel genes...")
        oos_preds = predict_oos(model, optimizer, data, device, oos_pos_edge_index, oos_neg_edge_index)
        save_dict['gene_mapping'] = gene_mapping
        save_dict['oos_samples_1'] = SL_data_oos['gene1'].values.tolist()
        save_dict['oos_samples_2'] = SL_data_oos['gene2'].values.tolist()
        save_dict['oos_pred'] = oos_preds
        
    if args.save_results:
        with open("../results/{}_{}.json".format(args.type, str(random_key)),"w") as f:
            json.dump(save_dict, f)