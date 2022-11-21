import numpy as np
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.nn import ResGatedGraphConv, TransformerConv


class GCN_pool(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_graph):
        super(GCN_pool, self).__init__()
        self.num_graph = num_graph
        self.conv1 = GCNConv(in_channels, 2*out_channels)
        self.conv2 = GCNConv(2*out_channels, out_channels)
        self.fc1 = torch.nn.Linear(2*out_channels, out_channels)
        self.fc2 = torch.nn.Linear(out_channels, int(out_channels/2))
        self.fc3 = torch.nn.Linear(int(out_channels/2), int(out_channels/4))
        self.fc4 = torch.nn.Linear(int(out_channels/4), 1)
        self.dropout = torch.nn.Dropout(0.5)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        x = torch.cat((z[edge_index[0]],z[edge_index[1]]),dim=1)
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = self.fc2(x).relu()
        x = self.dropout(x)
        x = self.fc3(x).relu()
        x = self.dropout(x)
        x = self.fc4(x)
        return torch.squeeze(x)
    

class GCN_conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_graph):
        super(GCN_conv, self).__init__()
        self.num_graph = num_graph
        self.conv1 = GCNConv(in_channels, 2*out_channels)
        self.conv2 = GCNConv(2*out_channels, out_channels)
        
        # in_channels is 1, number of filters is 32
        self.filter_num = 128
        self.conv_net = torch.nn.Conv2d(1,self.filter_num, (1,num_graph))
        
        self.fc1 = torch.nn.Linear(2*out_channels, out_channels)
        self.fc2 = torch.nn.Linear(out_channels, int(out_channels/2))
        self.fc3 = torch.nn.Linear(int(out_channels/2), int(out_channels/4))
        self.fc4 = torch.nn.Linear(int(out_channels/4), 1)
        self.dropout = torch.nn.Dropout(0.5)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        # adding one more dimension as channel, set to 1
        z = z.unsqueeze(1)
        z = self.conv_net(z).squeeze(3)
        z = F.max_pool2d(z, (self.filter_num,1)).squeeze(1)
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x = torch.cat((z[edge_index[0]],z[edge_index[1]]),dim=1)
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = self.fc2(x).relu()
        x = self.dropout(x)
        x = self.fc3(x).relu()
        x = self.dropout(x)
        x = self.fc4(x)
        return torch.squeeze(x)
    
    
class GAT_Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_graph):
        super(GAT_Net, self).__init__()
        self.conv1 = GATv2Conv(in_channels, 2*out_channels, heads=1, dropout=0.3)
        self.conv2 = GATv2Conv(2*out_channels, out_channels, dropout=0.3)
        #self.conv1 = TransformerConv(in_channels, 2*out_channels, heads=1, dropout=0.3)
        #self.conv2 = TransformerConv(2*out_channels, out_channels, heads=1, dropout=0.3)
        self.fc1 = torch.nn.Linear(num_graph*2*out_channels, num_graph*out_channels)
        self.fc2 = torch.nn.Linear(num_graph*out_channels, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 1)
        self.dropout = torch.nn.Dropout(0.5)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x = torch.cat((z[edge_index[0]],z[edge_index[1]]),dim=1)
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = self.fc2(x).relu()
        x = self.dropout(x)
        x = self.fc3(x).relu()
        x = self.dropout(x)
        x = self.fc4(x)
        return torch.squeeze(x)


class GCN_multi(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_graph):
        super(GCN_multi, self).__init__()

        self.conv1_list = []
        self.conv2_list = []
        for _ in range(num_graph):
            self.conv1_list.append( GCNConv(in_channels, 2*out_channels) )
            self.conv2_list.append( GCNConv(2*out_channels, out_channels) )

        self.fc1 = torch.nn.Linear(2*out_channels, out_channels)
        self.fc2 = torch.nn.Linear(out_channels, int(out_channels/2))
        self.fc3 = torch.nn.Linear(int(out_channels/2), int(out_channels/4))
        self.fc4 = torch.nn.Linear(int(out_channels/4), 1)
        self.dropout = torch.nn.Dropout(0.5)
        self.num_graph = num_graph

    def encode(self, x, edge_index, graph_idx):
        x = self.conv1_list[graph_idx].cuda()(x, edge_index).relu()
        x = self.conv2_list[graph_idx].cuda()(x, edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x = torch.cat((z[edge_index[0]],z[edge_index[1]]),dim=1)
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = self.fc2(x).relu()
        x = self.dropout(x)
        x = self.fc3(x).relu()
        x = self.dropout(x)
        x = self.fc4(x)
        return torch.squeeze(x)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, reverse=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.reverse = reverse
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):
        if self.reverse == True:
            # loss
            score = -val_loss
        else:
            # AUC/AUPR
            score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss