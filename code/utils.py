import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
import matplotlib.pyplot as plt
import os, random
import pickle, json,itertools
import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import Node2Vec
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from networkx.generators.random_graphs import fast_gnp_random_graph,gnp_random_graph


def generate_unique_samples(cell_name):
    # process raw data
    if cell_name == "K562":
        df = pd.read_table("../data/raw_SL_experiments/K562/CRISPRi_K562_replicateAverage_GIscores_genes_inclnegs.txt", index_col=0)
    else:
        df = pd.read_table("../data/raw_SL_experiments/Jurkat/CRISPRi_Jurkat_emap_gene_filt.txt", index_col=0)
    
    # remove negative samples
    num_genes = df.shape[0]
    df = df.iloc[:(num_genes-1),:(num_genes-1)]
    
    num_genes = df.shape[0]
    
    # don't consider self interactions
    GI_matrix = df.values
    GI_indexs = np.triu_indices(num_genes, k=1)
    GI_values = GI_matrix[GI_indexs]
    
    # get corresponding gene names
    row_indexs = GI_indexs[0]
    col_indexs = GI_indexs[1]
    row_genes = df.index[row_indexs]
    col_genes = df.columns[col_indexs]
    
    all_samples = pd.DataFrame({'gene1':list(row_genes),'gene2':list(col_genes),'GI_scores':GI_values})
    all_samples.to_csv("../data/{}_GI_scores.csv".format(cell_name), index=False)
    
    return all_samples


def load_SL_data(cell_name, threshold=-3):
    data = pd.read_csv("../data/{}_GI_scores.csv".format(cell_name))
    data['label'] = data['GI_scores'] <= threshold
    all_genes = set(np.unique(data['gene1'])) | set(np.unique(data['gene2']))
    
    return data, all_genes


def load_graph_data(graph_type):
    if graph_type == 'PPI-genetic' or graph_type == 'PPI-physical':
        data = pd.read_csv("../data/BIOGRID-9606.csv", index_col=0)
        all_genes = set(data['Official Symbol Interactor A'].unique()) | set(data['Official Symbol Interactor B'].unique())
        
        if graph_type == 'PPI-physical':
            data = data[data['Experimental System Type'] == 'physical']
        else:
            data = data[data['Experimental System Type'] != 'physical']
        print("Number of edges of {}: {}".format(graph_type, data.shape[0]))

        data = data[['Official Symbol Interactor A','Official Symbol Interactor B']]
        data.rename(columns={'Official Symbol Interactor A':'gene1', 'Official Symbol Interactor B':'gene2'}, inplace=True)

        # make it indirected graph
        data_dup = data.reindex(columns=['gene2','gene1'])
        data_dup.columns = ['gene1','gene2']
        data = data.append(data_dup)
    elif graph_type == 'co-exp' or graph_type == 'co-ess':
        if graph_type == 'co-exp':
            data = pd.read_csv("../data/coexpression_exp_0.5.csv")
        elif graph_type == 'co-ess':
            data = pd.read_csv("../data/coexpression_ess_0.2.csv")

        # make it indirected graph
        data_dup = data.reindex(columns=['gene2','gene1'])
        data_dup.columns = ['gene1','gene2']
        data = data.append(data_dup)
    elif graph_type == "random":
        data = pd.read_csv("../data/BIOGRID-9606.csv", index_col=0)
        
        data = data[data['Experimental System Type'] == 'physical']
        data = data[['Official Symbol Interactor A','Official Symbol Interactor B']]
        data.rename(columns={'Official Symbol Interactor A':'gene1', 'Official Symbol Interactor B':'gene2'}, inplace=True)
        all_genes = set(data['gene1'].unique()) | set(data['gene2'].unique())
        dict_mapping = dict(zip(range(len(all_genes)), all_genes))
        
        num_nodes = len(all_genes)
        num_edges = data.shape[0]
        p = 2*num_edges/(num_nodes*(num_nodes-1))
        # make it more sparse
        p = p/10.0
        
        G = fast_gnp_random_graph(num_nodes, p)
        data = nx.convert_matrix.to_pandas_edgelist(G,source='gene1',target='gene2')
        print("generated number of edges: {}".format(G.number_of_edges()))

    return data



def choose_node_attribute(attr, gene_mapping, cell_name, graph_data):
    num_nodes = len(gene_mapping)
    if attr == "identity":
        x = np.identity(num_nodes)
    elif attr == 'random':
        x = np.random.randn(num_nodes, 4)
    elif attr == 'raw_omics':
        feat_list = ['exp','mut','cnv','ess']
        dict_list = []
        for feat in feat_list:
            temp_df = pd.read_table("../data/cellline_feats/{}_{}.txt".format(cell_name,feat),
                                        names=['gene','value'], sep=' ')
            # filter genes
            temp_df = temp_df[temp_df['gene'].isin(list(gene_mapping.keys()))]
            temp_dict = dict(zip(temp_df['gene'].values, temp_df['value'].values))
            dict_list.append(temp_dict)
        
        x = np.zeros((num_nodes, len(feat_list)))
        for col_idx, feat_dict in enumerate(dict_list):
            for key, value in feat_dict.items():
                row_idx = gene_mapping[key]
                x[row_idx, col_idx] = value
        # standardize features
        x = scale(x)
    elif attr == 'node2vec':
        # need to first build a torch data
        data_x = torch.tensor(np.random.randn(num_nodes, 128), dtype=torch.float)

        # concat all types of graph data
        graph_data_overall = pd.concat(graph_data)
        data_edge_index = torch.tensor([graph_data_overall['gene1'].values, graph_data_overall['gene2'].values], dtype=torch.long)
        
        data = Data(x=data_x, edge_index=data_edge_index)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=10, context_size=10, sparse=True).to(device)
        loader = model.loader(batch_size=512, shuffle=True, num_workers=28)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

        def train():
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss/len(loader)
        
        for epoch in range(1, 51):
            loss = train()
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        
        # get embeddings
        model.eval()
        x = model(torch.arange(data.num_nodes, device=device))
        print(x.size())
        x = x.cpu().detach().numpy()

    return x


def load_CCLE_feats(process_method, feats_list, gene_mapping, hidden_dim):
    # load raw data
    df_list = []
    print("loading CCLE...")
    for feat in feats_list:
        print("loading {}".format(feat))
        df = pd.read_csv("../data/CCLE/CCLE_{}.csv".format(feat), index_col=0)
        df.fillna(0, inplace=True)
        df.columns = list(map(lambda x:x.split(" ")[0], df.columns))
        df_list.append(df)
    
    # transform raw data
    embedding_list = []
    for df in df_list:
        embed = np.zeros((len(gene_mapping), hidden_dim))
        if process_method == 'PCA':
            temp_df = df[df.columns[df.columns.isin(list(gene_mapping.keys()))]]
            data = temp_df.values.T
            pca = PCA(n_components=hidden_dim)
            z = pca.fit_transform(data)

            # generate embeddings
            for i, gene_id in enumerate(temp_df.columns):
                embed[ gene_mapping[gene_id] ] = z[i]
        elif process_method == 'raw':
            temp_df = df[df.columns[df.columns.isin(list(gene_mapping.keys()))]]
            data = temp_df.values.T

            # generate embeddings
            embed = np.zeros((len(gene_mapping), data.shape[1]))
            for i, gene_id in enumerate(temp_df.columns):
                embed[ gene_mapping[gene_id] ] = data[i]
        
        embedding_list.append(embed)

    # combine embeddings
    combined_embeds = np.hstack(embedding_list)
    with open("../data/CCLE/embeddings_{}.npy".format(process_method), "wb") as f:
        np.save(f, combined_embeds)
    
    return combined_embeds


def merge_and_mapping(SL_data, graph_data_list, SL_genes):
    # use the union of SL genes and graph genes as all genes
    temp_concat_graph_data = pd.concat(graph_data_list)
    graph_genes = set(temp_concat_graph_data['gene1'].unique()) | set(temp_concat_graph_data['gene2'].unique())
    all_genes = sorted(list(SL_genes | graph_genes))

    gene_mapping = dict(zip(all_genes, range(len(all_genes))))

    # converting gene names to id
    # iterating over all graph types
    for i in range(len(graph_data_list)):
        graph_data_list[i]['gene1'] = graph_data_list[i]['gene1'].apply(lambda x:gene_mapping[x])
        graph_data_list[i]['gene2'] = graph_data_list[i]['gene2'].apply(lambda x:gene_mapping[x])
    
    SL_data['gene1'] = SL_data['gene1'].apply(lambda x:gene_mapping[x])
    SL_data['gene2'] = SL_data['gene2'].apply(lambda x:gene_mapping[x])

    return SL_data, graph_data_list, gene_mapping


    
def generate_torch_geo_data(cell_name, CCLE_feats_flag, CCLE_hidden_dim, node2vec_feats_flag, threshold, graph_input, attr, predict_novel_flag, training_percent):
    # load data
    SL_data, SL_genes = load_SL_data(cell_name, threshold)
    
    # generate SL torch data, split into train, valid, test
    all_idx = list(range(len(SL_data)))
    np.random.seed(5959)
    np.random.shuffle(all_idx)
    
    graph_data_list = []
    for graph_type in graph_input:
        if graph_type == "SL":
            # use training part of SL data to construct input graph
            graph_data = SL_data.iloc[all_idx[:int(len(all_idx)*training_percent)]]
            graph_data = graph_data[graph_data['label']==True]
            graph_data = graph_data[['gene1','gene2']]
        elif graph_type == "PPI-genetic":
            graph_data = load_graph_data(graph_type)
            # removing edges already in the SL data
            SL_pos = SL_data[SL_data['label'] == True]
            SL_pos_list = sorted([tuple(r) for r in SL_pos[['gene1','gene2']].to_numpy()] + [tuple(r) for r in SL_pos[['gene2','gene1']].to_numpy()])
            graph_list = [tuple(r) for r in graph_data[['gene1','gene2']].to_numpy()]
            left = list(set(graph_list) - set(SL_pos_list))
            graph_data = pd.DataFrame(left, columns=['gene1','gene2'])
        else:
            graph_data = load_graph_data(graph_type)
        
        graph_data_list.append(graph_data)
        
    # merge, filter and mapping
    SL_data, graph_data_list, gene_mapping = merge_and_mapping(SL_data, graph_data_list, SL_genes)
    
    # generate node features
    x = choose_node_attribute(attr, gene_mapping, cell_name, graph_data_list)
    if node2vec_feats_flag:
        node2vec_feats = choose_node_attribute('node2vec', gene_mapping, cell_name, graph_data_list)
        x = np.hstack((x,node2vec_feats))
        x = scale(x)
    if CCLE_feats_flag:
        CCLE_feats = load_CCLE_feats("PCA", ['exp','ess'], gene_mapping, CCLE_hidden_dim)
        x = np.hstack((x,CCLE_feats))
        x = scale(x)
    
    # generate torch data
    data_x = torch.tensor(x, dtype=torch.float)
    
    data_edge_index_list = []
    for graph_data in graph_data_list:
        temp_edge_index = torch.tensor([graph_data['gene1'].values, graph_data['gene2'].values], dtype=torch.long)
        data_edge_index_list.append(temp_edge_index)
        
    data = Data(x=data_x, edge_index_list=data_edge_index_list)
    
    SL_data_train = SL_data.iloc[all_idx[:int(len(all_idx)*training_percent)]]
    SL_data_val = SL_data.iloc[all_idx[int(len(all_idx)*training_percent):int(len(all_idx)*(training_percent+0.1))]]
    SL_data_test = SL_data.iloc[all_idx[int(len(all_idx)*(training_percent+0.1)):]]
    
    # generate out of sample new prediction samples
    if predict_novel_flag:
        all_genes = set(list(gene_mapping.keys()))
        novel_genes = all_genes - set(SL_genes)
        #novel_genes = np.random.choice(list(novel_genes),3000,replace=False)
        novel_gene_idx = [gene_mapping[x] for x in novel_genes]
        novel_gene_pairs = list(itertools.combinations(novel_gene_idx, r=2))
        SL_data_oos = pd.DataFrame(novel_gene_pairs, columns=['gene1','gene2'])
        SL_data_oos = SL_data_oos.sample(frac=0.1, random_state=111)
        SL_data_oos['label'] = False
    else:
        SL_data_oos = None
    
    return data, SL_data_train, SL_data_val, SL_data_test, SL_data_oos, gene_mapping


def generate_torch_edges(df, balanced_sample, duplicate, device):
    df_pos = df[df['label'] == True]
    if balanced_sample == True:
        # balanced sample
        df_neg = df[df['label'] == False].sample(n=df_pos.shape[0])
    else:
        df_neg = df[df['label'] == False]
    
    pos_edge_idx = torch.tensor([df_pos['gene1'].values, df_pos['gene2'].values], dtype=torch.long, device=device)
    neg_edge_idx = torch.tensor([df_neg['gene1'].values, df_neg['gene2'].values], dtype=torch.long, device=device)

    if duplicate == True:
        pos_edge_idx = torch.tensor([np.concatenate((df_pos['gene1'].values, df_pos['gene2'].values)),
                                                     np.concatenate((df_pos['gene2'].values,df_pos['gene1'].values))], dtype=torch.long, device=device)
        neg_edge_idx = torch.tensor([np.concatenate((df_neg['gene1'].values, df_neg['gene2'].values)),
                                                     np.concatenate((df_neg['gene2'].values,df_neg['gene1'].values))], dtype=torch.long, device=device)
    else:
        pos_edge_idx = torch.tensor([df_pos['gene1'].values, df_pos['gene2'].values], dtype=torch.long, device=device)
        neg_edge_idx = torch.tensor([df_neg['gene1'].values, df_neg['gene2'].values], dtype=torch.long, device=device)
    
    return pos_edge_idx, neg_edge_idx


def get_link_labels(pos_edge_index, neg_edge_index, device):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels
    
    
def calculate_coexpression(data_type, rho_thres):
    df = pd.read_csv("../data/CCLE/CCLE_{}.csv".format(data_type), index_col=0)
    df.fillna(0, inplace=True)
    df.columns = list(map(lambda x:x.split(" ")[0], df.columns))
    ind_mapping = dict(zip(range(df.shape[1]),df.columns.values))

    print("Calculating coexpression...")
    rho, pval = stats.spearmanr(df.values)
    
    rho_lower = np.tril(rho >= rho_thres, k=-1)
    ind_keep = list(zip(*np.where(rho_lower==True)))

    # convert index to entrez ids
    ind_keep_df = pd.DataFrame(ind_keep, columns=['gene1','gene2'])
    ind_keep_df['gene1'] = ind_keep_df['gene1'].apply(lambda x:ind_mapping[x])
    ind_keep_df['gene2'] = ind_keep_df['gene2'].apply(lambda x:ind_mapping[x])

    ind_keep_df.to_csv("../data/coexpression_{}_{}.csv".format(data_type,str(rho_thres)), index=False)

    plt.hist(rho.flatten(), bins=1000)
    plt.xlim(-1,1)
    plt.savefig("hist_rho_{}.png".format(data_type))


def ranking_metrics(true_labels, pred_scores, top=0.05):
    sorted_index = np.argsort(-pred_scores)
    top_num = int(top * len(true_labels))
    sorted_true_labels = true_labels[sorted_index[:top_num]]
    acc = float(sorted_true_labels.sum())/float(top_num)
    return acc

def compute_fmax(true_labels, pred_scores):
    pred_scores = np.round(pred_scores, 2)
    true_labels = true_labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    sp_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (pred_scores > threshold).astype(np.int32)
        tp = np.sum(predictions * true_labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(true_labels) - tp
        sn = tp / (1.0 * np.sum(true_labels))
        sp = np.sum((predictions ^ 1) * (true_labels ^ 1))
        sp /= 1.0 * np.sum(true_labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max, p_max, r_max, t_max

def generate_random_negative_samples(pos_samples):
    # randomly generate same amount of negative samples as positive samples
    num = pos_samples.shape[0]
    all_genes = list(set(pos_samples['gene1'].unique()) | set(pos_samples['gene2'].unique()))
    neg_candidates_1 = random.choices(all_genes, k=2*num)
    neg_candidates_2 = random.choices(all_genes, k=2*num)
    
    pos_list = [tuple(r) for r in pos_samples[['gene1','gene2']].to_numpy()] + [tuple(r) for r in pos_samples[['gene2','gene1']].to_numpy()]
    sampled_list = list(zip(neg_candidates_1, neg_candidates_2))
    # remove gene pairs that have positive effects
    remained_list = set(sampled_list) - set(pos_list)
    # remove gene pairs where gene1 = gene2
    remained_list = [x for x in remained_list if x[0] != x[1]]
    
    return random.sample(remained_list, num)
