
import os
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
import networkx as nx
from tqdm import tqdm
import scipy.sparse as ssp
import multiprocessing as mp

import matplotlib.pyplot as plt
from sklearn import metrics

def plot_roc_pr_curves(fold_metrics, k_folds=5):
    """
    رسم نمودارهای ROC و PR برای هر فولد و میانگین ترکیبی.
    
    پارامترها:
        fold_metrics: لیستی از دیکشنری‌های مربوط به نتایج هر فولد.
        k_folds: تعداد فولد‌ها.
    """
    pooled_labels = np.concatenate([fold_metrics[i]['truth'] for i in range(k_folds)])
    pooled_predictions = np.concatenate([fold_metrics[i]['predictions'] for i in range(k_folds)])

    fpr, tpr, _ = metrics.roc_curve(pooled_labels, pooled_predictions)
    precision, recall, _ = metrics.precision_recall_curve(pooled_labels, pooled_predictions)
    avg_auc = metrics.auc(fpr, tpr)
    avg_aupr = metrics.auc(recall, precision)
    
    plt.figure(figsize=(12,6))
    # رسم منحنی‌های ROC برای هر فولد
    plt.subplot(1,2,1)
    for i in range(k_folds):
        plt.plot(fold_metrics[i]['fpr'], fold_metrics[i]['tpr'], 
                 label=f'Fold {i+1} (AUC = {fold_metrics[i]["test_auc"]:.4f})')
    plt.plot(fpr, tpr, color='black', linestyle='--', label=f'Pooled Average (AUC = {avg_auc:.4f})')
    plt.title("ROC Curve Across Folds")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    
    # رسم منحنی‌های PR برای هر فولد
    plt.subplot(1,2,2)
    for i in range(k_folds):
        plt.plot(fold_metrics[i]['recall_curve'], fold_metrics[i]['precision_curve'],
                 label=f'Fold {i+1} (AUPR = {fold_metrics[i]["aupr_score"]:.4f})')
    plt.plot(recall, precision, color='black', linestyle='--', label=f'Pooled Average (AUPR = {avg_aupr:.4f})')
    plt.title("Precision-Recall Curve Across Folds")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

class MyDataset(InMemoryDataset):
    """یک دیتاست InMemory برای ذخیره داده‌های پردازش‌شده PyG"""
    def __init__(self, data_list, root, transform=None, pre_transform=None):
        self.data_list = data_list
        super(MyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass
    
    def process(self):
        data_list = self.data_list
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        del self.data_list

def nx_to_PyGGraph(g, graph_label, node_labels, node_features, max_node_label, class_values):
    """
    تبدیل گراف NetworkX به شی Data متعلق به PyTorch Geometric.
    """
    y = torch.tensor([graph_label], dtype=torch.float)
    if len(g.edges()) == 0:
        i, j = [], []
    else:
        i, j = zip(*g.edges())
    edge_index = torch.LongTensor([list(i) + list(j), list(j) + list(i)])
    x_labels = torch.FloatTensor(one_hot(node_labels, max_node_label + 1))
    if node_features is not None and node_features.shape[1] == x_labels.shape[1]:
        x_features = torch.FloatTensor(node_features)
        x = torch.cat([x_labels, x_features], dim=1)
    else:
        x = x_labels
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros((len(idx), length))
    x[np.arange(len(idx)), idx] = 1.0
    return x

def neighbors(fringe, A, row=True):
    """یافتن همسایگان یک یا چند گره در ماتریس پراکندگی A"""
    res = set()
    for node in fringe:
        if row:
            _, nei, _ = ssp.find(A[node, :])
        else:
            _, nei, _ = ssp.find(A[:, node])
        res = res.union(set(nei))
    return res

def subgraph_extraction_labeling(ind, A, h=1, u_features=None, v_features=None, class_values=None, max_node_label=None):
    """استخراج زیرگراف h-hop به همراه برچسب‌گذاری گره‌ها"""
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])
    
    for dist in range(1, h+1):
        u_fringe = neighbors(u_fringe, A, row=True) - u_visited
        v_fringe = neighbors(v_fringe, A, row=False) - v_visited
        if not u_fringe and not v_fringe:
            break
        u_nodes.extend(list(u_fringe))
        v_nodes.extend(list(v_fringe))
        u_dist.extend([dist]*len(u_fringe))
        v_dist.extend([dist]*len(v_fringe))
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)
    
    subgraph = A[u_nodes, :][:, v_nodes]
    g = nx.Graph()
    g.add_nodes_from(range(len(u_nodes)), bipartite='u')
    g.add_nodes_from(range(len(u_nodes), len(u_nodes)+len(v_nodes)), bipartite='v')
    u, v, r = ssp.find(subgraph)
    r = r.astype(int)
    v = v + len(u_nodes)
    g.add_edges_from(zip(u, v))
    node_labels = [min(d*2, max_node_label) for d in u_dist] + [min(d*2+1, max_node_label) for d in v_dist]
    
    if u_features is not None and v_features is not None:
        u_feat = u_features[u_nodes]
        v_feat = v_features[v_nodes]
        min_cols = min(u_feat.shape[1], v_feat.shape[1])
        u_feat = u_feat[:, :min_cols]
        v_feat = v_feat[:, :min_cols]
        node_features = np.concatenate([u_feat, v_feat], axis=0)
    else:
        node_features = None
    return g, node_labels, node_features

def parallel_worker(g_label, ind, A, h=1, u_features=None, v_features=None, class_values=None, max_node_label=None):
    g, node_labels, node_features = subgraph_extraction_labeling(ind, A, h, u_features, v_features, class_values, max_node_label)
    return g_label, g, node_labels, node_features, ind

def extracting_subgraphs(A, all_indices, all_labels, h=1, u_features=None, v_features=None, max_node_label=None):
    """
    استخراج زیرگراف‌های h-hop برای تمام نمونه‌ها به صورت موازی.
    """
    if max_node_label is None:
        max_node_label = h*2+1
    class_values = np.array([0, 1], dtype=float)
    
    def helper(A, links, g_labels):
        g_list = []
        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap_async(parallel_worker, [
            (g_label, (i, j), A, h, u_features, v_features, class_values, max_node_label)
            for i, j, g_label in zip(links[0], links[1], g_labels)
        ])
        results = results.get()
        pool.close()
        for g_label, g, n_labels, n_features, _ in tqdm(results, desc="Extracting subgraphs"):
            g_list.append(nx_to_PyGGraph(g, g_label, n_labels, n_features, max_node_label, class_values))
        return g_list

    graphs = helper(A, all_indices, all_labels)
    return graphs
