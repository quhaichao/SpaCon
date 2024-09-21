import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
from Cluster_Network.model_pyg import GATE_PyG_3Layers_Encapsulation
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer

import torch
from torch_geometric.data import Data
from tqdm import tqdm
from scipy.sparse import issparse
import scanpy as sc
import torch.nn.functional as F
import copy
import os


'''train model and eval'''
def model_train_and_eval(adata, 
                         graph_data, 
                         num_epoch, 
                         lr, 
                         weight_decay, 
                         hidden_dims, 
                         device, 
                         kwargs_train, 
                         kwargs_test, 
                         batch_eval):
    loader_NT = NeighborLoader(graph_data, shuffle=False, persistent_workers=False, **kwargs_train)
    subgraph_loader = NeighborLoader(copy.copy(graph_data), input_nodes=None, shuffle=False, persistent_workers=False, **kwargs_test)   # test_loader: num_neighbors=[-1]
    # Add global node index information.
    subgraph_loader.data.num_nodes = graph_data.num_nodes
    subgraph_loader.data.n_id = torch.arange(graph_data.num_nodes)  # !!!!!!!!!!!!!!!

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = GATE_PyG_3Layers_Encapsulation(hidden_dims=hidden_dims).to(device)
    loss_list = []
    # for training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 启用 batch normalization 和 dropout
    model.train()
    print('Start training the model:')
    for epoch in tqdm(range(1, num_epoch+1)):
        loss_batch = 0
        for batch_NT in loader_NT:
            # model.train()
            optimizer.zero_grad()
            batch_NT = batch_NT.to(device)
            # print('batch_NT.x:', batch_NT.x)
            # print('batch_NT.edge_index:', batch_NT.edge_index)
            z, out = model(batch_NT.x, batch_NT.edge_index)
            loss = F.mse_loss(batch_NT.x, out, reduction="sum") #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss_batch += loss.to('cpu').detach().numpy()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        loss_list.append(loss_batch)
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', linestyle='-')
    plt.grid(True)
    plt.show()

    # for evaluating
    model.eval()
    print('Start evaluating the model:')
    # batch eval or no batch
    if batch_eval == True:
        feature = model.inference(graph_data.x, subgraph_loader, device=device)
        # add the feature to the adata
        GATE_feature = feature.to('cpu').detach().numpy()
        # adata.uns['GATE_loss'] = loss_list
        adata.obsm['GATE_feature'] = GATE_feature
    # all data evaluate
    else:
        GATE_feature, reconstructed = model(graph_data.x, graph_data.edge_index)
        # add the feature to the adata
        # adata.uns['GATE_loss'] = loss_list
        adata.obsm['GATE_feature'] = GATE_feature.to('cpu').detach().numpy()

    return adata


def model_explain(adata,
                  model_path,
                  NT_adj,
                  device,
                  data_loder,
                  top_spot_num=1,
                  num_neighbors=[10, 5, 5],
                  explain_method='CaptumExplainer_Saliency'
                  ):
    # load the pretrained parameter
    hidden_dims = [adata.X.shape[1]] + [256, 128, 32]
    model = GATE_PyG_3Layers_Encapsulation(hidden_dims=hidden_dims).to(device)
    model.load_state_dict(torch.load(f'{model_path}/model_params.pth'))

    explain_results_save_path = f'{model_path}{explain_method}_input_cluster_node_{data_loder}/'
    GATE_feature = pd.DataFrame(adata.obsm['GATE_feature'])
    GATE_feature['clusters'] = adata.obs['clusters'].to_list()
    for c in tqdm(GATE_feature['clusters'].unique()):
        cluster_GATE_feature = GATE_feature[GATE_feature['clusters'] == c]
        # 计算所有spot的平均编码
        cluster_average_feature = np.mean(cluster_GATE_feature.iloc[:, :-1], axis=0)
        # 计算每个spot与平均编码的欧氏距离
        distances = np.linalg.norm(cluster_GATE_feature.iloc[:, :-1] - cluster_average_feature, axis=1)
        # 找出与平均编码最接近的spot的index
        closest_spot_index = np.argsort(distances)
        # 取前n个索引
        smallest_indices_top = closest_spot_index[:top_spot_num]
        smallest_indices_top =  sorted(smallest_indices_top)
        # print(smallest_indices_top)
        # the node index to explain, this node is most similar to the cluster, which represent the cluster
        node_index = [cluster_GATE_feature.index[i] for i in smallest_indices_top]
        # print(node_index)

        adata_cluster = adata[adata.obs['clusters'] == c]
        cluster_index = adata_cluster.obs['spot_num'].tolist()
        cluster_adj = NT_adj[cluster_index][:, cluster_index]
        for i in range(cluster_adj.shape[0]):
            cluster_adj[i, i] = 1
        edgeList = np.argwhere(cluster_adj>0.001)
        data = Data(edge_index=torch.LongTensor(edgeList.T), x=torch.FloatTensor(adata_cluster.X.A))
        if data_loder=='NeighborLoader':
            explain_dataloder = NeighborLoader(data, shuffle=False, num_neighbors=num_neighbors, batch_size=1, num_workers=0)
        elif data_loder=='LinkNeighborLoader':
            explain_dataloder = LinkNeighborLoader(data, shuffle=False, num_neighbors=num_neighbors, batch_size=1, num_workers=4, persistent_workers=False)
        adata_cluster.obs['index'] = [i for i in range(adata_cluster.shape[0])]
        # 3.explain the index nod
        explanation = []
        batch_i = 0
        for batch in explain_dataloder:
            if batch_i in smallest_indices_top:
                if explain_method.split('_')[0]=='CaptumExplainer':
                    explainer = Explainer(
                                        model=model.encoder,
                                        algorithm=CaptumExplainer(explain_method.split('_')[1]),
                                        explanation_type='model',
                                        node_mask_type='attributes',
                                        edge_mask_type='object',
                                        model_config=dict(
                                            mode='multiclass_classification',
                                            task_level='node',
                                            return_type='log_probs',
                                        ),
                                    )
                elif explain_method=='GNNExplainer':
                    explainer = Explainer(
                                        model=model.encoder,
                                        algorithm=GNNExplainer(epochs=200),
                                        explanation_type='model',
                                        node_mask_type='attributes',
                                        edge_mask_type='object',
                                        model_config=dict(
                                            mode='multiclass_classification',
                                            task_level='node',
                                            return_type='log_probs',
                                        ),
                                    )
                
                explanation_temp = explainer(batch.x.to(device), batch.edge_index.to(device), index=0)  # 计算给定输入和目标的GNN的解释
                explanation.append(explanation_temp)
            if batch_i==smallest_indices_top[-1]:
                break
            batch_i += 1
        # build the spot explain save path
        if -1 in num_neighbors:
            gnnexplain_path = explain_results_save_path + f'/represent_spot_gnnexplainer_result/loader_-1_top{top_spot_num}spot/Class_{c}/'
        else:
            gnnexplain_path = explain_results_save_path + f'/represent_spot_gnnexplainer_result/loader_{num_neighbors[0]}_{num_neighbors[1]}_{num_neighbors[2]}_top{top_spot_num}spot/Class_{c}/'
        os.makedirs(gnnexplain_path, exist_ok=True)  # build all the folders

        node_importance = pd.DataFrame(index= adata.var_names)
        for r, s in zip(explanation, node_index):
            node_mask = r.get('node_mask')
            feat_importance = node_mask.sum(dim=0).cpu().detach().numpy()
            node_importance[f'{s}'] = feat_importance

        node_importance['mean'] = node_importance.mean(axis=1)
        node_importance = node_importance.sort_values(by='mean', ascending=False)

        # topk
        df_top30 = node_importance.head(30)['mean']
        ax = df_top30.plot(kind='barh', figsize=(10, 7), ylabel='Feature label', xlim=[0, float(df_top30.values.max()) + 0.3], legend=False)
        plt.gca().invert_yaxis()
        ax.bar_label(container=ax.containers[0], label_type='edge')
        plt.savefig(gnnexplain_path + f'cluster{c}feature_importance.png')

        # node_importance_png_feature_name_path = gnnexplain_path + f'/Spot_{node_index}_feature_importance_name.png'
        # node_importance = visualize_feature_importance(explanation, path=node_importance_png_feature_name_path, top_k=30, adata_var_names=adata.var_names)
        node_importance.to_csv(gnnexplain_path + '/feature_importance.csv')

def top_gene_merge(input_adata,
                   adata_exp,
                    # pred,
                    target_domain,
                    # start_gene,
                    gene_rep,
                    filter_cell=False,
                    mean_diff=0,
                    early_stop=True,
                    max_iter=5,
                    use_raw=False):
    # meta_name=''
    adata=input_adata.copy()
    # if gene_rep != None:
    #     adata = adata[:, gene_rep]
    if filter_cell:
        adata.obs["meta"]=adata_exp[:,adata.var.index==gene_rep[0]]
        meta_name = gene_rep[0]
    else:
        adata.obs["meta"]=0
        meta_name=''
    # adata.obs["pred"]=pred
    # num_non_target=adata.shape[0]
    for gene in gene_rep:
        adata.obs[gene]=adata_exp[:,adata.var.index==gene]
        if filter_cell:
            tmp=adata[((adata.obs["meta"]>np.mean(adata.obs[adata.obs["clusters"]==target_domain]["meta"]))|(adata.obs["clusters"]==target_domain))]
            adata_target = tmp[tmp.obs['clusters'] == target_domain]
            adata_other = tmp[tmp.obs['clusters'] != target_domain]
        else:
            adata_target = adata[adata.obs['clusters'] == target_domain]
            adata_other = adata[adata.obs['clusters'] != target_domain]
        target_avg_exp = adata_target.obs[gene].mean()
        other_avg_exp = adata_other.obs[gene].mean()
        if target_avg_exp > other_avg_exp:
            meta_name_cur=meta_name+"+"+gene
            adata.obs["meta_cur"]=adata.obs["meta"] + adata.obs[gene]
        else:
            meta_name_cur=meta_name+"-"+gene
            adata.obs["meta_cur"]=adata.obs["meta"] - adata.obs[gene]

        adata.obs["meta_cur"]=adata.obs["meta_cur"]-np.min(adata.obs["meta_cur"])
        #Select cells
        # tmp=adata[((adata.obs["meta"]>np.mean(adata.obs[adata.obs["pred"]==target_domain]["meta"]))|(adata.obs["pred"]==target_domain))]
        # tmp.obs["target"]=((tmp.obs["pred"]==target_domain)*1).astype('category').copy()
        # if (len(set(tmp.obs["target"]))<2) or (np.min(tmp.obs["target"].value_counts().values)<5):
        #     print("Meta gene is: ", meta_name)
        #     return meta_name, adata.obs["meta"].tolist()
        #DE
        # sc.tl.rank_genes_groups(tmp, groupby="target",reference="rest", n_genes=1,method='wilcoxon')
        # adj_g=tmp.uns['rank_genes_groups']["names"][0][0]
        # add_g=tmp.uns['rank_genes_groups']["names"][0][1]
        # meta_name_cur=meta_name+"+"+add_g+"-"+adj_g
        # print("Add gene: ", add_g)
        # print("Minus gene: ", adj_g)
        # #Meta gene
        # adata.obs[add_g]=adata.X.A[:,adata.var.index==add_g]
        # adata.obs[adj_g]=adata.X.A[:,adata.var.index==adj_g]
        # adata.obs["meta_cur"]=(adata.obs["meta"]+adata.obs[add_g]-adata.obs[adj_g])
        # adata.obs["meta_cur"]=adata.obs["meta_cur"]-np.min(adata.obs["meta_cur"])
        # mean_diff_cur=np.mean(adata.obs["meta_cur"][adata.obs["pred"]==target_domain])-np.mean(adata.obs["meta_cur"][adata.obs["pred"]!=target_domain])
        # num_non_target_cur=np.sum(tmp.obs["target"]==0)
        # if (early_stop==False) | ((num_non_target>=num_non_target_cur) & (mean_diff<=mean_diff_cur)):
        #     num_non_target=num_non_target_cur
        #     mean_diff=mean_diff_cur
        #     print("Absolute mean change:", mean_diff)
        #     print("Number of non-target spots reduced to:",num_non_target)
        # else:
        #     print("Stopped!", "Previous Number of non-target spots",num_non_target, num_non_target_cur, mean_diff,mean_diff_cur)
        #     print("Previous Number of non-target spots",num_non_target, num_non_target_cur, mean_diff,mean_diff_cur)
        #     print("Previous Number of non-target spots",num_non_target)
        #     print("Current Number of non-target spots",num_non_target_cur)
        #     print("Absolute mean change", mean_diff)
        #     print("===========================================================================")
        #     print("Meta gene: ", meta_name)
        #     print("===========================================================================")
        #     return meta_name, adata.obs["meta"].tolist()
        meta_name=meta_name_cur
        adata.obs["meta"]=adata.obs["meta_cur"]
        # print("===========================================================================")
        # print("Meta gene is: ", meta_name)
        # print("===========================================================================")
    return meta_name, adata.obs["meta"].tolist()



'''transfer the ST anndata to the pyg data'''
def Transfer_pytorch_Data(adata):
    # copy the adata to process
    G_df = adata.uns['Spatial_Net'].copy()
    # trans the spot's name to num
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    # trans the distance matrix(G_df) to the adjacent matrix(G)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])
    print('The proportion of the ST adj matrix is: ',np.nonzero(G)[0].shape[0]/(G.shape[0]**2))

    # trans the adjacent matrix(G) to PYG data
    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data, G

'''transfer NTdata and STdata to the pyg data'''
def Transfer_pytorch_NT_Data(adata, copy=False):
    # copy the adata to process
    if copy:
        G = adata.uns['Spatial_Net_NT_data'].copy()
    else:
        G = adata.uns['Spatial_Net_NT_data']
    # tran the adj matrix to 0-1 matrix
    G[G > 0] = 1
    # G = G + I
    # G = G + np.eye(G.shape[0])
    for i in range(G.shape[0]):
        G[i, i] = 1
    # tran numpy to scipy sparse
    # G = sp.csr_matrix(G)
    # print('The proportion of the NT adj matrix is: ',np.nonzero(G)[0].shape[0]/(G.shape[0]**2))

    # trans the adjacent matrix(G) to PYG data
    edgeList = np.argwhere(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(edgeList.T), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(edgeList.T), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data

'''Batch the anndata (2D)'''
def Batch_Data(adata, num_batch_x, num_batch_y, spatial_key=['X', 'Y'], plot_Stats=False):
    # extract the .obs['X', 'Y'] data
    Sp_df = adata.obs.loc[:, spatial_key].copy()  # not change original data
    # trans the sp_df to the numpy
    Sp_df = np.array(Sp_df)
    # 找到一组数的分位数值，如四分位数等(具体什么位置根据自己定义)
    batch_x_coor = [np.percentile(Sp_df[:, 0], (1/num_batch_x)*x*100) for x in range(num_batch_x+1)]   # [794.0, 2316.966666666666, 3376.8999999999996, 4903.1]
    batch_y_coor = [np.percentile(Sp_df[:, 1], (1/num_batch_y)*x*100) for x in range(num_batch_y+1)]   # [1066.9, 3366.4, 5451.5]

    Batch_list = []
    for it_x in range(num_batch_x):
        for it_y in range(num_batch_y):
            # set the threshold value
            min_x = batch_x_coor[it_x]
            max_x = batch_x_coor[it_x+1]
            min_y = batch_y_coor[it_y]
            max_y = batch_y_coor[it_y+1]
            # set the temp_adata
            temp_adata = adata.copy()
            # select the adata between the x threshold value
            temp_adata = temp_adata[temp_adata.obs[spatial_key[0]].map(lambda x: min_x <= x <= max_x)]
            # select the adata between the y threshold value
            temp_adata = temp_adata[temp_adata.obs[spatial_key[1]].map(lambda y: min_y <= y <= max_y)]
            Batch_list.append(temp_adata)
    if plot_Stats:
        f, ax = plt.subplots(figsize=(1, 3))
        plot_df = pd.DataFrame([x.shape[0] for x in Batch_list], columns=['#spot/batch'])
        sns.boxplot(y='#spot/batch', data=plot_df, ax=ax)
        sns.stripplot(y='#spot/batch', data=plot_df, ax=ax, color='red', size=5)
    return Batch_list


'''Batch the anndata (3D)'''
def Batch_Data_3D(adata, num_batch_x, num_batch_y, num_batch_z, spatial_key=['X', 'Y', 'Z'], plot_Stats=False):
    # extract the .obs['X', 'Y'] data
    Sp_df = adata.obs.loc[:, spatial_key].copy()  # not change original data
    # trans the sp_df to the numpy
    Sp_df = np.array(Sp_df)
    # 找到一组数的分位数值，如四分位数等(具体什么位置根据自己定义)
    batch_x_coor = [np.percentile(Sp_df[:, 0], (1/num_batch_x)*x*100) for x in range(num_batch_x+1)]   # [794.0, 2316.966666666666, 3376.8999999999996, 4903.1]
    batch_y_coor = [np.percentile(Sp_df[:, 1], (1/num_batch_y)*x*100) for x in range(num_batch_y+1)]   # [1066.9, 3366.4, 5451.5]
    batch_z_coor = [np.percentile(Sp_df[:, 2], (1/num_batch_z)*x*100) for x in range(num_batch_z+1)]

    Batch_list = []
    for it_x in range(num_batch_x):
        for it_y in range(num_batch_y):
            for it_z in range(num_batch_z):
                # set the threshold value
                min_x = batch_x_coor[it_x]
                max_x = batch_x_coor[it_x+1]
                min_y = batch_y_coor[it_y]
                max_y = batch_y_coor[it_y+1]
                min_z = batch_z_coor[it_z]
                max_z = batch_z_coor[it_z+1]
                # set the temp_adata
                temp_adata = adata.copy()
                # select the adata between the x threshold value
                temp_adata = temp_adata[temp_adata.obs[spatial_key[0]].map(lambda x: min_x <= x <= max_x)]
                # select the adata between the y threshold value
                temp_adata = temp_adata[temp_adata.obs[spatial_key[1]].map(lambda y: min_y <= y <= max_y)]
                # select the adata between the z threshold value
                temp_adata = temp_adata[temp_adata.obs[spatial_key[2]].map(lambda z: min_z <= z <= max_z)]
                Batch_list.append(temp_adata)
    if plot_Stats:
        f, ax = plt.subplots(figsize=(1, 3))
        plot_df = pd.DataFrame([x.shape[0] for x in Batch_list], columns=['#spot/batch'])
        sns.boxplot(y='#spot/batch', data=plot_df, ax=ax)
        sns.stripplot(y='#spot/batch', data=plot_df, ax=ax, color='red', size=5)
    return Batch_list




def Batch_data_function(X, A, n):
    '''
    input
        :param X: gene data
        :param A: adjacent matrix
        :param n: batch number
    return:
        :param train_X_list: batched gene data
        :param train_adj_list: batched adjacent matrix
        (data type: numpy)
    '''
    # get the divide number
    batch_X = [np.percentile(X.shape, (1/n)*x*100) for x in range(1,n+1)]
    # build the HashMap
    # number--[index_start, index_end]
    batch_num_2_index = {}
    for i in range(len(batch_X)):
        if i==0:
            batch_num_2_index[i] = [0, int(batch_X[i])-1]
        else:
            batch_num_2_index[i] = [int(batch_X[i-1]), int(batch_X[i])-1]

    # generate the input data
    # i,j are the selected data numbers
    train_X_list = []
    train_adj_list = []
    for i in tqdm(range(0, n-1)):
        for j in range(i+1, n):
            # index
            index_i = np.array(range(batch_num_2_index[i][0], batch_num_2_index[i][1] + 1))
            index_j = np.array(range(batch_num_2_index[j][0], batch_num_2_index[j][1] + 1))
            index = np.append(index_i,index_j)
            # gene data
            train_X = X[index,:]
            # adj matrix
            train_adj = A[index,:]
            train_adj = train_adj[:,index]
            # append
            train_X_list.append(train_X)
            train_adj_list.append(train_adj)

    return train_X_list, train_adj_list


'''SVG(calculate adj matrix)'''
import numba
@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1,t2):
    sum = 0
    for i in range(t1.shape[0]):
        sum += (t1[i]-t2[i])**2
    return np.sqrt(sum)

@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
    n = X.shape[0]
    adj = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j] = euclid_dist(X[i], X[j]) # X[i]: (x,y,z)
    return adj

def calculate_adj_matrix(x, y, z,x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
    #x,y,x_pixel, y_pixel are lists
    if histology:
        assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
        assert (len(x)==len(x_pixel)) & (len(y)==len(y_pixel))
        print("Calculateing adj matrix using histology image...")
        #beta to control the range of neighbourhood when calculate grey vale for one spot
        #alpha to control the color scale
        beta_half=round(beta/2)
        g=[]
        for i in range(len(x_pixel)):
            max_x=image.shape[0]
            max_y=image.shape[1]
            nbs=image[max(0,x_pixel[i]-beta_half):min(max_x,x_pixel[i]+beta_half+1),max(0,y_pixel[i]-beta_half):min(max_y,y_pixel[i]+beta_half+1)]
            g.append(np.mean(np.mean(nbs,axis=0),axis=0))
        c0, c1, c2=[], [], []
        for i in g:
            c0.append(i[0])
            c1.append(i[1])
            c2.append(i[2])
        c0=np.array(c0)
        c1=np.array(c1)
        c2=np.array(c2)
        print("Var of c0,c1,c2 = ", np.var(c0),np.var(c1),np.var(c2))
        c3=(c0*np.var(c0)+c1*np.var(c1)+c2*np.var(c2))/(np.var(c0)+np.var(c1)+np.var(c2))
        c4=(c3-np.mean(c3))/np.std(c3)
        z_scale=np.max([np.std(x), np.std(y)])*alpha
        z=c4*z_scale
        z=z.tolist()
        print("Var of x,y,z = ", np.var(x),np.var(y),np.var(z))
        X=np.array([x, y, z]).T.astype(np.float32)
    else:
        print("Calculateing adj matrix using xyz only...")
        X = np.array([x, y, z]).T.astype(np.float32)
    return pairwise_distance(X)


'''searche radius of neighborhood'''
def count_nbr(target_cluster,cell_id, x, y, z, pred, radius):
    # adj_2d=calculate_adj_matrix(x=x,y=y,z=z, histology=False)
    # cluster_num = dict()
    df = {'cell_id': cell_id, 'x': x, "y":y, 'z':z, "pred":pred}
    df = pd.DataFrame(data=df)
    df.index=df['cell_id']
    target_df=df[df["pred"]==target_cluster]
    # row_index=0
    num_nbr=[]
    for index, row in target_df.iterrows():
        x = row["x"]
        y = row["y"]
        z = row["z"]
        tmp_nbr = df[((df["x"]-x)**2+(df["y"]-y)**2+(df["z"]-z)**2) <= (radius**2)]
        num_nbr.append(tmp_nbr.shape[0])
    return np.mean(num_nbr)

def search_radius(target_cluster,cell_id, x, y, z, pred, start, end, num_min=8, num_max=15,  max_run=100):
    run=0
    num_low = count_nbr(target_cluster,cell_id, x, y, z, pred, start)
    num_high = count_nbr(target_cluster,cell_id, x, y, z, pred, end)
    if num_min<=num_low<=num_max:
        print("recommended radius = ", str(start))
        return start
    elif num_min<=num_high<=num_max:
        print("recommended radius = ", str(end))
        return end
    elif num_low>num_max:
        print("Try smaller start.")
        return None
    elif num_high<num_min:
        print("Try bigger end.")
        return None
    while (num_low<num_min) and (num_high>num_min):
        run+=1
        print("Run "+str(run)+": radius ["+str(start)+", "+str(end)+"], num_nbr ["+str(num_low)+", "+str(num_high)+"]")
        if run > max_run:
            print("Exact radius not found, closest values are:\n"+"radius="+str(start)+": "+"num_nbr="+str(num_low)+"\nradius="+str(end)+": "+"num_nbr="+str(num_high))
            return None
        mid = (start+end)/2
        num_mid = count_nbr(target_cluster,cell_id, x, y, z, pred, mid)
        if num_min <= num_mid <= num_max:
            print("recommended radius = ", str(mid), "num_nbr="+str(num_mid))
            return mid
        if num_mid < num_min:
            start = mid
            num_low = num_mid
        elif num_mid > num_max:
            end = mid
            num_high = num_mid

'''find neighbor clusters'''
def find_neighbor_clusters(target_cluster,cell_id, x, y, z,pred,radius, ratio=1/2):
    cluster_num = dict()
    for i in pred:
        cluster_num[i] = cluster_num.get(i, 0) + 1
    df = {'cell_id': cell_id, 'x': x, "y":y, "z":z, "pred":pred}
    df = pd.DataFrame(data=df)
    df.index=df['cell_id']
    target_df=df[df["pred"]==target_cluster]
    nbr_num={}
    row_index=0
    num_nbr=[]
    for index, row in target_df.iterrows():
        x = row["x"]
        y = row["y"]
        tmp_nbr = df[((df["x"]-x)**2+(df["y"]-y)**2+(df["z"]-z)**2)<=(radius**2)]
        #tmp_nbr=df[(df["x"]<x+radius) & (df["x"]>x-radius) & (df["y"]<y+radius) & (df["y"]>y-radius)]
        num_nbr.append(tmp_nbr.shape[0])
        for p in tmp_nbr["pred"]:
            nbr_num[p]=nbr_num.get(p,0)+1
    del nbr_num[target_cluster]
    nbr_num=[(k, v)  for k, v in nbr_num.items() if v>(ratio*cluster_num[k])]
    nbr_num.sort(key=lambda x: -x[1])
    # print("radius=", radius, "average number of neighbors for each spot is", np.mean(num_nbr))
    # print(" Cluster",target_cluster, "has neighbors:")
    # for t in nbr_num:
    #     print("Dmain ", t[0], ": ",t[1])
    ret=[t[0] for t in nbr_num]
    if len(ret)==0:
        print("No neighbor domain found, try bigger radius or smaller ratio.")
    else:
        return ret


'''get the SVG DE information'''
def rank_genes_groups(input_adata, target_cluster,nbr_list, label_col, adj_nbr=True, log=False):
    # 1.select the spots in the target domain and neighbor domain
    if adj_nbr:
        nbr_list=nbr_list+[target_cluster]
        adata=input_adata[input_adata.obs[label_col].isin(nbr_list)]
    else:
        adata=input_adata.copy()
    adata.var_names_make_unique()
    # 2.spots in target(1), spots in neghbor(0)
    adata.obs["target"]=((adata.obs[label_col]==target_cluster)*1).astype('category')
    # 3.calculate the values
    sc.tl.rank_genes_groups(adata, groupby="target",reference="rest", n_genes=adata.shape[1],method='wilcoxon')
    # 4.calculate the index value
    pvals_adj=[i[0] for i in adata.uns['rank_genes_groups']["pvals_adj"]]
    genes=[i[1] for i in adata.uns['rank_genes_groups']["names"]]
    if issparse(adata.X):
        obs_tidy=pd.DataFrame(adata.X.A)
    else:
        obs_tidy=pd.DataFrame(adata.X)
    # change the  dataframe index and columns
    obs_tidy.index=adata.obs["target"].tolist()
    obs_tidy.columns=adata.var.index.tolist()
    obs_tidy=obs_tidy.loc[:,genes]  # change the order by gene order
    # 1. compute mean value
    mean_obs = obs_tidy.groupby(level=0).mean()  # groupby函数主要的作用是进行数据的分组以及分组后地组内运算
    # 2. compute fraction of cells having value >0
    obs_bool = obs_tidy.astype(bool)  # (X: 0->False, 1->True)
    fraction_obs = obs_bool.groupby(level=0).sum() / obs_bool.groupby(level=0).count()
    # compute fold change(FC:差异倍数).
    if log: #The adata already logged
        fold_change=np.exp((mean_obs.loc[1] - mean_obs.loc[0]).values)
    else:
        fold_change = (mean_obs.loc[1] / (mean_obs.loc[0]+ 1e-9)).values
    df = {'genes': genes, 'in_group_fraction': fraction_obs.loc[1].tolist(), "out_group_fraction":fraction_obs.loc[0].tolist(),"in_out_group_ratio":(fraction_obs.loc[1]/fraction_obs.loc[0]).tolist(),"in_group_mean_exp": mean_obs.loc[1].tolist(), "out_group_mean_exp": mean_obs.loc[0].tolist(),"fold_change":fold_change.tolist(), "pvals_adj":pvals_adj}
    df = pd.DataFrame(data=df)
    return df


def find_meta_gene(input_adata,
                   pred,
                   target_domain,
                   start_gene,
                   mean_diff=0,
                   early_stop=True,
                   max_iter=5,
                   use_raw=False):
    # 1.build the adata which has the meata gene expression
    meta_name = start_gene
    adata = input_adata.copy()
    adata.obs["meta"] = adata.X[:, adata.var.index == start_gene]  # get all spots' start_gene expression
    adata.obs["pred"] = pred
    num_non_target = adata.shape[0]  # 3639
    # 2.Start iteration
    for i in range(max_iter):
        # 2.1.Select spots(target domain spots + no_target domain spots have high expression value in gene meta)
        tmp = adata[((adata.obs["meta"] > np.mean(adata.obs[adata.obs["pred"] == target_domain]["meta"])) | (
                    adata.obs["pred"] == target_domain))]
        # target spots -> 1, not target spots -> 0
        tmp.obs["target"] = ((tmp.obs["pred"] == target_domain) * 1).astype(
            'category').copy()  # target spots -> 1, not target spots -> 0
        # check if the class num < 2 and min spots number in class < 5
        if (len(set(tmp.obs["target"])) < 2) or (np.min(tmp.obs["target"].value_counts().values) < 5):
            print("Meta gene is: ", meta_name)
            return meta_name, adata.obs["meta"].tolist()
        # 2.2.DE
        sc.tl.rank_genes_groups(tmp, groupby="target", reference="rest", n_genes=1, method='wilcoxon')
        # select the DE result gene
        adj_g = tmp.uns['rank_genes_groups']["names"][0][0]  # Genes that are highly expressed in the target domain
        add_g = tmp.uns['rank_genes_groups']["names"][0][1]  # Genes that are highly expressed in the other domains
        meta_name_cur = meta_name + "+" + add_g + "-" + adj_g
        print("Add gene: ", add_g)
        print("Minus gene: ", adj_g)
        # update the Meta gene's expression
        adata.obs[add_g] = adata.X[:, adata.var.index == add_g]
        adata.obs[adj_g] = adata.X[:, adata.var.index == adj_g]
        adata.obs["meta_cur"] = (adata.obs["meta"] + adata.obs[add_g] - adata.obs[adj_g])
        adata.obs["meta_cur"] = adata.obs["meta_cur"] - np.min(adata.obs["meta_cur"])
        # 2 stop values(smaller spots number, bigger expression)
        # spots expression's mean in target - spots expression's mean not in target
        mean_diff_cur = np.mean(adata.obs["meta_cur"][adata.obs["pred"] == target_domain]) - np.mean(
            adata.obs["meta_cur"][adata.obs["pred"] != target_domain])
        # number
        num_non_target_cur = np.sum(tmp.obs["target"] == 0)
        if (early_stop == False) | ((num_non_target >= num_non_target_cur) & (mean_diff <= mean_diff_cur)):
            num_non_target = num_non_target_cur  # non target spots number
            mean_diff = mean_diff_cur  # spots expression's mean in target - spots expression's mean not in target
            print("Absolute mean change:", mean_diff)
            print("Number of non-target spots reduced to:", num_non_target)
        else:
            print("Stopped!")
            print("Previous Number of non-target spots", num_non_target)
            print("Current Number of non-target spots", num_non_target_cur)
            print("Previous absolute mean change", mean_diff)
            print("Current absolute mean change", mean_diff_cur)
            print("===========================================================================")
            print("Meta gene: ", meta_name)
            print("===========================================================================")
            return meta_name, adata.obs["meta"].tolist()
        meta_name = meta_name_cur
        adata.obs["meta"] = adata.obs["meta_cur"]
        print("===========================================================================")
        print("Meta gene is: ", meta_name)
        print("===========================================================================")

    return meta_name, adata.obs["meta"].tolist()

"""
function: n spots were randomly selected for the clustering results
input:  adata_clusters_obs_names: obs name
        adata_clusters: cluster result
        n: generate number
"""
import random
def generate_n_spot_for_each_clase(adata_clusters_obs_names, adata_clusters, n):
    cluster_spotname = {}
    for i in tqdm(range(len(adata_clusters_obs_names))):
        cluster = adata_clusters[i]
        spotname = i
        if cluster in cluster_spotname:
            cluster_spotname[cluster].append(spotname)
        else:
            cluster_spotname[cluster] = [spotname]

    cluster_representative_spotname = {}
    for key, values in cluster_spotname.items():
        for i in range(n):
            value = int(values[random.randint(0, len(values)-1)])
            if key in cluster_representative_spotname:
                cluster_representative_spotname[key].append(value)
            else:
                cluster_representative_spotname[key] = [value]

    return cluster_spotname, cluster_representative_spotname

# define the visual function
from typing import Optional
def visualize_feature_importance(
        explanation,
        path: Optional[str] = None,
        top_k: Optional[int] = None,
        adata_var_names = None,
):
    r"""Creates a bar plot of the node features importance by summing up
    :attr:`self.node_mask` across all nodes.

    Args:
        path (str, optional): The path to where the plot is saved.
            If set to :obj:`None`, will visualize the plot on-the-fly.
            (default: :obj:`None`)
        feat_labels (List[str], optional): Optional labels for features.
            (default :obj:`None`)
        top_k (int, optional): Top k features to plot. If :obj:`None`
            plots all features. (default: :obj:`None`)
    """

    node_mask = explanation.get('node_mask')
    feat_importance = node_mask.sum(dim=0).cpu().numpy()
    # feat_labels = range(feat_importance.shape[0])
    feat_labels = adata_var_names

    df = pd.DataFrame({'feat_importance': feat_importance}, index=feat_labels)
    df = df.sort_values("feat_importance", ascending=False)
    df = df.round(decimals=3)

    # topk
    df_top30 = df.head(top_k)
    title = f"Feature importance for top {len(df_top30)} features"

    # plot
    ax = df_top30.plot(
        kind='barh',
        figsize=(10, 7),
        title=title,
        ylabel='Feature label',
        xlim=[0, float(feat_importance.max()) + 0.3],
        legend=False,
    )
    plt.gca().invert_yaxis()
    ax.bar_label(container=ax.containers[0], label_type='edge')

    if path is not None:
        plt.savefig(path)

    # plt.show()
    # plt.close()
    return df


import scipy
def refine_label(adata, radius=50, key='mclust'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    # distance = ot.dist(position, position, metric='euclidean')
    distance = scipy.spatial.distance.cdist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type