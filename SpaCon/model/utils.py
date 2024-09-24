import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
from Cluster_Network.model_pyg import GATE_PyG_3Layers_Encapsulation
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
import sklearn.neighbors
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



def Cal_Spatial_Net(adata, section_x='x', section_y='y', rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    ##############################################
    # 1.put the coordinate in the variable coor
    coor = pd.DataFrame(pd.concat([adata.obs[section_x],adata.obs[section_y]],axis=1))
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    #############################################
    # 2.calculate the KNN_list by the coor
    if model == 'Radius':
        # build the RNN model
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        # 查找位于以测试样本为中心的这个半径圈上的数据. 返回:训练样本离半径圈的距离与对应训练样本中的下标
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    #############################################
    # 3.trans the KNN_list to dataframe, filter and change the cell's name
    KNN_df = pd.concat(KNN_list)
    # set the column's name
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    # build the spatial_net
    Spatial_Net = KNN_df.copy()  # 6319751*3
    # filter the spatial_net
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]  # 6235102*3
    # save the spot's index spatialNet
    Spatial_Net_spots_index = Spatial_Net.copy()

    # trans the index to spot's name
    # build the index map coor.num--coor.index
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))  # 0--'295_467'
    # change the cell name
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net
    adata.uns['Spatial_Net_spots_index'] = Spatial_Net_spots_index





def Cal_Spatial_Net_3D(adata, rad_cutoff_2D, rad_cutoff_Zaxis,section_x='z', section_y='y',
                       key_section='Section_id', section_order=None, verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff_2D
        radius cutoff for 2D SNN construction.
    rad_cutoff_Zaxis
        radius cutoff for 2D SNN construction for consturcting SNNs between adjacent sections.
    key_section
        The columns names of section_ID in adata.obs.
    section_order
        The order of sections. The SNNs between adjacent sections are constructed according to this order.

    Returns
    -------
    The 3D spatial networks are saved in adata.uns['Spatial_Net'].
    """
    # build the empty dataframe
    adata.uns['Spatial_Net_2D'] = pd.DataFrame()
    adata.uns['Spatial_Net_Zaxis'] = pd.DataFrame()
    # get the section num
    num_section = np.unique(adata.obs[key_section]).shape[0]  # num_section=7
    if verbose:
        print('Radius used for 2D SNN:', rad_cutoff_2D)
        print('Radius used for SNN between sections:', rad_cutoff_Zaxis)

    # for every section
    for temp_section in tqdm(np.unique(adata.obs[key_section])):
        if verbose:
            print('\n------Calculating 2D SNN of section ', temp_section)
        # add the current section data to the temp_adata
        temp_adata = adata[adata.obs[key_section] == temp_section,]
        # calculate the current section 2D spatial_net
        Cal_Spatial_Net(temp_adata, section_x=section_x, section_y=section_y, rad_cutoff=rad_cutoff_2D, verbose=False)
        temp_adata.uns['Spatial_Net']['SNN'] = temp_section
        if verbose:
            print('This graph contains %d edges, %d cells.' %
                  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
            print('%.4f neighbors per cell on average.' %
                  (temp_adata.uns['Spatial_Net'].shape[0] / temp_adata.n_obs))
        # add the  temp_adata.uns['Spatial_Net']: ([22974 rows x 4 columns])  to the  adata.uns['Spatial_Net_2D']
        adata.uns['Spatial_Net_2D'] = pd.concat([adata.uns['Spatial_Net_2D'], temp_adata.uns['Spatial_Net']])

    # for every section to calculate the adjacent sections spatialNet
    for it in tqdm(range(num_section - 1)):
        # current section name and next section name
        section_1 = section_order[it]
        section_2 = section_order[it + 1]
        if verbose:
            print('\n------Calculating SNN between adjacent section %s and %s.' %
                  (section_1, section_2))
        # build the Z_Net name
        Z_Net_ID = section_1 + '-' + section_2  # 'Puck_180531_13-Puck_180531_16'
        # add the data in section_1/2 to the temp_adata
        temp_adata = adata[adata.obs[key_section].isin([section_1, section_2]), ]
        # calculate the adjacent section spots spatial net
        Cal_Spatial_Net(temp_adata, section_x=section_x, section_y=section_y, rad_cutoff=rad_cutoff_Zaxis, verbose=False)
        # build the pairs of spot--section
        spot_section_trans = dict(
            zip(temp_adata.obs.index, temp_adata.obs[key_section]))  # 'AAAAACAACCAAT-13': 'Puck_180531_13'
        # add the spots(section) to the temp_adata.uns['Spatial_Net']['Section_id_1']
        temp_adata.uns['Spatial_Net']['Section_id_1'] = temp_adata.uns['Spatial_Net']['Cell1'].map(
            spot_section_trans)
        temp_adata.uns['Spatial_Net']['Section_id_2'] = temp_adata.uns['Spatial_Net']['Cell2'].map(
            spot_section_trans)
        # if temp_adata.cell1 in the samethe value is true or false
        used_edge = temp_adata.uns['Spatial_Net'].apply(lambda x: x['Section_id_1'] != x['Section_id_2'],
                                                        axis=1)  # (780002,1)
        # select the adata.uns['Spatial_Net'] which the two cells are not in the same section
        temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[used_edge,]
        temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[:, ['Cell1', 'Cell2',
                                                                              'Distance']]  # [34482 rows x 5 columns]  Cell1, Cell2, Distance, Section_id_1, Section_id_2
        temp_adata.uns['Spatial_Net']['SNN'] = Z_Net_ID
        if verbose:
            print('This graph contains %d edges, %d cells.' % (
            temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
            print(
                '%.4f neighbors per cell on average.' % (temp_adata.uns['Spatial_Net'].shape[0] / temp_adata.n_obs))
        # add the  temp_adata.uns['Spatial_Net']: ([22974 rows x 4 columns])  to the  adata.uns['Spatial_Net_Zaxis']
        adata.uns['Spatial_Net_Zaxis'] = pd.concat([adata.uns['Spatial_Net_Zaxis'], temp_adata.uns['Spatial_Net']])

    # end the for:
    # Cat the Spatial_Net_2D and Spatial_Net_Zaxis
    adata.uns['Spatial_Net'] = pd.concat([adata.uns['Spatial_Net_2D'], adata.uns['Spatial_Net_Zaxis']])
    if verbose:
        print('3D SNN contains %d edges, %d cells.' %
              (adata.uns['Spatial_Net'].shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %
              (adata.uns['Spatial_Net'].shape[0] / adata.n_obs))


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