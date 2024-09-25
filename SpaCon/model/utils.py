import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import scanpy as sc
import torch.nn.functional as F
import copy




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



def find_neighbors(coor, model, cutoff):
    """\
    Find neighbors based on the specified model and cutoff.

    Parameters
    ----------
    coor : DataFrame
        Coordinates of cells.
    model : str
        The model used for neighbor finding.
    cutoff : float or int
        The radius or number of nearest neighbors.

    Returns
    -------
    indices : array
        Indices of the neighbors found.
    """
    if model == 'Radius':
        nbrs = NearestNeighbors(radius=cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor)
    elif model == 'KNN':
        nbrs = NearestNeighbors(n_neighbors=cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
    return indices


def update_adjacency_matrix(G, temp_adata, indices):
    """\
    Update the adjacency matrix based on neighbors found.

    Parameters
    ----------
    G : sparse matrix
        The adjacency matrix to be updated.
    temp_adata : AnnData
        The subset of AnnData for the current section.
    indices : array
        Indices of neighbors.
    """
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            if neighbor != i:  # Ignore self-loops
                G[temp_adata.obs.index[i], temp_adata.obs.index[neighbor]] = 1


def build_spatial_graph(adata, rad_cutoff=None, k_cutoff=None, model='Radius', sec_x='x', sec_y='y', is_3d=False,
                           rad_cutoff_Zaxis=None, key_section=None, section_order=None):
    """\
    Construct the spatial neighbor networks and convert them to PyTorch Geometric format.

    Parameters
    ----------
    adata : AnnData
        AnnData object from the scanpy package.
    rad_cutoff : float, optional
        Radius cutoff for the 'Radius' model.
    k_cutoff : int, optional
        Number of nearest neighbors for the 'KNN' model.
    model : str, {'Radius', 'KNN'}
        The network construction model.
    is_3d : bool, optional
        If True, construct a 3D spatial network.
    rad_cutoff_Zaxis : float, optional
        Radius cutoff for constructing connections between adjacent sections in 3D.
    key_section : str, optional
        Column name of section ID in adata.obs.
    section_order : list, optional
        The order of sections for constructing inter-section connections.

    Returns
    -------
    data : Data
        The spatial network as a PyTorch Geometric Data object.
    """
    
    # Set the obs index to natural order (integer index)
    adata.obs.index = range(adata.n_obs)

    # Create an empty adjacency matrix
    G = sp.coo_matrix((np.zeros((adata.n_obs, adata.n_obs))), dtype=int)

    if is_3d:
        # Loop through each section to find neighbors within the section
        for section in np.unique(adata.obs[key_section]):
            temp_adata = adata[adata.obs[key_section] == section]
            coor = pd.DataFrame({'x': temp_adata.obs[sec_x], 'y': temp_adata.obs[sec_y]})

            # Find neighbors for the current section
            indices = find_neighbors(coor, model, rad_cutoff if model == 'Radius' else k_cutoff)

            # Update the adjacency matrix for the current section
            update_adjacency_matrix(G, temp_adata, indices)

        # Loop through adjacent sections to find inter-section neighbors
        for it in range(len(section_order) - 1):
            section_1 = section_order[it]
            section_2 = section_order[it + 1]
            temp_adata = adata[adata.obs[key_section].isin([section_1, section_2])]
            coor = pd.DataFrame({'x': temp_adata.obs[sec_x], 'y': temp_adata.obs[sec_y]})

            # Find neighbors between adjacent sections
            indices = find_neighbors(coor, model, rad_cutoff_Zaxis if model == 'Radius' else k_cutoff)

            # Update the adjacency matrix for inter-section neighbors
            for i, neighbors in enumerate(indices):
                for neighbor in neighbors:
                    if temp_adata.obs[key_section].iloc[i] != temp_adata.obs[key_section].iloc[neighbor]:
                        G[temp_adata.obs.index[i], temp_adata.obs.index[neighbor]] = 1

    else:  # For 2D
        coor = pd.DataFrame({'x': adata.obs[sec_x], 'y': adata.obs[sec_y]})

        # Find neighbors for the entire dataset
        indices = find_neighbors(coor, model, rad_cutoff if model == 'Radius' else k_cutoff)

        # Update the adjacency matrix for 2D
        update_adjacency_matrix(G, adata, indices)

    # Add self-loops
    G = G + sp.eye(G.shape[0])  # Add self-loops

    # Convert to PyTorch Geometric Data
    edge_index = torch.LongTensor(np.array([G.row, G.col]))  # Convert COO format to edge_index
    x = torch.FloatTensor(adata.X.todense() if sp.issparse(adata.X) else adata.X)

    data = Data(edge_index=edge_index, x=x)

    return data, G




'''transfer NTdata and STdata to the pyg data'''
def Transfer_pytorch_NT_Data(adata, copy=False):
    # copy the adata to process
    if copy:
        G = adata.uns['Spatial_Net'].copy()
    else:
        G = adata.uns['Spatial_Net']
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