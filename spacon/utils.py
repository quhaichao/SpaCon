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
from torch_geometric.loader import NeighborLoader
import os



def neighbor_sample(NT_graph_data, ST_graph_data, batch_size=64, train_num_neighbors=[20, 10, 10], eval_num_neighbors=[-1], num_workers=4):
   """
    Create neighbor sampling data loaders for neural tracing and spatial transcriptomic data.
   
    This function generates PyTorch Geometric NeighborLoader objects for
    training and evaluation phases. It creates separate loaders for training
    on neural tracing connectivity data and evaluation on both neural tracing
    connectivity and spatial transcriptomic data with appropriate neighbor
    sampling strategies.
    
    Parameters
    ----------
    NT_graph_data : torch_geometric.data.Data
        Graph data constructed from neural tracing connectivity data, used
        for training and connectivity-based evaluation.
    ST_graph_data : torch_geometric.data.Data
        Graph data constructed from spatial positions in spatial transcriptomic
        data, used for spatial evaluation.
    batch_size : int, default=64
        Number of nodes in each batch for sampling.
    train_num_neighbors : list of int, default=[20, 10, 10]
        Number of neighbors to sample at each layer during training.
        Length determines the number of GNN layers.
    eval_num_neighbors : list of int, default=[-1]
        Number of neighbors to sample during evaluation. -1 means all neighbors.
    num_workers : int, default=4
        Number of worker processes for data loading.
    
    Returns
    -------
    train_loader : NeighborLoader
        Data loader for training with neighbor sampling on neural tracing
        connectivity graph data.
    evaluate_loader_con : NeighborLoader
        Data loader for evaluation on neural tracing connectivity graph data.
    evaluate_loader_spa : NeighborLoader
        Data loader for evaluation on spatial transcriptomic graph data
        constructed from spatial positions.
   """
   train_loader = NeighborLoader(NT_graph_data, shuffle=False, num_neighbors=train_num_neighbors, batch_size=batch_size, num_workers=num_workers)
   # evaluate loader
   evaluate_loader_con = NeighborLoader(copy.copy(NT_graph_data), input_nodes=None, shuffle=False, num_neighbors=eval_num_neighbors, batch_size=batch_size, num_workers=num_workers)
   # Add global node index information.
   evaluate_loader_con.data.num_nodes = NT_graph_data.num_nodes
   evaluate_loader_con.data.n_id = torch.arange(NT_graph_data.num_nodes)
   # evaluate
   evaluate_loader_spa = NeighborLoader(copy.copy(ST_graph_data), input_nodes=None, shuffle=False, num_neighbors=eval_num_neighbors, batch_size=batch_size, num_workers=num_workers)
   # Add global node index information.
   evaluate_loader_spa.data.num_nodes = ST_graph_data.num_nodes
   evaluate_loader_spa.data.n_id = torch.arange(ST_graph_data.num_nodes)
   return train_loader, evaluate_loader_con, evaluate_loader_spa



def model_train(num_epoch, lr, weight_decay, model, train_loader, st_adj, model_save_path=None,
               device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
   """
   Train a graph neural network model using dual graph structures.
   
   This function trains the provided model using batched neural tracing
   connectivity data while incorporating spatial transcriptomic adjacency
   information. The model learns to reconstruct input features using both
   connectivity-based and spatial-based graph structures with MSE loss.
   
   Parameters
   ----------
   num_epoch : int
       Number of training epochs.
   lr : float
       Learning rate for the Adam optimizer.
   weight_decay : float
       Weight decay (L2 regularization) coefficient for the optimizer.
   model : torch.nn.Module
       The graph neural network model to be trained. Should return
       (f_con, f_spa, re) where f_con is connectivity features, f_spa is 
       spatial features, and re is the reconstruction.
   train_loader : NeighborLoader
       Data loader providing batched neural tracing connectivity graph data
       for training.
   st_adj : array-like
       Spatial transcriptomic adjacency matrix constructed from spatial
       positions, used to create spatial edges for each batch.
   model_save_path : str or None, default=None
       Directory path to save the trained model parameters. If None,
       model is not saved.
   device : torch.device, default=torch.device('cuda:0' if available else 'cpu')
       Device to run the training on (CPU or GPU).
   
   Returns
   -------
   model : torch.nn.Module
       The trained model with updated parameters.
   """
   optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
   model.train()
   for epoch in range(1, num_epoch+1):
       print(f'epoch:{epoch}|{num_epoch}')
       for batch_NT in tqdm(train_loader):
           optimizer.zero_grad()
           batch_NT = batch_NT.to(device)
           n_id = batch_NT.n_id.to('cpu').detach().numpy()
           st_adj_batch = st_adj[n_id][:, n_id]
           edgeList = np.argwhere(st_adj_batch)
           _, _, re = model(batch_NT.x, batch_NT.edge_index, torch.LongTensor(edgeList.T).to(device))
           loss = F.mse_loss(batch_NT.x, re, reduction="sum") 
           loss.backward()
           torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
           optimizer.step()
   if model_save_path is not None:
       print('\n')
       print(f"Training completed! The model parameters have been saved to {model_save_path+'model_params.pth'}")
       torch.save(model.state_dict(), model_save_path+'model_params.pth')
   else:
       print(f'Training completed!')
   return model



def model_eval(model, adata, NT_graph_data, ST_graph_data, evaluate_loader_con, evaluate_loader_spa, st_adj,
               layer_eval = True, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """
    Evaluate a trained graph neural network model and extract dual features.
    
    This function evaluates the trained model to extract both connectivity-based
    features (from neural tracing data) and spatial features (from spatial
    transcriptomic positions). It supports two evaluation modes: layer-wise
    inference for feature extraction only, or full model evaluation including
    reconstruction results.
    
    Parameters
    ----------
    model : torch.nn.Module
        The trained graph neural network model with con_inference and
        spa_inference methods for layer evaluation, or forward method
        for full evaluation that processes both connectivity and spatial graphs.
    adata : AnnData
        Annotated data object where results will be stored in .obsm and .layers.
    NT_graph_data : torch_geometric.data.Data
        Neural tracing connectivity graph data for connectivity feature extraction.
    ST_graph_data : torch_geometric.data.Data
        Spatial transcriptomic graph data constructed from spatial positions
        for spatial feature extraction.
    evaluate_loader_con : NeighborLoader
        Data loader for neural tracing connectivity graph evaluation.
    evaluate_loader_spa : NeighborLoader
        Data loader for spatial transcriptomic graph evaluation.
    st_adj : array-like
        Spatial transcriptomic adjacency matrix constructed from spatial
        positions, used for creating spatial edges during full model evaluation.
    layer_eval : bool, default=True
        If True, performs layer-wise inference to extract features only.
        If False, performs full model evaluation including reconstruction.
    device : torch.device, default=torch.device('cuda:0' if available else 'cpu')
        Device to run the evaluation on (CPU or GPU).
    
    Returns
    -------
    adata : AnnData
        Updated annotated data object containing:
        - adata.obsm['feature_con']: connectivity-based features from neural tracing
        - adata.obsm['feature_spa']: spatial features from transcriptomic positions
        - adata.layers['exp_reconstructed']: reconstructed expression (if layer_eval=False)
    """
    model.eval()
    if layer_eval == True:
        feature_con = model.con_inference(NT_graph_data.x, evaluate_loader_con, device=device)
        feature_spa = model.spa_inference(ST_graph_data.x, evaluate_loader_spa, device=device)
        adata.obsm['feature_spa'] = feature_spa.to('cpu').detach().numpy()
        adata.obsm['feature_con'] = feature_con.to('cpu').detach().numpy()
    # all data evaluate
    else:
        f_con_list = []
        f_spa_list = []
        re_list = []
        for batch_NT in tqdm(evaluate_loader_con):
            batch_NT = batch_NT.to(device)
            n_id = batch_NT.n_id.to('cpu').detach().numpy()
            st_adj_batch = st_adj[n_id][:, n_id]
            edgeList = np.argwhere(st_adj_batch)
            f_con, f_spa, re = model(batch_NT.x, batch_NT.edge_index, torch.LongTensor(edgeList.T).to(device))

            f_con_list.append(f_con[:batch_NT.batch_size].to('cpu').detach().numpy())
            f_spa_list.append(f_spa[:batch_NT.batch_size].to('cpu').detach().numpy())
            re_list.append(re[:batch_NT.batch_size].to('cpu').detach().numpy())

        f_con = np.concatenate(f_con_list, axis=0)
        f_spa = np.concatenate(f_spa_list, axis=0)
        re = np.concatenate(re_list, axis=0)
        adata.obsm['feature_spa'] =f_spa
        adata.obsm['feature_con'] =f_con
        adata.layers['exp_reconstructed'] = re

    print('The results have been saved in adata.obsm')
    print(adata)
    return adata



def clustering(adata, alpha, adata_save_path, cluster_resolution=0.75):
    """
    Performs clustering on AnnData object using a weighted combination of
    pre-computed connectivity and spatial features, then saves the results.

    Parameters
    ----------
    adata : anndata.AnnData
        The input AnnData object containing 'feature_con' (connectivity features)
        and 'feature_spa' (spatial features) in adata.obsm.
    alpha : float
        A weighting parameter (0 to 1) that controls the contribution of
        connectivity features vs. spatial features to the combined feature set.
        A higher alpha emphasizes connectivity features.
    adata_save_path : str
        The base directory path where the clustering results and AnnData object
        will be saved.
    cluster_resolution : float, optional
        Resolution parameter for the Louvain clustering algorithm.
        Higher values lead to more clusters. Defaults to 0.75.

    Returns
    -------
    adata : anndata.AnnData
        The updated AnnData object with added 'feature_add' in .obsm,
        neighbors graph, UMAP embeddings, and 'clusters' in .obs.
    path : str
        The full path to the directory where the results were saved.
    """
    import copy
    adata = copy.deepcopy(adata)
    from sklearn.preprocessing import StandardScaler
    scaler_standard = StandardScaler()
    f_con = scaler_standard.fit_transform(adata.obsm['feature_con'])
    f_spa = scaler_standard.fit_transform(adata.obsm['feature_spa'])
    f_alp = (np.exp(-4 * alpha) - 1) / (np.exp(-4) - 1)
    f_add = f_alp*f_con + (1-f_alp)*f_spa
    adata.obsm['feature_add'] = f_add

    sc.pp.neighbors(adata, use_rep='feature_add', n_neighbors=40)
    sc.tl.umap(adata)


    path = adata_save_path + f'feature_add_weight{alpha}/Clusters_res{cluster_resolution}/'
    os.makedirs(path, exist_ok=True)

    sc.tl.louvain(adata,
                resolution=cluster_resolution, 
                key_added="clusters")
    sc.pl.umap(adata,color='clusters', show=False)
    plt.tight_layout()
    plt.savefig(path+f'cluster_umap_res{cluster_resolution}.png')
    adata.write_h5ad(path+'adata_cluster.h5ad')

    print(f'The clustering results have been saved in {path}')
    print(adata)

    return adata, path


def plot_all_results(adata, path, figsize, plot_x, plot_y):
    """
    Generates and saves 2D scatter plots of spatial transcriptomics data.
    It creates plots for each tissue section showing all cells colored by cluster,
    and then for each cluster, it generates plots for each section highlighting
    cells belonging to that specific cluster.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing spatial coordinates (`plot_x`, `plot_y` in .obs),
        section information (`section` in .obs), and cluster assignments (`clusters` in .obs).
    path : str
        The base directory path where the generated plots will be saved.
    figsize : tuple
        A tuple (width, height) specifying the size of the matplotlib figures.
    plot_x : str
        The key in `adata.obs` corresponding to the X-axis coordinates for plotting.
    plot_y : str
        The key in `adata.obs` corresponding to the Y-axis coordinates for plotting.

    Returns
    -------
    None
        This function does not return any value; it saves plot images to the specified path.
    """

    for it, label in enumerate(np.unique(adata.obs['section'])):
        temp_adata = adata[adata.obs['section'] == label]
        fig = plt.figure(figsize=figsize)
        plt.scatter(temp_adata.obs[plot_x].values, temp_adata.obs[plot_y].values, c=temp_adata.obs['clusters'].astype('int').values, cmap='Spectral_r', s=10)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.savefig(path + '/section_'+str(label)+'.png')
        plt.close() 

    for cluster_i in tqdm(range(np.unique(np.array(adata.obs['clusters'])).shape[0])):  # adata.shape[0]
        # build the spot_i anndata
        cluster_i_adata = adata[adata.obs['clusters']==str(cluster_i)]
        cluster_i_adata.obs.index = cluster_i_adata.obs.index.astype(str)
        # build the save path
        fig_eachclass_path = path+f'/each_cluster_result/cluster_num_{str(cluster_i)}'
        if os.path.exists(fig_eachclass_path) == False:
            os.makedirs(fig_eachclass_path)  # build all the folders
        # plot same class as the spoti on every section
        for label in set(cluster_i_adata.obs['section'].values):
            adata_section = adata[adata.obs['section'] == label]
            adata_class = cluster_i_adata[cluster_i_adata.obs['section'] == label]

            fig = plt.figure(figsize=figsize)
            # plot
            plt.scatter(adata_section.obs[plot_x].values, adata_section.obs[plot_y].values, c='#D3D3D3', s=10)
            plt.scatter(adata_class.obs[plot_x].values, adata_class.obs[plot_y].values, c='#FF6347', s=10)
            plt.gca().invert_yaxis()
            # plt.legend(loc='upper right', prop={'size':5})
            plt.savefig(fig_eachclass_path + '/section_'+str(label)+'.png')
            # plt.show()
            plt.close()




def result_plot_3D(adata, highlight_section, cluster_color):
    """
    Generates an interactive 3D plot of spatial transcriptomics data,
    displaying cell clusters across different tissue sections with an option to highlight a specific section.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing spatial coordinates in `adata.obs`
        (specifically 'x_section_mean', 'z', 'y') and cluster assignments
        in `adata.obs['clusters']`. It should also contain 'section' in `adata.obs`.
    highlight_section : str or int
        The identifier of the section to be highlighted and offset in the 3D plot.
    cluster_color : dict
        A dictionary mapping cluster labels (integers) to their corresponding colors.

    Returns
    -------
    None
        Displays a matplotlib 3D plot.
    """
    adata.obs['clu'] = adata.obs['clusters'].astype(int)
    adata.obs['x_section_mean'] = adata.obs['x_section_mean'].round(3)

    cells = adata.obs[['x_section_mean', 'z', 'y', 'clu']].values
    cells[:, 1] = -cells[:, 1]
    cells[:, 2] = -cells[:, 2]

    spacing_multiplier = 3
    # Data preprocessing
    original_slices = np.unique(cells[:, 0])  # Get original slice positions
    sorted_slices = np.sort(original_slices)    # Sort slice positions

    # Generate new slice positions (maintain relative order, magnify spacing)
    new_slice_positions = sorted_slices[0] + np.arange(len(sorted_slices)) * spacing_multiplier

    # Create position mapping dictionary (original coordinates -> mapped coordinates)
    position_map = {old: new for old, new in zip(sorted_slices, new_slice_positions)}

    # Apply new coordinates to data (only change X coordinate here)
    cells[:, 0] = np.vectorize(position_map.get)(cells[:, 0])

    # Create reverse mapping dictionary (mapped coordinates -> original coordinates)
    reverse_position_map = {new: old for new, old in position_map.items()}

    # Define slices to highlight (based on original coordinates) and offsets
    highlight_section_x = adata[adata.obs['section'] == highlight_section].obs['x_section_mean'].unique()
    highlight_slices = [highlight_section_x]      # Original X coordinates of highlighted slices
    offset_y = 40                              # Y-direction offset
    offset_z = 40                              # Z-direction offset

    # ================= Visualization Phase =================
    # Create 3D canvas
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Visualization parameter configuration
    viz_params = {
        'plane_alpha': 0.01,       # Plane transparency
        'edge_width': 0.3,         # Edge line width
        'point_size': 4,          # Cell point size
        'point_alpha': 0.7,        # Cell point transparency
        'line_alpha': 0.2          # Edge transparency
    }

    # Automatically calculate coordinate ranges (all planes share the same range, ignoring offset effects)
    coord_ranges = [
        (cells[:, 0].min()-0.5, cells[:, 0].max()+0.5),  # X range
        (cells[:, 1].min()-5, cells[:, 1].max()+5),      # Y range
        (cells[:, 2].min()-5, cells[:, 2].max()+5)       # Z range
    ]

    # Create plane template (un-offset version)
    Y0, Z0 = np.meshgrid(
        np.linspace(*coord_ranges[1], 2),
        np.linspace(*coord_ranges[2], 2)
    )

    # Draw slices layer by layer
    for x in np.unique(cells[:, 0]):
        # Get the original X coordinate
        original_x = reverse_position_map.get(x, x)
        
        # Determine if the current slice should be highlighted (based on original X coordinate)
        if original_x in highlight_slices:
            add_y, add_z = offset_y, offset_z
            x_plot = x  # Highlighted slice uses original X coordinate
        else:
            add_y, add_z = 0, 0
            x_plot = x          # Normal slice uses mapped X coordinate
        
        # Generate plane data, X coordinate uniformly uses x_plot
        X_plane = np.full_like(Y0, x_plot)
        # For highlighted slices, the plane is offset in Y and Z coordinates; no offset for other slices
        Y_plane = Y0 + add_y
        Z_plane = Z0 + add_z
        
        # Draw transparent plane
        ax.plot_surface(X_plane, Y_plane, Z_plane,
                        color='lightgray',
                        alpha=viz_params['plane_alpha'],
                        linewidth=0)
        
        # Draw plane borders, border X coordinate also uses x_plot
        border_lines = [
            ([x_plot, x_plot],
            [coord_ranges[1][0] + add_y, coord_ranges[1][1] + add_y],
            [coord_ranges[2][0] + add_z, coord_ranges[2][0] + add_z]),  # Bottom edge
            ([x_plot, x_plot],
            [coord_ranges[1][0] + add_y, coord_ranges[1][1] + add_y],
            [coord_ranges[2][1] + add_z, coord_ranges[2][1] + add_z]),  # Top edge
            ([x_plot, x_plot],
            [coord_ranges[1][0] + add_y, coord_ranges[1][0] + add_y],
            [coord_ranges[2][0] + add_z, coord_ranges[2][1] + add_z]),  # Left edge
            ([x_plot, x_plot],
            [coord_ranges[1][1] + add_y, coord_ranges[1][1] + add_y],
            [coord_ranges[2][0] + add_z, coord_ranges[2][1] + add_z])   # Right edge
        ]
        
        for line in border_lines:
            ax.plot(*line,
                    color='k',
                    linewidth=viz_params['edge_width'],
                    alpha=viz_params['line_alpha'])
        
        # Extract cells for the current layer
        mask = cells[:, 0] == x
        layer_data = cells[mask]
        
        # Convert cluster labels to integers
        class_labels = layer_data[:, 3].astype(int)
        
        # Generate color list
        colors = [cluster_color[cls] for cls in class_labels]
        
        # Draw cell points, for highlighted slices, X coordinate is also replaced with original coordinate, Y and Z coordinates are offset
        ax.scatter(
            np.full(len(layer_data), x_plot),    # Use x_plot instead of original mapped value
            layer_data[:, 1] + add_y,            # Y coordinate offset
            layer_data[:, 2] + add_z,            # Z coordinate offset
            c=colors,                            # Cluster color
            s=viz_params['point_size'],
            alpha=viz_params['point_alpha'],
            edgecolor='w',
            linewidth=0.1
        )

    # ================= Graphic Embellishments =================
    # Hide coordinate system
    ax.set_axis_off()
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_visible(False)

    # Calculate actual data range (including offset)
    all_x = cells[:, 0]
    all_y = cells[:, 1]
    all_z = cells[:, 2]

    # Set tight axis limits
    buffer = 2  # Buffer area
    ax.set_xlim(all_x.min() - buffer, all_x.max() + buffer)
    ax.set_ylim(all_y.min() - buffer, all_y.max() + offset_y + buffer)
    ax.set_zlim(all_z.min() - buffer, all_z.max() + offset_z + buffer)

    # Set viewing angle
    ax.view_init(elev=5, azim=125)

    # Adjust box aspect to match actual data range
    x_range = all_x.max() - all_x.min() + 2 * buffer
    y_range = all_y.max() - all_y.min() + offset_y + 2 * buffer  
    z_range = all_z.max() - all_z.min() + offset_z + 2 * buffer

    ax.set_box_aspect([x_range, y_range, z_range])

    # Adjust plot layout to reduce whitespace
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    plt.tight_layout()
    plt.show()






def find_neighbors(coor, model, cutoff):
    """
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
    """
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



def build_spatial_graph(adata, rad_cutoff=None, k_cutoff=None, model='Radius', sec_x='x', sec_y='y', is_3d=True,
                           rad_cutoff_Zaxis=None, key_section=None, section_order=None):
    """
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
    G = sp.lil_matrix((adata.n_obs, adata.n_obs))

    if is_3d:
        # Loop through each section to find neighbors within the section
        for section in tqdm(np.unique(adata.obs[key_section])):
            temp_adata = adata[adata.obs[key_section] == section]
            coor = pd.DataFrame({'x': temp_adata.obs[sec_x], 'y': temp_adata.obs[sec_y]})

            # Find neighbors for the current section
            indices = find_neighbors(coor, model, rad_cutoff if model == 'Radius' else k_cutoff)

            # Update the adjacency matrix for the current section
            update_adjacency_matrix(G, temp_adata, indices)

        # Loop through adjacent sections to find inter-section neighbors
        for it in tqdm(range(len(section_order) - 1)):
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
    G = G.tocoo()

    # Convert to PyTorch Geometric Data
    edge_index = torch.LongTensor(np.array([G.row, G.col]))  # Convert COO format to edge_index
    x = torch.FloatTensor(adata.X.todense() if sp.issparse(adata.X) else adata.X)

    data = Data(edge_index=edge_index, x=x)

    return data, G.tocsr()


import anndata
def create_spatial_graph_from_anndata(
    adata: anndata.AnnData,
    n_neighbors: int = 10,
    spatial_key: str = 'spatial',
    gene_expr_layer: str = None,
    use_highly_variable: bool = False
) -> tuple:
    """
    Creates a spatial graph from AnnData with gene expression data as node features.

    Parameters:
        adata: AnnData object containing gene expression data and spatial coordinates.
        n_neighbors: Number of neighbors to connect for each cell.
        spatial_key: Key for spatial coordinates in adata.obsm, defaults to 'spatial'.
        gene_expr_layer: Gene expression layer to use; if None, adata.X will be used.
        use_highly_variable: Whether to use only highly variable genes as features.

    Returns:
        graph_data: PyG Data object.
        adj_matrix: Sparse adjacency matrix.
    """
    # 1. Get spatial coordinates
    if spatial_key in adata.obsm:
        spatial_coords = adata.obsm[spatial_key]
    else:
        raise ValueError(f"Spatial coordinates '{spatial_key}' not found in adata.obsm")

    # 2. Get gene expression as node features
    if gene_expr_layer is not None:
        if gene_expr_layer in adata.layers:
            gene_expr = adata.layers[gene_expr_layer]
        else:
            raise ValueError(f"Layer '{gene_expr_layer}' not found in adata.layers")
    else:
        gene_expr = adata.X

    # Convert to dense matrix if it's a scipy sparse matrix
    if sp.issparse(gene_expr):
        gene_expr = gene_expr.toarray()

    # Use highly variable genes if required
    if use_highly_variable:
        if 'highly_variable' in adata.var:
            gene_expr = gene_expr[:, adata.var['highly_variable']]
        else:
            print("Warning: Highly variable gene information not found, using all genes.")

    # 3. Use KNN to find nearest neighbors based on spatial distance
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)

    # 4. Build edge index
    edge_index = []
    for i in range(len(indices)):
        # Skip the first neighbor (itself)
        for j in indices[i][1:]:
            edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # 5. Convert node features to PyTorch tensor
    node_features = torch.tensor(gene_expr, dtype=torch.float)

    # 6. Create PyG Data object
    graph_data = Data(x=node_features, edge_index=edge_index)

    # 7. Build adjacency matrix
    n_cells = len(adata)
    adj_rows, adj_cols = edge_index
    adj_values = torch.ones(edge_index.shape[1])
    adj_matrix = sp.coo_matrix(
        (adj_values.numpy(), (adj_rows.numpy(), adj_cols.numpy())),
        shape=(n_cells, n_cells)
    )

    return graph_data, adj_matrix.tocsr()



def build_connection_graph(adata, adj, threshold=0.001):
   """
   Build a connection graph from adjacency matrix for PyTorch Geometric.
   
   This function processes an adjacency matrix by applying a threshold filter,
   binarizing the connections, adding self-loops, and converting the result
   into a PyTorch Geometric Data object suitable for graph neural networks.
   
   Parameters
   ----------
   adata : AnnData
       Annotated data object containing gene expression data in .X attribute.
   adj : array-like
       Adjacency matrix representing connections between cells/nodes.
   threshold : float, default=0.001
       Minimum value threshold for connections. Values below this threshold
       are set to zero.
   
   Returns
   -------
   data : torch_geometric.data.Data
       PyTorch Geometric Data object containing:
       - edge_index: tensor of shape (2, num_edges) with edge connections
       - x: tensor of shape (num_nodes, num_features) with node features
   """
   adj[adj < threshold] = 0
   adj[adj > 0] = 1
   for i in range(adj.shape[0]):
       adj[i, i] = 1
   # Convert to PyTorch Geometric Data
   edgeList = np.argwhere(adj)
   edge_index = torch.LongTensor(edgeList.T)
   x = torch.FloatTensor(adata.X.todense() if sp.issparse(adata.X) else adata.X)
   data = Data(edge_index=edge_index, x=x)
   return data



import scipy
def refine_label(adata, radius=50, key='clusters'):
   """
   Refine cell type labels based on spatial neighborhood majority voting.
   
   This function refines cell type annotations by assigning each cell the most
   common label among its spatial neighbors. For each cell, it finds the nearest
   neighbors within the specified radius and assigns the majority label from
   those neighbors.
   
   Parameters
   ----------
   adata : AnnData
       Annotated data object containing spatial coordinates in .obsm['spatial']
       and cell type labels in .obs[key].
   radius : int, default=50
       Number of nearest neighbors to consider for label refinement.
   key : str, default='clusters'
       Key in adata.obs containing the original cell type labels to be refined.
   
   Returns
   -------
   new_type : list of str
       Refined cell type labels based on spatial neighborhood majority voting.
       Each element corresponds to the refined label for the cell at the same
       index position.
   """
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