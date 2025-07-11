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



# Neighbor-based subgraph sampling
def neighbor_sample(NT_graph_data, ST_graph_data, batch_size=64, train_num_neighbors=[20, 10, 10], eval_num_neighbors=[-1], num_workers=4):

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
    # for training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(1, num_epoch+1):
        print(f'epoch:{epoch}|{num_epoch}')
        for batch_NT in tqdm(train_loader):
            # model.train()
            optimizer.zero_grad()
            batch_NT = batch_NT.to(device)
            n_id = batch_NT.n_id.to('cpu').detach().numpy()
            st_adj_batch = st_adj[n_id][:, n_id]
            edgeList = np.argwhere(st_adj_batch)
            f_con, f_spa, re = model(batch_NT.x, batch_NT.edge_index, torch.LongTensor(edgeList.T).to(device))
            loss = F.mse_loss(batch_NT.x, re, reduction="sum") 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            # break

    if model_save_path is not None:
        print(f"Training completed! The model parameters have been saved to {model_save_path+'/model_params.pth'}")
        torch.save(model.state_dict(), model_save_path+'/model_params.pth')
    else:
        print(f'Training completed!')

    return model



def model_eval(model, adata, NT_graph_data, ST_graph_data, evaluate_loader_con, evaluate_loader_spa, st_adj,
               layer_eval = True,
               device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    
    # model evaluate
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



def clustering(adata, alpha, adata_save_path, figsize, plot_x, plot_y, cluster_resolution=0.75, 
               plot_all_cluster_results=False):
    from sklearn.preprocessing import StandardScaler
    scaler_standard = StandardScaler()
    f_con = scaler_standard.fit_transform(adata.obsm['feature_con'])
    f_spa = scaler_standard.fit_transform(adata.obsm['feature_spa'])
    f_alp = (np.exp(-4 * alpha) - 1) / (np.exp(-4) - 1)
    f_add = f_alp*f_con + (1-f_alp)*f_spa
    adata.obsm['feature_add'] = f_add

    sc.pp.neighbors(adata, use_rep='feature_add')
    sc.tl.umap(adata)


    path = adata_save_path + f'/feature_add_weight{alpha}/Clusters_res{cluster_resolution}/'
    os.makedirs(path, exist_ok=True)

    sc.tl.louvain(adata,
                resolution=cluster_resolution,   # default=1  resolution = k*num (k>0)
                key_added="clusters")
    sc.pl.umap(adata,color='clusters', show=False)
    plt.tight_layout()
    plt.savefig(path+f'cluster_umap_res{cluster_resolution}.png')
    # sc.pl.umap(adata,color='clusters', show=True)
    adata.write_h5ad(path+'/adata_cluster.h5ad')

    print(f'The clustering results have been saved in {path}')
    print(adata)



    if plot_all_cluster_results:
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

    return adata



def result_plot_3D(adata, highlight_section, cluster_color):
    adata.obs['clu'] = adata.obs['clusters'].astype(int)
    adata.obs['x_section_mean'] = adata.obs['x_section_mean'].round(3)

    cells = adata.obs[['x_section_mean', 'z', 'y', 'clu']].values
    cells[:, 1] = -cells[:, 1]
    cells[:, 2] = -cells[:, 2]

    spacing_multiplier = 3
    # 预处理数据
    original_slices = np.unique(cells[:, 0])  # 获取原始切片位置
    sorted_slices = np.sort(original_slices)    # 排序切片位置

    # 生成新切片位置（保持相对顺序，放大间距）
    new_slice_positions = sorted_slices[0] + np.arange(len(sorted_slices)) * spacing_multiplier

    # 创建位置映射字典（原始坐标 -> 映射后的坐标）
    position_map = {old: new for old, new in zip(sorted_slices, new_slice_positions)}

    # 应用新坐标到数据（这里只改变X坐标）
    cells[:, 0] = np.vectorize(position_map.get)(cells[:, 0])

    # 创建反向映射字典（映射后的坐标 -> 原始坐标）
    reverse_position_map = {new: old for old, new in position_map.items()}

    # 定义需要高亮显示的切片（基于原始坐标）以及偏移量
    highlight_section_x = adata[adata.obs['section'] == highlight_section].obs['x_section_mean'].unique()
    highlight_slices = [highlight_section_x]      # 高亮切片原始X坐标
    offset_y = 40                     # Y方向偏移量
    offset_z = 40                     # Z方向偏移量

    # ================= 可视化阶段 =================
    # 创建3D画布
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 可视化参数配置
    viz_params = {
        'plane_alpha': 0.01,     # 平面透明度
        'edge_width': 0.3,       # 边框线宽
        'point_size': 4,        # 细胞点大小
        'point_alpha': 0.7,      # 细胞点透明度
        'line_alpha': 0.2        # 边框透明度
    }

    # 自动计算坐标范围（所有平面共用相同范围，不考虑偏移影响）
    coord_ranges = [
        (cells[:, 0].min()-0.5, cells[:, 0].max()+0.5),  # X范围
        (cells[:, 1].min()-5, cells[:, 1].max()+5),        # Y范围
        (cells[:, 2].min()-5, cells[:, 2].max()+5)         # Z范围
    ]

    # 创建平面模板（未偏移版本）
    Y0, Z0 = np.meshgrid(
        np.linspace(*coord_ranges[1], 2),
        np.linspace(*coord_ranges[2], 2)
    )

    # 逐层绘制切片
    for x in np.unique(cells[:, 0]):
        # 获取原始的X坐标
        original_x = reverse_position_map.get(x, x)
        
        # 判断是否高亮显示当前切片（基于原始X坐标）
        if original_x in highlight_slices:
            add_y, add_z = offset_y, offset_z
            x_plot = x  # 高亮切片使用原始X坐标
        else:
            add_y, add_z = 0, 0
            x_plot = x           # 普通切片使用映射后的X坐标
        
        # 生成平面数据，X 坐标统一使用 x_plot
        X_plane = np.full_like(Y0, x_plot)
        # 对高亮切片，平面在 Y、Z 坐标上偏移；其余切片不偏移
        Y_plane = Y0 + add_y
        Z_plane = Z0 + add_z
        
        # 绘制透明平面
        ax.plot_surface(X_plane, Y_plane, Z_plane,
                        color='lightgray',
                        alpha=viz_params['plane_alpha'],
                        linewidth=0)
        
        # 绘制平面边框，边框X坐标同样使用 x_plot
        border_lines = [
            ([x_plot, x_plot],
            [coord_ranges[1][0] + add_y, coord_ranges[1][1] + add_y],
            [coord_ranges[2][0] + add_z, coord_ranges[2][0] + add_z]),  # 底边
            ([x_plot, x_plot],
            [coord_ranges[1][0] + add_y, coord_ranges[1][1] + add_y],
            [coord_ranges[2][1] + add_z, coord_ranges[2][1] + add_z]),  # 顶边
            ([x_plot, x_plot],
            [coord_ranges[1][0] + add_y, coord_ranges[1][0] + add_y],
            [coord_ranges[2][0] + add_z, coord_ranges[2][1] + add_z]),  # 左边
            ([x_plot, x_plot],
            [coord_ranges[1][1] + add_y, coord_ranges[1][1] + add_y],
            [coord_ranges[2][0] + add_z, coord_ranges[2][1] + add_z])   # 右边
        ]
        
        for line in border_lines:
            ax.plot(*line,
                    color='k',
                    linewidth=viz_params['edge_width'],
                    alpha=viz_params['line_alpha'])
        
        # 提取当前层细胞
        mask = cells[:, 0] == x
        layer_data = cells[mask]
        
        # 转换类别标签为整数
        class_labels = layer_data[:, 3].astype(int)
        
        # 生成颜色列表
        colors = [cluster_color[cls] for cls in class_labels]
        
        # 绘制细胞点，对于高亮切片，X坐标同样替换为原始坐标，Y、Z坐标偏移
        ax.scatter(
            np.full(len(layer_data), x_plot),    # 使用x_plot代替原映射值
            layer_data[:, 1] + add_y,              # Y坐标偏移
            layer_data[:, 2] + add_z,              # Z坐标偏移
            c=colors,                            # 类别颜色
            s=viz_params['point_size'],
            alpha=viz_params['point_alpha'],
            edgecolor='w',
            linewidth=0.1
        )

    # ================= 图形修饰 =================
    # 隐藏坐标系统
    ax.set_axis_off()
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_visible(False)

    # 计算实际数据范围（包括偏移）
    all_x = cells[:, 0]
    all_y = cells[:, 1]
    all_z = cells[:, 2]

    # 设置紧凑的坐标轴范围
    buffer = 2  # 缓冲区域
    ax.set_xlim(all_x.min() - buffer, all_x.max() + buffer)
    ax.set_ylim(all_y.min() - buffer, all_y.max() + offset_y + buffer)
    ax.set_zlim(all_z.min() - buffer, all_z.max() + offset_z + buffer)

    # 设置观察角度
    ax.view_init(elev=5, azim=125)

    # 调整box aspect以匹配实际数据范围
    x_range = all_x.max() - all_x.min() + 2 * buffer
    y_range = all_y.max() - all_y.min() + offset_y + 2 * buffer  
    z_range = all_z.max() - all_z.min() + offset_z + 2 * buffer

    ax.set_box_aspect([x_range, y_range, z_range])

    # 调整图形布局以减少空白
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    plt.tight_layout()
    plt.show()






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



def build_spatial_graph(adata, rad_cutoff=None, k_cutoff=None, model='Radius', sec_x='x', sec_y='y', is_3d=True,
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
    从AnnData创建基于空间位置的图，节点特征为基因表达数据
    
    参数:
        adata: AnnData对象，包含基因表达数据和空间坐标
        n_neighbors: 每个细胞连接的邻居数量
        spatial_key: 存储空间坐标的key，默认为'spatial'
        gene_expr_layer: 使用的基因表达层，如果为None则使用adata.X
        use_highly_variable: 是否仅使用高变基因作为特征
        
    返回:
        graph_data: PyG的Data对象
        adj_matrix: 稀疏邻接矩阵
    """
    # 1. 获取空间坐标
    if spatial_key in adata.obsm:
        spatial_coords = adata.obsm[spatial_key]
    else:
        raise ValueError(f"找不到空间坐标 '{spatial_key}' 在 adata.obsm 中")
    
    # 2. 获取基因表达作为节点特征
    if gene_expr_layer is not None:
        if gene_expr_layer in adata.layers:
            gene_expr = adata.layers[gene_expr_layer]
        else:
            raise ValueError(f"找不到层 '{gene_expr_layer}' 在 adata.layers 中")
    else:
        gene_expr = adata.X
    
    # 如果是scipy稀疏矩阵，转换为密集矩阵
    if sp.issparse(gene_expr):
        gene_expr = gene_expr.toarray()
    
    # 如果需要，仅使用高变基因
    if use_highly_variable:
        if 'highly_variable' in adata.var:
            gene_expr = gene_expr[:, adata.var['highly_variable']]
        else:
            print("警告: 未找到高变基因信息，使用所有基因")
    
    # 3. 使用KNN找到基于空间距离的最近邻
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)
    
    # 4. 构建边索引
    edge_index = []
    for i in range(len(indices)):
        # 跳过第一个邻居(自身)
        for j in indices[i][1:]:
            edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # 5. 转换节点特征为PyTorch张量
    node_features = torch.tensor(gene_expr, dtype=torch.float)
    
    # 6. 创建PyG的Data对象
    graph_data = Data(x=node_features, edge_index=edge_index)
    
    # 7. 构建邻接矩阵
    n_cells = len(adata)
    adj_rows, adj_cols = edge_index
    adj_values = torch.ones(edge_index.shape[1])
    adj_matrix = sp.coo_matrix(
        (adj_values.numpy(), (adj_rows.numpy(), adj_cols.numpy())),
        shape=(n_cells, n_cells)
    )
    
    return graph_data, adj_matrix.tocsr()



'''transfer NTdata and STdata to the pyg data'''
def build_connection_graph(adata, adj, threshold=0.001):
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