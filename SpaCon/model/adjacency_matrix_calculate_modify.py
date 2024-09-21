import pandas as pd
import numpy as np
import sklearn.neighbors
from tqdm import tqdm


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


def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge / adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df / adata.shape[0]
    fig, ax = plt.subplots(figsize=[8, 4])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)
    ax.bar(plot_df.index, plot_df)
    plt.show()

