import numpy as np

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', key_added='clusters', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()

    # set the randseed
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)

    # build the r objcet to cluster
    rmclust = robjects.r['Mclust']

    # cluster use model: EEE
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    # add the result to the adata
    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')
    return adata