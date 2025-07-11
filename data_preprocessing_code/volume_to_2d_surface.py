import os
import h5py
import numpy as np
import anndata as ad
import pandas as pd
from scipy.sparse import issparse
import multiprocessing
from tqdm import tqdm

class CorticalMap(object):
    # 原有代码保持不变
    REFERENCE_SHAPE = (132, 80, 114)
    VALID_PROJECTIONS = ('top_view', 'dorsal_flatmap')
    
    @staticmethod
    def _load_paths(projection):
        path = './top_view_paths_100.h5'
        with h5py.File(path, 'r') as f:
            view_lookup = f['view lookup'][:]
            paths = f['paths'][:]
        return view_lookup, paths
        
    def __init__(self, projection='top_view'):
        if projection not in self.VALID_PROJECTIONS:
            raise ValueError('projection must be one of %s,  not %s'
                             % (self.VALID_PROJECTIONS, projection))
        self.view_lookup, self.paths = self._load_paths(projection)
        self.projection = projection
        
    def transform(self, volume, agg_func=np.mean, fill_value=0):
        # Transforms image volume values to 2D cortical surface.
        def apply_along_path(i):
            path = self.paths[i]
            arr = volume.flat[path[path.nonzero()]]
            return agg_func(arr) if len(arr) > 0 else fill_value
            
        if volume.shape != self.REFERENCE_SHAPE:
            raise ValueError('volume must have shape %s, not %s'
                             % (self.REFERENCE_SHAPE, volume.shape))
        if not callable(agg_func):
            raise ValueError('agg_func must be callable, not %s' % agg_func)
            
        result = np.zeros(self.view_lookup.shape, dtype=volume.dtype)
        apply_along_paths_ = np.vectorize(apply_along_path)
        idx = np.where(self.view_lookup > -1)
        result[idx] = apply_along_paths_(self.view_lookup[idx])
        
        if fill_value != 0:
            result[self.view_lookup == -1] = fill_value
            
        return result

# 定义一个处理单个基因的函数，用于并行处理
def process_gene(args):
    gene_idx, gene_name, spatial_coords, gene_expr, ref_shape, cortical_map = args
    
    # 创建空的3D体积
    volume = np.zeros(ref_shape, dtype=np.float32)
    
    # 根据坐标填充体积
    for i, (x, y, z) in enumerate(spatial_coords):
        # 四舍五入到最近的整数
        x, y, z = int(round(x)), int(round(y)), int(round(z))
        if 0 <= x < ref_shape[0] and 0 <= y < ref_shape[1] and 0 <= z < ref_shape[2]:
            volume[x, y, z] = gene_expr[i]
    
    # 转换到2D皮质图
    cortical_map_2d = cortical_map.transform(volume, agg_func=np.mean)
    
    return gene_idx, cortical_map_2d

def map_spatial_transcriptome_to_cortical_surface(input_h5ad_path, output_h5ad_path, projection='top_view', batch_size=50, n_jobs=None):
    """
    优化版：将h5ad中的空间转录组数据映射到2D皮质图，并保存为新的h5ad文件
    
    参数:
    input_h5ad_path: 输入的h5ad文件路径
    output_h5ad_path: 输出的h5ad文件路径
    projection: 投影类型，'top_view'或'dorsal_flatmap'
    batch_size: 每批处理的基因数量
    n_jobs: 并行处理的作业数量（None表示使用所有可用CPU核心）
    """
    print(f"加载空间转录组数据: {input_h5ad_path}")
    adata = ad.read_h5ad(input_h5ad_path)
    
    # 初始化皮质图映射器
    cortical_map = CorticalMap(projection=projection)
    cortical_shape = cortical_map.view_lookup.shape
    
    # 获取空间坐标
    if 'X_spatial' in adata.obsm:
        spatial_coords = adata.obsm['X_spatial']
    elif all(coord in adata.obs.columns for coord in ['x', 'y', 'z']):
        spatial_coords = adata.obs[['x', 'y', 'z']].values
    else:
        raise ValueError("无法找到空间坐标。应在adata.obsm['X_spatial']或adata.obs中的x、y、z列")
    
    # 创建新的AnnData对象来存储2D皮质图结果
    # 计算2D点的数量（非-1的值）
    valid_points = np.sum(cortical_map.view_lookup > -1)
    y_coords, x_coords = np.where(cortical_map.view_lookup > -1)
    cortical_coords = np.column_stack((x_coords, y_coords))
    
    # 获取基因信息
    num_genes = adata.n_vars
    genes = adata.var_names.tolist()
    
    # 创建新的AnnData对象
    adata_2d = ad.AnnData(
        X=np.zeros((valid_points, num_genes)),
        obs=pd.DataFrame(index=[f"point_{i}" for i in range(valid_points)]),
        var=adata.var.copy()
    )
    
    # 添加2D坐标
    adata_2d.obsm['X_spatial_2d'] = cortical_coords
    
    # 设置并行处理的作业数
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    print(f"使用 {n_jobs} 个CPU核心进行并行处理")
    
    # 批量处理基因，避免内存溢出
    for batch_start in range(0, num_genes, batch_size):
        batch_end = min(batch_start + batch_size, num_genes)
        print(f"处理基因批次 {batch_start+1}-{batch_end} (共 {num_genes} 个)")
        
        # 为当前批次准备数据
        batch_args = []
        for gene_idx in range(batch_start, batch_end):
            gene_name = genes[gene_idx]
            
            # 获取当前基因的表达值
            if issparse(adata.X):
                gene_expr = adata.X[:, gene_idx].toarray().flatten()
            else:
                gene_expr = adata.X[:, gene_idx]
                
            batch_args.append((gene_idx, gene_name, spatial_coords, gene_expr, 
                              cortical_map.REFERENCE_SHAPE, cortical_map))
        
        # 并行处理当前批次的基因
        with multiprocessing.Pool(n_jobs) as pool:
            results = list(tqdm(pool.imap(process_gene, batch_args), 
                               total=len(batch_args), 
                               desc="处理基因"))
        
        # 保存结果到AnnData对象
        for gene_idx, cortical_map_2d in results:
            valid_values = cortical_map_2d[y_coords, x_coords]
            adata_2d.X[:, gene_idx] = valid_values
    
    # 添加元数据
    # adata_2d.uns['cortical_map'] = {
    #     'projection': projection,
    #     'original_shape': cortical_map.view_lookup.shape
    # }
    
    # 为避免内存问题，不存储完整的皮质图到uns中
    # 如果需要，可以稍后根据2D坐标和值重建完整皮质图
    
    # 保存结果
    print(f"保存结果到: {output_h5ad_path}")
    adata_2d.write_h5ad(output_h5ad_path)
    print("完成!")
    
    return adata_2d

# 使用示例
if __name__ == "__main__":
    import pandas as pd
    
    input_h5ad = "/mnt/Data16Tc/home/haichao/code/SpaCon/Data/N_20231213_zxw/mouse_1/adata_processed.h5ad"
    output_h5ad = "/mnt/Data16Tc/home/haichao/code/SpaCon/ST_FC_cluster/mouse1/data/zxw1_wide_field/zxw1_cortical_map.h5ad"
    
    # 使用4个并行进程，每次处理50个基因
    adata_2d = map_spatial_transcriptome_to_cortical_surface(
        input_h5ad, 
        output_h5ad,
        batch_size=300,
        n_jobs=50
    )