## SpaCon

![image](https://github.com/quhaichao/SpaCon/blob/main/Workflow.png)


## Overview
Understanding brain organization requires integrating spatial transcriptomics and connectomics, yet specialized methods are lacking. Here, we elucidated gene expression patterns in anatomically connected regions at the cellular level and introduced SpaCon, a deep-learning approach that integrates connectivity with spatial transcriptomic data to identify spatial domains where gene expression aligns with connectivity. Validated across multiple datasets, SpaCon robustly bridges the gap between connectome and transcriptome, offering critical insights into brain architecture.



## Software dependencies

* torch
* torch-geometric
* scanpy
* anndata
* seaborn
* rpy2
* R

## Raw data
The raw data used in this study can be accessed as follows: 10x Visium human DLPFC dataset: (https://research.libd.org/spatialLIBD/). STARmap dataset of mouse visual cortex: (https://drive.google.com/file/d/1r5IVrGz4353qMXoRotJNlNdvk7MGz8Ri/view?usp=drive_link). MERFISH dataset of mouse brain: (https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang_dataset.html). Slide-seq dataset of mouse brain: (https://www.braincelldata.org/). Stereo-seq dataset of mouse brain: (https://doi.org/10.12412/BSDC.1699433096.20001). Allen structural connectivity data: (https://connectivity.brain-map.org). MERFISH cell segmentation and decoded results of mouse brain data: (https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d/processed_data/).
