## SpaCon

![image](https://github.com/quhaichao/SpaCon/blob/main/Workflow.png)


## Overview
The brain’s structure and function arise from its complex molecular composition and neural connectivity, yet the relationship between cell-type-specific gene expression and brain-wide connectivity is not well understood. By integrating single-cell resolution spatial transcriptomics and connectomics, we reveal a tight coupling between gene expression and anatomical connectivity in the cortico-thalamic circuit, and identify specific gene expression in axons of the corpus callosum that reflect their cortical origins. Building on these findings, we developed SpaCon, a graph attention autoencoder that uses a tunable hyperparameter to flexibly integrate global connectivity with local gene expression. To enable analysis at the whole-brain scale, SpaCon introduces an efficient neighbor sampling strategy that drastically reduces computational requirements without compromising performance. This architecture allows the model to identify functionally relevant three-dimensional domains defined by both molecular identity and shared connectivity patterns, even when spatially distant. Validated across diverse datasets and species, SpaCon significantly enhances the prediction of connectivity from gene expression and improves the spatial classification of neuronal subtypes. SpaCon thus provides a powerful, scalable, and versatile framework for understanding the complex relationship between transcriptome and connectome.



## Main software dependencies

* torch==2.0.1
* torch-geometric==2.3.1
* numpy==1.22.3
* scanpy==1.8.2
* anndata==0.9.1
* pandas==1.5.3
* rpy2==3.5.12
* scipy==1.10.1



## Tutorials

 * Source code to reproduce the main results of this article is available at https://spacon-results-reproduction.readthedocs.io/en/latest/.
 * The step-by-step tutorial of SpaCon is available at https://spacon-tutorials.readthedocs.io/en/latest/



## Raw data
 The raw data used in this study can be accessed as follows: MERFISH dataset of mouse brain is available at https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang_dataset.html. Slide-seq dataset of mouse brain is available at https://www.braincelldata.org/. Stereo-seq dataset of mouse brain is available at https://doi.org/10.12412/BSDC.1699433096.20001. Allen structural connectivity data is available at https://connectivity.brain-map.org. MERFISH cell segmentation and decoded results of mouse brain data is available at https://download.brainimagelibrary.org/29/3c/293cc39ceea87f6d/processed_data/.

 Our processed data can be found in the tutorial above.