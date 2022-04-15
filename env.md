```
conda create -n openks-models python=3.8
conda activate openks-models
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install nltk tqdm pyyaml scikit-learn pandas numpy -c conda-forge
conda install -c dglteam dgl-cuda11.0 -c conda-forge
```
