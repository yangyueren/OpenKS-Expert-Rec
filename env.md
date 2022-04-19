```
conda create -n openks-models python=3.8
conda activate openks-models
conda install nltk tqdm pyyaml scikit-learn pandas numpy requests -c conda-forge
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -c dglteam dgl-cuda11.0 -c conda-forge
pip install transformers==4.18.0
pip install sentence-transformers==2.2.0
conda install xmltodict ujson ordered-set -c conda-forge
```
