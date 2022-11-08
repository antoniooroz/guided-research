## Run the code
- Start the environment with "conda activate p_ppnp"

- To train all models you can execute the ./run_scripts/run.sh file (The MCD versions and the DE version are not trained as they use the pretrained corresponding standard model).

- To evaluate all models you can execute the ./run_scripts/load.sh file.

- The default dataset is Cora. Change "cora" to "citeseer" in run.sh and load.sh in order to train and evaluate on Citeseer.

## Windows Afterwards
- pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
- pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

## Requirements
- Tested under Ubuntu/WSL

- Install CUDA and cuDNN for GPU support.

- Setup the conda environment: 
conda env create -f p_ppnp_environment.yml python==3.9 --force

- Log in to WandB and change the project and entity for wandb.init in /pgnn/logger/logger.py

- Permission to execute the files run.sh and load.sh might need to be given ("chmod +x run.sh", "chmod +x load.sh") 

- If you use the `networkx_to_sparsegraph` method for importing other datasets you will additionally need NetworkX.

## Code
As the basis for this project we relied on the original PPNP implementation (basic structure, data, preprocessing, training, PPNP model, etc.):
Johannes Gasteiger, Aleksandar Bojchevski and Stephan Günnemann. *"Predict then Propagate: Graph Neural Networks meet Personalized PageRank."* ICLR 2019.
The github repository for this project: https://github.com/gasteigerjo/ppnp

The OOD Detection and the GPN model originate from:
M. Stadler, B. Charpentier, S. Geisler, D. Zügner, and S. Günnemann. *"Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification.*" 2021. doi: 10.48550/ARXIV.2110.14012.
The github repository for this project: https://github.com/stadlmax/Graph-Posterior-Network

For the PyTorch GCN implementation, we utilized code from the original TensorFlow implementation: 
T. N. Kipf and M. Welling. *“Semi-Supervised Classification with Graph Convolutional Networks.”* In: CoRR abs/1609.02907 (2016). arXiv: 1609.02907.
The github repository for this project: https://github.com/tkipf/gcn

The GAT implementation in PyTorch was provided by: 
A. Gordić. pytorch-GAT. https://github.com/gordicaleksa/pytorch-GAT. 2020.
The github repository for this project: https://github.com/gordicaleksa/pytorch-GAT

The according licenses for the listed projects are provided in ./licenses
## Datasets
In the `data` folder you can find several datasets. If you want to use other (external) datasets, you can e.g. use the `networkx_to_sparsegraph` method in `ppnp.data.io` for converting NetworkX graphs to our SparseGraph format.

The Cora-ML graph was extracted by Aleksandar Bojchevski, and Stephan Günnemann. *"Deep gaussian embedding of attributed graphs: Unsupervised inductive learning via ranking."* ICLR 2018,   
while the raw data was originally published by Andrew Kachites McCallum, Kamal Nigam, Jason Rennie, and Kristie Seymore. *"Automating the construction of internet portals with machine learning."* Information Retrieval, 3(2):127–163, 2000.

The Citeseer graph was originally published by Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Gallagher, and Tina Eliassi-Rad.
*"Collective Classification in Network Data."* AI Magazine, 29(3):93–106, 2008.