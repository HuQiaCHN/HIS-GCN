# HIS-GCN: Hierarchical graph sampling based minibatch learning with chain preservation and variance reduction

Qia Hu<sup>\*</sup>, Bo Jiao<sup>\*</sup>

The source code of [Hierarchical graph sampling based minibatch learning with chain preservation and variance reduction](https://doi.org/10.48550/arXiv.2503.00860)

## Directory Structure

```
HIS-GCN-master/
│   README.md
│   requirements.txt
│   ...
│
└───HIS/
│   │   globals.py
│   │   HISsampler.py
│   │   ...
│   │
│   └───pytorch_model/
│       │    train.py
│       │    model.py
│       │    ...
│
└───datasets/
│   └───ppi-large/
│   │   │    adj_train.npz
│   │   │    adj_full.npz
│   │   │    ...
│   │	
│   │
│   └───...
│
```

## Setup

#### Software Dependencies

- Python 3.9
- numpy>=1.26.4
- torch>=1.12.0
- scipy>=1.11.3
- pyyaml>=6.0.1
- networkx>=3.2.1
- scikit-learn>=1.3.2
- tqdm>=4.66.1
- GraphRicciCurvature>=0.5.3.2

## Datasets

We use six datasets, namely, Citeseer, Pubmed, PPI (large version), Reddit, OGBN-arxiv and OGBN-products, for evaluating HIS-GCN on node classification tasks. The datasets are available in [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT) and [FastGCN](https://github.com/matenure/FastGCN). The datasets we use have the same structure as GraphSAINT.  PPI (large version) has been included as a test case.

The datasets for other experiments, such as Ollivier-Ricci curvature calculation and core-periphery partition, can be found in [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/).

All txt files in `./HIS/data` folder correspond to raw data of graphs. The modules for converting the raw data into undirected graphs and removing duplicate edges and self-loops have been included in our code.

## Training

The hyperparameters needed in training can be set via the configuration file: `./train_config/<name>.yml`. Use the hyperparameters suggested in the paper for training.


```
python -m HIS.model.train --data_prefix ./datasets/<dataset_name> --train_config <path to train_config yml> --gpu <GPU number>
```

 `--gpu 0` indicates running on the first GPU. For OGBN-products, the verification and testing process may cause memory issue for some GPUs. If an out-of-memory error occurs, please use `--cpu_eval` to force the val / test set evaluation to run on CPU while the minibatch training is still running on GPU.

Sample output(PPI (large version)):

```
Epoch   49
 TRAIN (Ep avg): loss = 5.7514  mic = 0.9748    mac = 0.9703    train time = 0.9251 sec
 VALIDATION:     loss = 2.0129  mic = 0.9802    mac = 0.9782
  Saving model ...
Optimization Finished!
  Restoring model ...
Full validation (Epoch   49):
  F1_Micro = 0.9802     F1_Macro = 0.9782
Full test stats:
  F1_Micro = 0.9850     F1_Macro = 0.9842
Total training time:  50.08 sec
```

## Others

In addition, we provide two experiments about the Ollivier-Ricci curvature calculation and core-periphery partition. The txt files of graphs in `./HIS/data` folder can be replaced by other graph datasets. The Ollivier-Ricci curvature experiment ouputs the average of edge curvatures on the input graph and the average of edge curvatures of 1,000 subgraphs with 10% partial nodes, and the core-periphery partition experiment outputs degree threshold and process time. Please note that the input graph of the two experiments has been transformed into a simple and undirected graph.

Curvature experiment:

```
python -m HIS.curvature_exp
```

Please note that the curvature experiment only supports running on Linux.

Output (ego-facebook):

```
OllivierRicci curvature: 0.3244994421467344
Progress: 100%|██████████████████████████████| 1000/1000 [00:00<00:00, 345.99it/s]
subgraph OllivierRicci curvature：0.4061342377303167
```

Core-periphery partition experiment:

```
python -m HIS.Core_peripheryPartition_exp
```

Output (ppi-large):

```
process ppi-large ...
degree threshold: 57
process time:   0.03 sec
```

## License

Copyright (c) 2025 HuQia. All rights reserved.

Licensed under the [MIT](https://github.com/HuQiaCHN/HIS-GNN/blob/master/LICENSE) license.
