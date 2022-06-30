Two-Dimensional Weisfeiler-Lehman Graph Neural Networks for Link Prediction
===========================================================================

About
-----

This repository is the official implementation of the models in the [following paper](https://arxiv.org/abs/2205.11172v1):

Yang Hu, Xiyuan Wang, Zhouchen Lin, Pan Li, Muhan Zhang: Two-Dimensional Weisfeiler-Lehman Graph Neural Networks for Link Prediction. CoRR/abs:2206.09567 (2022)

```{bibtex}
@misc{2wl,
  title = {Two-Dimensional Weisfeiler-Lehman Graph Neural Networks for Link Prediction},
  author = {Yang Hu and Xiyuan Wang and Zhouchen Lin and Pan Li and Muhan Zhang},
  publisher = {arXiv},
  year = {2022}
}
```

2WLNet is a series of link prediction algorithms that directly use links (2-node-tuples) as message passing unit and stimulate 2-WL test to realise its message passing. 
It first takes node feature or node degree as initial input, then use 1-WL-GNN and pooling function to obtain link representation, finally use 2-WL-GNN to get prediction 
score for every questioned links. We adopt four different 2-WL tests: 2-WL, Local 2-WL, 2-FWL, Local 2-FWL, depending on which we construct four types of 2-WL-GNN layers.

Denote graph $G=(V,E), V=[n]$, 
four 2-WL tests define neighborhood of link $(p,q)$ as follows:

2-WL:
$$N(p,q) = \Big(\big\lbrace (r, q)\ \vert\ r\in[n]\big\rbrace,\ \big\lbrace (p, s)\ \vert\ s\in[n]\big\rbrace\Big) $$

Local 2-WL:
$$N(p,q) = \Big(\big\lbrace (u, q)\ \vert\ (u, q)\in E, u\in [n] \big\rbrace,\ \big\lbrace (p, v)\ \vert\ (p, v)\in E, v \in [n] \big\rbrace \Big)$$

2-FWL:
$$N(p,q) = \Big\lbrace\big((r, q),\ (p, r)\big)\ \vert\ r\in[n]\Big\rbrace$$

Local 2-FWL:
$$N(p,q) = \Big\lbrace\big((r, q),\ (p, r)\big)\ \vert\ (r,q)\in E,\ (p,r)\in E,\ r\in[n]\Big\rbrace$$

Installation
------------

To reproduce our results: Python 3.8 + Pytorch 1.10.0 + Pytorch-Geometric 2.0.2

Other python libraries for train: Optuna 2.10.0

Usage
-----

To reproduce results of Local 2-WL models on USAir using a designated gpu, you can use the following command:
```
python 2WLtest.py --dataset USAir --pattern 2wl_l --device $gpu-id
```
The other three models are used by command `--patterns 2wl`, `--patterns 2fwl`, `--patterns 2fwl_l`. You may also use `--device -1` to run code on a CPU.

To tune hyperparameters yourself, you can use the command:
```
python 2WLwork.py --dataset USAir --pattern 2wl_l --device $gpu-id
```
Notice that the parameters catagory and scope should be manually adjusted according to models and datasets.
