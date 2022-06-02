This is the implementation of Local 2-WL for homogeneous graphs.

# Reproduce results 
```
python main.py --dataset Cora --test 
```

# Tune hyperparameters
```
python main.py --dataset Cora
```

# Current best hyperparameters:

| dataset  | hidden_dimension   |   number of layers       |         dropout rate           |    lr    |
|----------|--------------------|--------------------------|--------------------------------|----------|
| Celegans | 64                 |          (2,3)           |      (0.2, 0.2, 0.2, 0.2)      |   0.005  | 
|  USAir   | 64                 |          (3,2)           |      (0.1, 0.1, 0.1, 0.1)      |   0.001  |
|    PB    | 32                 |          (3,2)           |      (0.0, 0.0, 0.0, 0.0)      |   0.01   |
|    NS    | 48                 |          (3,1)           |      (0.1, 0.1, 0.1, 0.1)      |   0.001  |
|  Ecoli   | 32                 |          (3,1)           |      (0.1, 0.1, 0.1, 0.1)      |   0.005  |
|  Router  | 48                 |          (3,2)           |      (0.1, 0.1, 0.1, 0.1)      |   0.01   |
|  Power   | 64                 |          (3,3)           |      (0.1, 0.1, 0.1, 0.1)      |   0.05   |
|  Yeast   | 64                 |          (3,3)           |      (0.0, 0.0, 0.0, 0.0)      |   0.005  |
|   Cora   | 64                 |          (2,2)           |      (0.3, 0.3, 0.1, 0.0)      |   0.005  |
| Citeseer | 32                 |          (2,1)           |      (0.0, 0.4, 0.1, 0.0)      |   0.005  |
|  Pubmed  | 128                |          (1,2)           |      (0.1, 0.1, 0.1, 0.1)      |   0.01   |