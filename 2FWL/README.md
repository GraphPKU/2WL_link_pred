This is the implementation of 2-FWL for homogeneous graphs.

# Build environment:
python 3.8 + pytorch 1.10.0 + pytorch-geometric 2.0.2 + optuna 2.10.0 + ogb 1.3.3

# Reproduce results 
```
python main.py --dataset Cora --test --use_best
```

# Tune hyperparameters
```
python main.py --dataset Cora
```

# Current best hyperparameters:

| dataset  | hidden_dimension   |   number of layers       |       dropout rate        |    lr    |
|----------|--------------------|--------------------------|---------------------------|----------|
| Celegans | (64,64)            |          (2,2)           |      (0.3, 0.3, 0.3)      |   0.005  | 
|  USAir   | (24,24)            |          (1,2)           |      (0.4, 0.0, 0.1)      |   0.01   |
|    PB    | (32,32)            |          (3,3)           |      (0.4, 0.4, 0.4)      |   0.005  |
|    NS    | (16,16)            |          (3,1)           |      (0.1, 0.1, 0.1)      |   0.005  |
|  Ecoli   | (32,32)            |          (3,1)           |      (0.3, 0.0, 0.4)      |   0.01   |
|  Router  | (16,16)            |          (2,1)           |      (0.4, 0.0, 0.0)      |   0.01   |
|  Power   | (32,8)             |          (3,3)           |      (0.3, 0.0, 0.1)      |   0.01   |
|  Yeast   | (64,32)            |          (1,2)           |      (0.4, 0.4, 0.4)      |   0.005  |
|   Cora   | (64,32)            |          (2,1)           |      (0.3, 0.5, 0.0)      |   0.005  |
| Citeseer | (24,16)            |          (1,1)           |      (0.1, 0.2, 0.4)      |   0.001  |