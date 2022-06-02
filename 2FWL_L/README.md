This is the implementation of local 2-FWL for homogeneous graphs.

# Reproduce results for Cora/Citeseer dataset
```
python Cora/main.py --dataset Cora --test --use_best
```
```
python Citeseer/main.py --dataset Citeseer --test --use_best
```

# Reproduce results for other datasets
```
python main.py --dataset USAir --test --use_best
```

# Tune hyperparameters
```
python Cora/main.py --dataset Cora
```
```
python main.py --dataset USAir
```

# Current best hyperparameters:

| dataset  | hidden_dimension   |   number of layers       |       dropout rate        |    lr    |
|----------|--------------------|--------------------------|---------------------------|----------|
| Celegans | (32,64)            |          (3,1)           |      (0.2, 0.0, 0.1)      |   0.01   | 
|  USAir   | (32,32)            |          (3,2)           |      (0.1, 0.0, 0.2)      |   0.01   |
|    PB    | (64,24)            |          (3,2)           |      (0.0, 0.0, 0.1)      |   0.005  |
|    NS    | (24,32)            |          (2,2)           |      (0.3, 0.0, 0.0)      |   0.05   |
|  Ecoli   | (64,24)            |          (2,1)           |      (0.1, 0.0, 0.1)      |   0.005  |
|  Router  | (64,24)            |          (3,1)           |      (0.2, 0.0, 0.0)      |   0.005  |
|  Power   | (24,64)            |          (2,3)           |      (0.1, 0.0, 0.2)      |   0.01   |
|  Yeast   | (32,24)            |          (2,2)           |      (0.3, 0.0, 0.1)      |   0.01   |
|   Cora   | (128,32)           |          (1,2)           |      (0.5, 0.0, 0.8)      |   0.001  |
| Citeseer | (256,32)           |          (2,2)           |    (0.8, 0.6, 0.1, 0.0)   |   0.0005 |
|  Pubmed  | (128,96)           |          (1,1)           |      (0.1, 0.0, 0.4)      |   0.005  |