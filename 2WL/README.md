This is the implementation of 2-WL for homogeneous graphs.

# Reproduce results
```
python main.py --dataset USAir
```


# Current best hyperparameters:

| dataset  | hidden_dimension   |       dropout rate       |    lr    |
|----------|--------------------|--------------------------|----------|
| Celegans | (32,24)            |     (0.0, 0.2, 0.3)      |   0.01   | 
|  USAir   | (24,24)            |     (0.0, 0.3, 0.0)      |   0.01   |
|    PB    | (24,24)            |     (0.0, 0.0, 0.3)      |   0.01   |
|    NS    | (16,16)            |     (0.0, 0.0, 0.3)      |   0.01   |
|  Ecoli   | (32,24)            |     (0.0, 0.0, 0.3)      |   0.01   |
|  Router  | (16,16)            |     (0.0, 0.3, 0.0)      |   0.01   |
|  Power   | (16,16)            |     (0.0, 0.0, 0.3)      |   0.01   |
|  Yeast   | (32,24)            |     (0.0, 0.0, 0.3)      |   0.01   |
|   Cora   | (32,24)            |     (0.0, 0.1, 0.3)      |   0.001  |
| Citeseer | (32,24)            |     (0.1, 0.2, 0.3)      |   0.01   |