# Transductive t-SNE
This code contains the implmentation of transduction t-SNE (tt-SNE) and its comparisons scripts. 

## Configuration
It was tested with the following environment setup
- python 3.7
- install dependency: pip install -r requirements.txt

## How to call Transductive t-SNE 
In order to get embeddings of both labeled and unlabeled data, the script is
```
model = St_SNE(no_dims=args[0], perplexity=args[1], rho=args[2])
Z_train, Z_test = model.obtain_Z_ByComment3(X_train, X_test, Y_train)
```

## Experiments
- comparisons experiments
```commandline
python run_experiments.py
```
- added experiments in the revision
```commandline
python run_experiments_R1.py
```
Notice that [SS.t-SNE](https://github.com/wserna/Semi-supervised.t-SNE) is implemented in MATLAB. In order to run
the comparison experiments, MATLAB should be available.