import os
import ray
import psutil
import numpy as np
import itertools
import pandas as pd
import scipy.sparse
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from St_SNE import St_SNE
from SPCA import SPCA
from KSPCA import KSPCA
from LDA import LDA
from Fisher_tSNE import Fisher_tSNE

num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cpus)


@ray.remote
def tSt_SNE(X_train, X_test, Y_train, Y_test, args):
    model = St_SNE(no_dims=args[0], perplexity=args[1], rho=args[2])
    Z_train = model.transform(X_train, Y_train)
    Z_test = model.obtain_Z2_tSNE(X_train, X_test, Z_train)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Z_train, Y_train)
    accuracy = knn.score(Z_test, Y_test)
    return accuracy


@ray.remote
def tt_SNE(X_train, X_test, Y_train, Y_test, args):
    model = St_SNE(no_dims=args[0], perplexity=args[1], rho=args[2])
    Z_train, Z_test = model.obtain_Z_ByComment3(X_train, X_test, Y_train)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Z_train, Y_train)
    accuracy = knn.score(Z_test, Y_test)
    return accuracy


@ray.remote
def spca(X_train, X_test, Y_train, Y_test, args):
    model = SPCA(n_components=args[0], kernel_metric=args[1])
    Z_train = model.transform(X_train, Y_train).real
    Z_test = model.embed_test_data(X_test).real
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Z_train, Y_train)
    accuracy = knn.score(Z_test, Y_test)
    return accuracy


@ray.remote
def kspca(X_train, X_test, Y_train, Y_test, args):
    model = KSPCA(n_components=args[0], kernel_metric_X=args[1], kernel_metric_Y=args[1])
    Z_train = model.transform(X_train, Y_train).real
    Z_test = model.embed_test_data(X_train, X_test).real
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Z_train, Y_train)
    accuracy = knn.score(Z_test, Y_test)
    return accuracy


@ray.remote
def lda(X_train, X_test, Y_train, Y_test, args):
    model = LDA(n_components=args[0])
    Z_train = model.transform(X_train, Y_train).real
    Z_test = model.embed_test_data(X_test).real
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Z_train, Y_train)
    accuracy = knn.score(Z_test, Y_test)
    return accuracy


@ray.remote
def fisher_SNE(X_train, X_test, Y_train, Y_test, args):
    model = Fisher_tSNE(no_dims=args[0], perplexity=args[1], sigma_param=args[2])
    Z_train = model.transform(X_train, Y_train)
    Z_test = model.embed_test_data(X_train, X_test, Z_train)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Z_train, Y_train)
    accuracy = knn.score(Z_test, Y_test)
    return accuracy


def run_comparisons(data_file, out_prefix, test_ratio):
    data = loadmat(data_file)
    Y = np.squeeze(data["y"])
    X = data["X"]  # .toarray()
    if scipy.sparse.issparse(X):
        X = X.toarray()
    n, d = X.shape
    if d > 50:
        pca = PCA(n_components=50)
        X = pca.fit_transform(X).astype(np.float32, copy=False)

    # Split data into train/test sets
    X_train, X_test, Y_train, Y_test, indices_train, indices_test \
        = train_test_split(X, Y, list(range(n)), test_size=test_ratio)

    # set random seed
    np.random.seed(0)

    X_train_ray = ray.put(X_train)
    X_test_ray = ray.put(X_test)
    Y_train_ray = ray.put(Y_train)
    Y_test_ray = ray.put(Y_test)
    no_dims = [2, 3]

    # set parameter for tSt_SNE and our method
    perp = [10, 20, 30, 40, 50]
    rho = [0.25, 0.5, 0.75]
    params = list(itertools.product(no_dims, perp, rho))
    results = pd.DataFrame(data=np.asarray(params), columns=["dim", "perplexity", 'rho'])
    results["tSt_SNE"] = ray.get([tSt_SNE.remote(X_train_ray, X_test_ray,
                                                 Y_train_ray, Y_test_ray, args) for args in params])
    results["our"] = ray.get([tt_SNE.remote(X_train_ray, X_test_ray,
                                            Y_train_ray, Y_test_ray, args) for args in params])
    results.to_csv(f"{out_prefix}_tStSNE_our.csv")

    # set parameters for sparse pca and kernel sparse pca
    kernels = ['linear', 'rbf']
    spca_params = list(itertools.product(no_dims, kernels))
    spca_results = pd.DataFrame(data=np.asarray(spca_params), columns=["dim", "kernel"])
    spca_results["SPCA"] = ray.get([spca.remote(X_train_ray, X_test_ray,
                                                Y_train_ray, Y_test_ray, args) for args in spca_params])
    spca_results["KSPCA"] = ray.get([kspca.remote(X_train_ray, X_test_ray,
                                                  Y_train_ray, Y_test_ray, args) for args in spca_params])
    spca_results.to_csv(f"{out_prefix}_spca_kspca.csv")

    # set parameters for Fisher_tSNE
    perp = [10, 20, 30, 40, 50]
    sigmas = [0.001, 0.01, 0.1, 1.0]
    fisher_params = list(itertools.product(no_dims, perp, sigmas))
    fisher_results = pd.DataFrame(data=np.asarray(fisher_params), columns=["dim", "perplexity", 'sigma'])
    fisher_results["Fisher_SNE"] = ray.get([fisher_SNE.remote(X_train_ray, X_test_ray,
                                                              Y_train_ray, Y_test_ray, args) for args in fisher_params])
    fisher_results.to_csv(f"{out_prefix}_fisherSNE.csv")

    # LDA does not have parameter
    lda_param = list(itertools.product(no_dims))
    lda_results = pd.DataFrame(data=np.asarray(lda_param), columns=["dim"])
    lda_results["LDA"] = ray.get([lda.remote(X_train_ray, X_test_ray,
                                             Y_train_ray, Y_test_ray, args) for args in lda_param])
    lda_results.to_csv(f"{out_prefix}_lda.csv")


if __name__ == "__main__":
    data_name = 'usps'
    out_dir = "results"
    test_ratio = 0.9
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    for rep in range(10):
        input_data = f"datasets/{data_name}.mat"
        out = f"{out_dir}/{data_name}_{test_ratio}_{rep}"
        run_comparisons(input_data, out, test_ratio)
        break