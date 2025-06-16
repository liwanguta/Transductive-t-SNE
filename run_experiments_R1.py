import ray
import psutil
import numpy as np
import pandas as pd
import scipy.sparse
import itertools
import tempfile
import os
from pathlib import Path
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cpus)

@ray.remote
def nca(X_train, X_test, Y_train, Y_test, n_components):
    nca = NeighborhoodComponentsAnalysis(random_state=42, n_components=n_components)
    nca.fit(X_train, Y_train)
    Z_train = nca.transform(X_train)
    Z_test = nca.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Z_train, Y_train)
    accuracy = knn.score(Z_test, Y_test)
    return accuracy

@ray.remote
def tsne(X_train, X_test, Y_train, Y_test, args):
    n_components, perp = args
    X = np.concatenate([X_train, X_test])
    n_train = X_train.shape[0]
    X_embedded = TSNE(n_components=n_components,
                      perplexity=perp).fit_transform(X)
    Z_train = X_embedded[:n_train, :]
    Z_test = X_embedded[n_train:, :]

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Z_train, Y_train)
    accuracy = knn.score(Z_test, Y_test)
    return accuracy

@ray.remote
def sst_sne(X_train, X_test, Y_train, Y_test, args):
    dim, thresh_cat = args

    X = np.concatenate([X_train, X_test])
    n_train = X_train.shape[0]
    label = np.concatenate([Y_train, -1 * np.ones_like(Y_test)])
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname)
        X_csv_file = tmp_path / "X.csv"
        np.savetxt(X_csv_file, X, delimiter=",", fmt='%f')
        label_csv = tmp_path / "label_csv"
        np.savetxt(label_csv, label, delimiter=",", fmt='%d')
        out_csv = tmp_path / "Z.csv"
        dim = str(dim)
        thresh_cat = str(thresh_cat)
        cmd = f"/Applications/MATLAB_R2022b.app/bin/matlab -nodisplay -nosplash -nodesktop -r \"run_SStSNE('{X_csv_file}', '{label_csv}', '{dim}', '{thresh_cat}', '{out_csv}');exit;\""
        print(cmd)
        os.system(cmd)
        X_embedded = np.loadtxt(out_csv, delimiter=',')

        Z_train = X_embedded[:n_train, :]
        Z_test = X_embedded[n_train:, :]

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

    perp = [10, 20, 30, 40, 50]
    params = list(itertools.product(no_dims, perp))
    tsne_results = pd.DataFrame(data=np.asarray(params), columns=["dim", "perplexity"])
    tsne_results["TSNE"] = ray.get([tsne.remote(X_train_ray, X_test_ray,
                                                 Y_train_ray, Y_test_ray, args) for args in params])
    tsne_results.to_csv(f"{out_prefix}_tsne.csv")

    params = no_dims
    nca_results = pd.DataFrame(data=np.asarray(params), columns=["dim"])
    nca_results["nca"] = ray.get([nca.remote(X_train_ray, X_test_ray,
                                            Y_train_ray, Y_test_ray, args) for args in params])
    nca_results.to_csv(f"{out_prefix}_nca.csv")

    thresh_cat = [0.5, 0.6, 0.7, 0.8, 0.9]
    params = list(itertools.product(no_dims, thresh_cat))
    sstsne_results = pd.DataFrame(data=np.asarray(params), columns=["dim", "thresh_cat"])
    sstsne_results["sstsne"] = ray.get([sst_sne.remote(X_train_ray, X_test_ray,
                                                 Y_train_ray, Y_test_ray, args) for args in params])
    sstsne_results.to_csv(f"{out_prefix}_sstsne.csv")


if __name__ == "__main__":
    data_name = 'usps'
    out_dir = "results_R1"
    test_ratio = 0.9
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    for rep in range(10):
        input_data = f"datasets/{data_name}.mat"
        out = f"{out_dir}/{data_name}_{test_ratio}_{rep}"
        run_comparisons(input_data, out, test_ratio)
        break
