"""
The following class is an implementations of supervised kernel PCA (KSPCA).
This algorithm can be found in the paper
    Supervised principal component analysis: Visualization, classification and
    regression on subspaces and submanifolds
by Elnaz Barshan et al.
The code is based off code from the following GitHub webpage.
    https://github.com/bghojogh/Principal-Component-Analysis
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import pairwise_kernels
from generalized_eigenvalue_problem import generalized_eigenvalue_problem
from sklearn.neighbors import KNeighborsClassifier

class KSPCA:
    def __init__(self, n_components=2, kernel_metric_X='linear', kernel_metric_Y='linear'):
        # n_components = dimension for embedded data
        # kernel_metric_X = kernel metric used to compute the kernel matrix of training/test data
        # kernel_metric_Y = kernel metric used to compute the kernel matrix of target variable
        
        self.n_components = n_components
        self.kernel_metric_X = kernel_metric_X
        self.kernel_metric_Y = kernel_metric_Y
        self.eig_vec = None
        
    def transform(self, X=np.array([]), Y=np.array([])):
        # X = data matrix of shape (n_samples, n_features)
        # Y = target vector of shape (n_samples,)
        """
            Runs KSPCA on the dataset X with labels Y to reduce its dimensionality to n_compoents dimensions
        """
        
        (n,d) = X.shape
        
        # One-hot encode labels
        Y = np.reshape(Y,(n,1))
        ohe = OneHotEncoder()
        Y = ohe.fit_transform(Y)
        Y = Y.toarray()
        
        # Calculate eigenvectors of Q to obtain embedding
        H = np.eye(n) - ((1/n) * np.ones((n,n)))
        Kx = pairwise_kernels(X, metric=self.kernel_metric_X)
        Ky = pairwise_kernels(Y, metric=self.kernel_metric_Y)
        Q = (Kx).dot(H).dot(Ky).dot(H).dot(Kx)
        gen_eigen_prob = generalized_eigenvalue_problem(A=Q, B=Kx)
        self.eig_vec, eig_val = gen_eigen_prob.solve()
        Z = np.dot(Kx, self.eig_vec[:,0:self.n_components]) # Low-dimensional embedding
        
        return Z
    
    def embed_test_data(self, X_train=np.array([]), X_test=np.array([])):
        # X_test = test data of shape (n_samples_test, n_features)
        """
            Embeds test data into low dimensional space
        """
        
        Kx_test = pairwise_kernels(X_test, X_train, metric=self.kernel_metric_X)
        Z_test = np.dot(Kx_test, self.eig_vec[:,0:self.n_components]) # Low-dimensional embedding for test data
        
        return Z_test
    
    def obtain_test_labels_KNN(self, Z_train=np.array([]), Z_test=np.array([]), Y_train=np.array([]), n_neighbors=1):
        # Z_train = low dimensional embedding for X_train, of shape (n_samples_train, n_features)
        # Z_test = low dimensional embedding for X_test, of shape (n_samples_test, n_features)
        # Y_train = labels for X_trian, of shape (n_samples_train, )
        # n_neighbors = number of neighbors to use for KNN classifier
        """
            Obtains missing labels by training a KNN classifier on the embedded data  
        """
        
        n_train = Z_train.shape[0]
        
        # One-hot encode labels
        Y_train = np.reshape(Y_train,(n_train,1))
        ohe = OneHotEncoder()
        Y_train = ohe.fit_transform(Y_train)
        Y_train = Y_train.toarray()
        
        # Train classifier
        KNN = KNeighborsClassifier(n_neighbors=n_neighbors)
        KNN.fit(Z_train, Y_train)
        
        # Estimate unknown labels
        Y_test = KNN.predict(Z_test)
        
        return Y_test
    
    def score(self, Y_test=np.array([]), Y_test_approx=np.array([])):
        # Y_test = true labels for X_test, of shape (n_samples_test, )
        # Y_test_approx = estimated labels for X_test, one-hot encoded, of shape (n_samples_test, n_classes)
        """
            Calculates prediction accuracy of estimated labels
        """
        
        # One-hot encode true labels
        n_test = len(Y_test)
        Y_test = np.reshape(Y_test, (n_test,1))
        ohe = OneHotEncoder()
        Y_test = ohe.fit_transform(Y_test)
        Y_test = Y_test.toarray()
        
        acc = np.sum(Y_test==Y_test_approx, axis=1)/Y_test.shape[1]
        acc = np.sum(acc)/n_test
        
        return acc
    
if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.datasets import load_iris
    from sklearn.datasets import load_breast_cancer
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    
    # Load data
    data = load_digits()
    #data = load_iris()
    #data = load_breast_cancer()
    X = data.data
    if X.shape[1]>50:
        pca = PCA(n_components=50) # Preprocess data so n_features <=50
        X = pca.fit_transform(X).astype(np.float32, copy=False)
    (n, d) = X.shape
    #X = X - np.tile(np.mean(X, 0), (n, 1)) # Center data, performance seems to improve when data is centered
    Y = data.target
    indices = list(range(n))
    
    n_components = 2
    kernel_metric_X = 'linear'
    kernel_metric_Y = 'linear'
    test_size=0.2
    n_neighbors=5
    
    # Split data into train/test sets
    X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(X, Y, indices, test_size=test_size, random_state=None)
    
    KSPCA = KSPCA(n_components, kernel_metric_X, kernel_metric_Y)
    
    # Obtain KSPCA embedding
    Z_train = KSPCA.transform(X_train,Y_train).real
    Z_test = KSPCA.embed_test_data(X_train, X_test).real
    Z = np.zeros((n,n_components))
    Z[indices_train,:] = Z_train
    Z[indices_test,:] = Z_test
    
    # Plot 2D embedding for training and test data
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    plt.scatter(Z[:,0], Z[:,1], c=Y, cmap=cmap)
    plt.title('KSPCA')
    plt.show()
    
    Y_test_approx = KSPCA.obtain_test_labels_KNN(Z_train, Z_test, Y_train, n_neighbors)
    acc = KSPCA.score(Y_test, Y_test_approx)

    print(f"Accuracy of Y_test_approx using KNN = {acc}")