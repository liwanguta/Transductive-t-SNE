"""
The following class is an implementation of Fisher Kernel t-SNE (Fisher t-SNE) for dimensionality reduction.
The algorithm can be found in the paper
    Parametric nonlinear dimensionality reduction using kernel t-SNE
by Andrej Gisbrecht et al.
Note: this implementation uses T=1, where T is the number of equidistant points considered in the calculation of Fisher distances.
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

class Fisher_tSNE:
    
    def __init__(self, no_dims=2, train_split=0.5, perplexity=30.0, sigma_param=1.0):
        # no_dims = dimension for embedded data
        # perplexity = t-SNE perplexity parameter, typically in the range [5, 50]
        # sigma_param = scaling parameter used to calculate kernel bandwidths, typically in the range [0.1, 1], 
        
        self.no_dims = no_dims
        self.perplexity = perplexity
        self.sigma_param = sigma_param
    
    
    def calculate_J(self, idx, B, sigma, P=np.array([])):
        # idx = current index (1,...,n)
        # B = b(x,c) values, list of size n_classes containing arrays of size (n, d)
        # sigma = bandwidths used to calculate Fisher distances
        # P = p(c|x) values, array of size (n, n_classes)
        """
            Computes Fisher information matrix J
        """
        
        n_classes = len(B)
        rows = []
        for i in range(n_classes):
            rows.append(B[i][idx,:])
        B = np.vstack(rows)
        J = (1/sigma[idx]**4)*np.matmul(B.T, B*P[:,np.newaxis])
        
        return  J
    
    def calculate_Fisher_distance(self, X=np.array([]), Y = np.array([])):
        # X = data matrix of shape (n_samples, n_features)
        # Y = labels, of shape (n_samples, )
        """
            Computes Fisher distance matrix D_trDisc
        """
        
        print("Calculating Fisher distances...")
        
        (n, d) = X.shape
        D_trDisc = np.zeros((n,n))
        
        # One-hot encode labels
        Y = np.reshape(Y,(n,1))
        ohe = OneHotEncoder()
        Y = ohe.fit_transform(Y)
        Y = Y.toarray()
        
        # Calculate sigma (bandwidths used to calculate Fisher distances)
        D = pairwise_distances(X, metric='euclidean')**2
        s = np.sqrt(np.sum(D, axis=1)/n)
        q3, q1 = np.percentile(D, [75 ,25])
        iqr = q3 - q1
        sigma = 1.06 * np.minimum(s, iqr/1.34) * n**(-1/5) # Silverman's rule of thumb
        H = (-0.5/sigma**2)

        # Compute p(c|x) values
        expD = np.exp(H[:,np.newaxis]*D)
        sum_expD = np.sum(expD, axis=1)
        e_ix = expD / sum_expD[:,np.newaxis]
        tops = np.matmul(expD, Y)
        P = tops / sum_expD[:,np.newaxis]
        
        # Compute b(x,c) values 
        second_term = np.matmul(e_ix,X)
        n_classes = Y.shape[1]
        B = []
        for i in range(n_classes):
            E = expD*Y[:,i]
            sumE = np.sum(E, axis=1)
            e_ixc = E / sumE[:,np.newaxis]
            first_term = np.matmul(e_ixc,X)
            B.append(first_term-second_term)
 
        # Calculate Fisher distances
        D_trDisc = np.zeros((n,n))
        for i in range(n):
            J_i = self.calculate_J(i, B, sigma, P[i,:])
            for j in range(n):
                xdiff = (X[i,:]-X[j,:])
                D_trDisc[i,j] = (xdiff).dot(J_i).dot(xdiff.T)

        return D_trDisc
    
    def calculate_kernel_bandwidth(self, D=np.array([])):
        # D = distance matrix of shape (n_samples, n_samples) or (n_samples_test, n_samples_train)
        """
            Determines bandwidth parameters for kernel mapping K
        """
        
        np.fill_diagonal(D, np.inf)
        sigma = self.sigma_param*np.sqrt(np.min(D,axis=0))
        np.fill_diagonal(D, 0)
        
        return sigma
    
    def calculate_K(self, D=np.array([]), sigma=np.array([])):
        # D = distance matrix of shape (n_samples, n_samples) or (n_samples_test, n_samples_train)
        # sigma = bandwidth parameters for kernel mapping  
        """
            Calculates Kernel matrix K
        """
        
        K = np.exp(-0.5/np.maximum(1e-12,sigma**2)*D)
        sumK = np.sum(K, axis=1)
        K = K / np.maximum(1e-12, sumK[:,np.newaxis])
        
        return K
    
    def transform(self, X=np.array([]), Y=np.array([])):
        # X = data matrix of shape (n_samples, n_features)
        # Y = labels, of shape (n_samples, )
        """
            Uses t-SNE with Fisher distances to obtain low-dim embedding Z for data X with labels Y
        """
        
        (n,d) = X.shape
        
        # Obtain embedding Z for training data
        D_trDisc = self.calculate_Fisher_distance(X, Y)
        print("Calculating embedding for training data using t-SNE and Fisher distances...")
        Z = TSNE(n_components=self.no_dims, perplexity=self.perplexity, early_exaggeration=4.0, 
                 learning_rate=500, metric='precomputed', method='exact', init='random').fit_transform(D_trDisc)
        
        return Z

    def embed_test_data(self, X_train=np.array([]), X_test = np.array([]), Z_train=np.array([])):
        # X_train = training data of shape (n_samples_train, n_features)
        # X_test = test data of shape (n_samples_test, n_features)
        # Z_train = low dimensional embedding for X_train, of shape (n_samples_train, n_features)
        """
            Embeds test data into low dimensional space
        """
        
        # Distance matrices
        D_tr = pairwise_distances(X_train, metric='euclidean')**2
        D_test = pairwise_distances(X_test, X_train, metric='euclidean')**2
        
        # Calculate bandwidth parameters for kernel mapping K_train
        sigma = self.calculate_kernel_bandwidth(D_tr)
        
        # Calculate kernel mapping K_train for training data
        K_train = self.calculate_K(D_tr, sigma)
        
        K_train_inv = np.linalg.pinv(K_train)
        A = np.matmul(K_train_inv, Z_train)        
        
        # Calculate kernel mapping K_test for test data
        K_test = self.calculate_K(D_test, sigma)
        
        # Obtain embedding Z_test for test data
        Z_test = np.matmul(K_test,A)
        
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
    #X = X - np.tile(np.mean(X, 0), (n, 1)) # Center data
    Y = data.target
    indices = list(range(n))
    
    no_dims = 2
    perplexity=30
    sigma_param=1.0
    test_size=0.2
    n_neighbors=5

    # Split data into train/test sets
    X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(X, Y, indices, test_size=test_size, random_state=None)
    
    Fisher_tSNE = Fisher_tSNE(no_dims, perplexity, sigma_param)
    
    # Obtain Fisher t-SNE embedding
    Z_train = Fisher_tSNE.transform(X_train,Y_train)
    Z_test = Fisher_tSNE.embed_test_data(X_train,X_test,Z_train)
    Z = np.zeros((n,no_dims))
    Z[indices_train,:] = Z_train
    Z[indices_test,:] = Z_test
    
    # Plot 2D embedding for training and test data
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    plt.scatter(Z[:,0], Z[:,1], c=Y, cmap=cmap)
    plt.title('Fisher t-SNE')
    plt.show()
    
    Y_test_approx = Fisher_tSNE.obtain_test_labels_KNN(Z_train, Z_test, Y_train, n_neighbors)
    acc = Fisher_tSNE.score(Y_test, Y_test_approx)

    print(f"Accuracy of Y_test_approx using KNN = {acc}")
    