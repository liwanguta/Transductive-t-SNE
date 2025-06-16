"""
The following two class is an implementation of the Linear Discriminant Analysis (LDA) algorithm for dimensionality reduction.
This code is based off code from the following webpage.
    https://www.python-engineer.com/courses/mlfromscratch/14-lda/
"""

import numpy as np
from generalized_eigenvalue_problem import generalized_eigenvalue_problem
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

class LDA:

    def __init__(self, n_components=2):
        # n_components = dimension for embedded data
        
        self.n_components = n_components
        self.eig_vec = None

    def transform(self, X=np.array([]), Y=np.array([])):
        # X = data matrix of shape (n_samples, n_features)
        # Y = target vector of shape (n_samples,)
        """
            Runs LDA on the dataset X with labels Y to reduce its dimensionality to n_compoents dimensions
        """
        
        n_features = X.shape[1]
        class_labels = np.unique(Y)

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features)) # Within class scatter matrix
        SB = np.zeros((n_features, n_features)) # Between class scatter:
        for c in class_labels:
            X_c = X[Y == c]
            mean_c = np.mean(X_c, axis=0)
            SW += (X_c - mean_c).T.dot((X_c - mean_c))
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)
        
        gen_eigen_prob = generalized_eigenvalue_problem(A=SB, B=SW)
        self.eig_vec, eig_val = gen_eigen_prob.solve()
        Z = np.dot(X, self.eig_vec[:,0:self.n_components]) # Low-dimensional embedding
        
        return Z
    
    def embed_test_data(self, X_test=np.array([])):
        # X_test = test data of shape (n_samples_test, n_features)
        """
            Embeds test data into low dimensional space
        """
        
        Z_test = np.dot(X_test, self.eig_vec[:,0:self.n_components]) # Low-dimensional embedding for test data
        
        return Z_test
    
    def obtain_test_labels_KNN(self, Z_train=np.array([]), Z_test=np.array([]), Y_train=np.array([]), n_neighbors=5):
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
    
    # Define parameters
    n_components = 2
    test_size=0.2
    n_neighbors=5
    
    # Split data into train/test sets
    X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(X, Y, indices, test_size=test_size, random_state=None)
    
    LDA = LDA(n_components)
    
    # Obtain SPCA embedding
    Z_train = LDA.transform(X_train,Y_train).real
    Z_test = LDA.embed_test_data(X_test).real
    Z = np.zeros((n,n_components))
    Z[indices_train,:] = Z_train
    Z[indices_test,:] = Z_test
    
    # Plot 2D embedding for training and test data
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    plt.scatter(Z[:,0], Z[:,1], c=Y, cmap=cmap)
    plt.title('LDA')
    plt.show()
    
    Y_test_approx = LDA.obtain_test_labels_KNN(Z_train, Z_test, Y_train, n_neighbors)
    acc = LDA.score(Y_test, Y_test_approx)

    print(f"Accuracy of Y_test_approx using KNN = {acc}")
