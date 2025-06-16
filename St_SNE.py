"""
The following class is an implementation of the supervised t-SNE (St-STE) algorithm for dimensionality reduction. 
This algorithm can be found in the paper
    Supervised t-distributed stochastic neighbor embedding for data visualization and classification
by Yichen Cheng et al.
"""


import numpy as np
import math
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

class St_SNE:
    
    def __init__(self, no_dims=2, perplexity=30.0, rho=0.5):
        # no_dims = lower dimension for embedded data
        # perplexity = perplexity parameter
        # rho = supervision parameter
        
        self.no_dims = no_dims
        self.perplexity = perplexity
        self.rho = rho

    def Hbeta(self, D=np.array([]), beta=1.0):
        """
            Computes the perplexity and the P-row for a specific value of the
            precision of a Gaussian distribution.
        """
    
        # Compute P-row and corresponding perplexity
        P = np.exp(-D.copy() * beta)
        sumP = sum(P)
        H = np.log(np.maximum(1e-12,sumP)) + beta * np.sum(D * P) / np.maximum(1e-12,sumP)
        P = P / np.maximum(1e-12,sumP)
        return H, P

    def x2p(self, X=np.array([]), tol=1e-5):
        """
            Performs a binary search to get P-values in such a way that each
            conditional Gaussian has the same perplexity.
        """
    
        # Initialize some variables
        (n, d) = X.shape                                             
        sum_X = np.sum(np.square(X), 1)                              
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X) #Pairwise distance matrix ||x_i-x_j||^2
        P = np.zeros((n, n))
        beta = np.ones((n, 1))
        logU = np.log(self.perplexity)
    
        # Loop over all datapoints
        for i in range(n):
            
            # Print progress
            #if i % 500 == 0:
            #    print("Computing P-values for point %d of %d..." % (i, n))
    
            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
            (H, thisP) = self.Hbeta(Di, beta[i])
    
            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0
            while np.abs(Hdiff) > tol and tries < 50:
    
                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.
    
                # Recompute the values
                (H, thisP) = self.Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1
    
            # Set the final row of P
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    
        # Return final P-matrix
        #print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
        return P

    def y2o(self, Y=np.array([])):
        
        D = pairwise_distances(Y, metric='euclidean')
    
        Dexp = np.exp(-(D**2))
        np.fill_diagonal(Dexp, 0)
        
        Z = Dexp.sum(axis=1)
        O_cond = Dexp/np.maximum(1e-12,Z)
        O = 0.5*(O_cond + np.transpose(O_cond))
        O = O/np.sum(O)
        
        return O, O_cond

    def transform(self, X=np.array([]), Y=np.array([])):
        # X = data matrix, of shape (n_samples, n_features)
        # Y = labels, of shape (n_samples,)
        """
            Runs St-SNE on the dataset X with labels Y to reduce its dimensionality to no_dims dimensions. 
        """
    
        # Check inputs
        if isinstance(self.no_dims, float):
            print("Error: array X should have type float.")
            return -1
        if round(self.no_dims) != self.no_dims:
            print("Error: number of dimensions should be an integer.")
            return -1
    
        print("Running St-SNE...")
        # Initialize variables
        (n, d) = X.shape
        max_iter = 500
        initial_momentum = 0.5
        final_momentum = 0.8
        eta = 500
        min_gain = 0.01
        Z = np.random.randn(n, self.no_dims)
        dZ = np.zeros((n, self.no_dims))
        iZ = np.zeros((n, self.no_dims))
        gains = np.ones((n, self.no_dims))
    
        # Compute P-values
        P = self.x2p(X, 1e-5)
        P = P + np.transpose(P)
        P = P / np.sum(P)
        P = P * 4.	# early exaggeration
        P = np.maximum(P, 1e-12)
        
        # One-hot encode labels
        Y = np.reshape(Y,(n,1))
        ohe = OneHotEncoder()
        Y = ohe.fit_transform(Y)
        Y = Y.toarray()
        
        # Compute O-values
        O = self.y2o(Y)[0]
        O = O * 4 # early exaggeration
        O = np.maximum(O, 1e-12)

        # Run iterations
        for iter in range(max_iter):       
    
            # Compute pairwise affinities
            sum_Z = np.sum(np.square(Z), 1)
            num = -2. * np.dot(Z, Z.T)
            num = 1. / (1. + np.add(np.add(num, sum_Z).T, sum_Z))
            num[range(n), range(n)] = 0.
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)
    
            # Compute gradient
            PQ = P - Q
            OQ = O - Q
            for i in range(n):
                dZ[i, :] = self.rho*np.sum(np.tile(PQ[:, i] * num[:, i], (self.no_dims, 1)).T * (Z[i, :] - Z), 0) + (1-self.rho)*np.sum(np.tile(OQ[:, i] * num[:, i], (self.no_dims, 1)).T * (Z[i, :] - Z), 0)
    
            # Perform the update
            if iter < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dZ > 0.) != (iZ > 0.)) + \
                    (gains * 0.8) * ((dZ > 0.) == (iZ > 0.))
            gains[gains < min_gain] = min_gain
            iZ = momentum * iZ - eta * (gains * dZ)
            Z = Z + iZ
            Z = Z - np.tile(np.mean(Z, 0), (n, 1))
            
            # Compute current value of cost function
            if (iter + 1) % 10 == 0:
                C = self.rho*np.sum(P * np.log(P / Q)) + (1-self.rho)*np.sum(O * np.log(O / Q))
                print("Iteration %d: error is %f" % (iter + 1, C))

            # Stop lying about P-values
            if iter == 100:
                P = P / 4.
                O = O / 4.
                
        return Z
    
    def obtain_Z2_tSNE(self, X1=np.array([]), X2=np.array([]), Z1=np.array([])):
        # X1 = data matrix for points with known labels, of shape (n1, n_features)
        # X2 = data matrix for points with unknown labels, of shape (n2, n_features)
        # Z1 = low dimensional embedding for X1 obtained by St-SNE, of shape (n1, no_dims)
        """
            Runs t-SNE on X1 and X2, keeping Z1 fixed, to obtain low-dim embedding Z2 for X2
        """
        
        # Check inputs
        if isinstance(self.no_dims, float):
            print("Error: array X should have type float.")
            return -1
        if round(self.no_dims) != self.no_dims:
            print("Error: number of dimensions should be an integer.")
            return -1
        
        print("Running t-SNE to obtain Z2...")
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        X = np.vstack((X1, X2))
        (n, d) = X.shape
        max_iter = 500
        initial_momentum = 0.5
        final_momentum = 0.8
        eta = 500
        min_gain = 0.01
        
        # Compute P-values
        P = self.x2p(X, 1e-5)
        P = P + np.transpose(P)
        P = P / np.sum(P)
        P = P * 4.									# early exaggeration
        P = np.maximum(P, 1e-12)
        
        Z2 = np.random.randn(n2, self.no_dims)
        Z = np.vstack((Z1,Z2))
        dZ2 = np.zeros((n2, self.no_dims))
        iZ2 = np.zeros((n2, self.no_dims))
        gains = np.ones((n2, self.no_dims))

        # Run iterations
        for iter in range(max_iter):       

            # Compute pairwise affinities
            sum_Z = np.sum(np.square(Z), 1)
            num = -2. * np.dot(Z, Z.T)
            num = 1. / (1. + np.add(np.add(num, sum_Z).T, sum_Z))
            num[range(n), range(n)] = 0.
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            # Compute gradient
            PQ = P-Q
            for i in range(n2):
                
                dZ2[i, :] = np.sum(np.tile(PQ[:, i+n1] * num[:, i+n1], (self.no_dims, 1)).T * (Z[i+n1, :] - Z), 0)

            # Perform the update
            if iter < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dZ2 > 0.) != (iZ2 > 0.)) + \
                    (gains * 0.8) * ((dZ2 > 0.) == (iZ2 > 0.))
            gains[gains < min_gain] = min_gain
            iZ2 = momentum * iZ2 - eta * (gains * dZ2)
            Z2 = Z2 + iZ2
            
            Z = np.vstack((Z1,Z2))
            Z = Z - np.tile(np.mean(Z, 0), (n, 1))
            
            # Compute current value of cost function
            if (iter + 1) % 10 == 0:
                C = np.sum(P * np.log(P / Q))
                print("Iteration %d: error is %f" % (iter + 1, C))

            # Stop lying about P-values
            if iter == 100:
                P = P / 4.
        
        return Z2
    
    def obtain_Z_ByComment3(self, X1=np.array([]), X2=np.array([]), Y1=np.array([])):
        # X1 = data matrix for points with known labels, of shape (n1, n_features)
        # X2 = data matrix for points with unknown labels, of shape (n2, n_features)
        # Y1 = labels for X1, of shape (n1, )
        """
            Runs St-SNE on the dataset X=(X1, X2) with labels Y1 to reduce its dimensionality to no_dims dimensions. 
        """
        
        print("Implementing Comment 3 to obtain Z1 and Z2...")
        # Check inputs
        if isinstance(self.no_dims, float):
            print("Error: array X should have type float.")
            return -1
        if round(self.no_dims) != self.no_dims:
            print("Error: number of dimensions should be an integer.")
            return -1
        
        n1 = X1.shape[0]
        X = np.vstack((X1, X2))
        
        # Initialize variables
        (n, d) = X.shape
        max_iter = 500
        initial_momentum = 0.5
        final_momentum = 0.8
        eta = 500
        min_gain = 0.01
        Z = np.random.randn(n, self.no_dims)
        dZ = np.zeros((n, self.no_dims))
        iZ = np.zeros((n, self.no_dims))
        gains = np.ones((n, self.no_dims))
    
        # Compute P-values
        P = self.x2p(X, 1e-5)
        P = P + np.transpose(P)
        P = P / np.sum(P)
        P = P * 4.	# early exaggeration
        P = np.maximum(P, 1e-12)
        
        # One-hot encode labels
        Y1 = np.reshape(Y1,(n1,1))
        ohe = OneHotEncoder()
        Y1 = ohe.fit_transform(Y1)
        Y1 = Y1.toarray()
        
        # Compute O-values for Y1
        O1 = self.y2o(Y1)[0]
        O1 = O1 * 4 # early exaggeration
        O1 = np.maximum(O1, 1e-12)

        # Run iterations
        for iter in range(max_iter):       
    
            # Compute pairwise affinities
            sum_Z = np.sum(np.square(Z), 1)
            num = -2. * np.dot(Z, Z.T)
            num = 1. / (1. + np.add(np.add(num, sum_Z).T, sum_Z))
            num[range(n), range(n)] = 0.
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)
            
            Z1 = Z[0:n1,:]
            # Compute pairwise affinities considering only data points with known labels
            sum_Z1 = np.sum(np.square(Z1), 1)
            num1 = -2. * np.dot(Z1, Z1.T)
            num1 = 1. / (1. + np.add(np.add(num1, sum_Z1).T, sum_Z1))
            num1[range(n1), range(n1)] = 0.
            Q1 = num1 / np.sum(num1)
            Q1 = np.maximum(Q1, 1e-12)
    
            # Compute gradient
            PQ = P - Q
            OQ1 = O1 - Q1
            
            for i in range(n):
                
                if i < n1:
                    dZ[i, :] = self.rho*np.sum(np.tile(PQ[:, i] * num[:, i], (self.no_dims, 1)).T * (Z[i, :] - Z), 0) + (1-self.rho)*np.sum(np.tile(OQ1[:, i] * num1[:, i], (self.no_dims, 1)).T * (Z1[i, :] - Z1), 0)
                else:
                    dZ[i, :] = self.rho*np.sum(np.tile(PQ[:, i] * num[:, i], (self.no_dims, 1)).T * (Z[i, :] - Z), 0)
                        
    
            # Perform the update
            if iter < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dZ > 0.) != (iZ > 0.)) + \
                    (gains * 0.8) * ((dZ > 0.) == (iZ > 0.))
            gains[gains < min_gain] = min_gain
            iZ = momentum * iZ - eta * (gains * dZ)
            Z = Z + iZ
            Z = Z - np.tile(np.mean(Z, 0), (n, 1))
            
            # Compute current value of cost function
            if (iter + 1) % 10 == 0:
                C = self.rho*np.sum(P * np.log(P / Q)) + (1-self.rho)*np.sum(O1 * np.log(O1 / Q1))
                print("Iteration %d: error is %f" % (iter + 1, C))

            # Stop lying about P-values
            if iter == 100:
                P = P / 4.
                O1 = O1 / 4.
                
        Z1 = Z[0:n1,:]
        Z2 = Z[n1:n,:]
                
        return Z1, Z2
    
    def obtain_Y2_GradDescent(self, Z1=np.array([]), Z2=np.array([]), Y1=np.array([]), eta=0.1):
        # Z1 = low dimensional embedding for X1, of shape (n1, no_dims)
        # Z2 = low dimensional embedding for X2, of shape (n2, no_dims)
        # Y1 = labels for X1, of shape (n1, )
        # eta = learning rate
        """
            Runs gradient descent to minimize KL(O|Q) with respect to unknown labels Y2 to obtain estimates for unknown labels
        """
        
        print("Minimizing KL(O|Q) via gradient descent to obtain Y2...")
        n1 = Z1.shape[0]
        n2 = Z2.shape[0]
        n = n1+n2
        
        # One-hot encode labels
        Y1 = np.reshape(Y1,(n1,1))
        ohe = OneHotEncoder()
        Y1 = ohe.fit_transform(Y1)
        Y1 = Y1.toarray()

        # Initialize Y2 by finding nearest neighbor with known label
        num_classes = Y1.shape[1]
        Y2 = np.zeros((n2, num_classes))
        D = pairwise_distances(Z2, Z1, metric='euclidean')
        NN_idx = np.argmin(D, axis=1)
        Y2 = Y1[NN_idx,:]
        Y = np.vstack((Y1,Y2))
        
        dY2 = np.zeros((n2, num_classes))
        iY2 = np.zeros((n2, num_classes))
        max_iter = 200
        eta = eta
        
        # Compute Q
        Z = np.vstack((Z1, Z2))
        sum_Z = np.sum(np.square(Z), 1)
        num = -2. * np.dot(Z, Z.T)
        num = 1. / (1. + np.add(np.add(num, sum_Z).T, sum_Z))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        
        # Run iterations
        for iter in range(max_iter):       
            # Compute O-values
            O, O_cond = self.y2o(Y)
            O = np.maximum(O, 1e-12)
            O_cond2 = (O_cond**2 + np.transpose(O_cond**2))/(2*n)
            
            # Compute gradient
            for i in range(n2):
                
                dY2[i, :] = -8*np.sum(np.tile((1+np.log(O/Q)[:, i+n1]) * (O-O_cond2[:,i+n1])[:, i+n1], (num_classes, 1)).T * (Y[i+n1, :] - Y), 0)
              
            # Perform the update
            iY2 = - eta * dY2
            Y2 = Y2 + iY2
            
            Y = np.vstack((Y1,Y2))
            
            # Compute current value of cost function
            if (iter + 1) % 10 == 0:
                C = np.sum(O * np.log(O / Q))
                print("Iteration %d: error is %f" % (iter + 1, C))
        
        # Set max row values as 1 and others as 0
        max_value = Y2.max(axis=1).reshape(-1, 1)
        Y2 = np.where(Y2 == max_value, 1, 0)
        
        return Y2
    
    def obtain_Y2_OptimizeOverEachLabel(self, Z1=np.array([]), Z2=np.array([]), Y1=np.array([])):
        # Z1 = low dimensional embedding for X1, of shape (n1, no_dims)
        # Z2 = low dimensional embedding for X2, of shape (n2, no_dims)
        # Y1 = labels for X1, of shape (n1, )
        """
            Optimizes KL(O|Q) over each unknown label individually    
        """
        
        print("Optimizing KL(O|Q) over each variable to obtain Y2...")
        Z = np.vstack((Z1,Z2))
        n = Z.shape[0]
        n1 = Z1.shape[0]
        n2 = Z2.shape[0]
        
        # One-hot encode labels
        Y1 = np.reshape(Y1,(n1,1))
        ohe = OneHotEncoder()
        Y1 = ohe.fit_transform(Y1)
        Y1 = Y1.toarray()
        num_classes = Y1.shape[1]

        # Compute Q
        sum_Z = np.sum(np.square(Z), 1)
        num = -2. * np.dot(Z, Z.T)
        num = 1. / (1. + np.add(np.add(num, sum_Z).T, sum_Z))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        
        Y2 = np.zeros((n2, num_classes))
        
        # Run iterations
        for iter in range(n2):      
            
            L=list(range(n1,n1+n2))
            idx_delete = np.delete(L,iter)
            Q_iter = np.delete(Q, idx_delete, 0)
            Q_iter = np.delete(Q_iter, idx_delete, 1)
            Q_iter = Q_iter / Q_iter.sum()
            
            I = np.eye(num_classes, num_classes)
            C_list = []
            for i in range(num_classes):
                
                Y_current = np.vstack((Y1,I[i,:]))
                O_i = self.y2o(Y_current)[0]
                O_i = np.maximum(O_i, 1e-12)
                
                C = np.sum(O_i * np.log(O_i / Q_iter))
                C_list.append(C)
                
            best_idx = np.argmin(C_list)
            Y2[iter,:] = I[best_idx,:]
            
            progress = math.floor((iter+1)/n2*100)
            # Compute current value of cost function
            if (iter + 1) % 5 == 0 or iter==n2-1:
                print(f"Progress: {progress}% complete")
        
        return Y2
    
    def obtain_Y2_KNN(self, Z1=np.array([]), Z2=np.array([]), Y1=np.array([]), n_neighbors=5):
        # Z1 = low dimensional embedding for X1, of shape (n1, no_dims)
        # Z2 = low dimensional embedding for X2, of shape (n2, no_dims)
        # Y1 = labels for X1, of shape (n1, )
        # n_neighbors = number of neighbors to use for KNN classifier
        """
            Obtains missing labels by training a KNN classifier on the embedded data Z=(Z1, Z2)  
        """
        
        n1 = Z1.shape[0]
        
        # One-hot encode labels
        Y1 = np.reshape(Y1,(n1,1))
        ohe = OneHotEncoder()
        Y1 = ohe.fit_transform(Y1)
        Y1 = Y1.toarray()
        
        # Train classifier
        KNN = KNeighborsClassifier(n_neighbors=n_neighbors)
        KNN.fit(Z1, Y1)
        
        # Estimate unknown labels
        Y2 = KNN.predict(Z2)
        
        return Y2
    
    def score(self, Y2=np.array([]), Y2_approx=np.array([])):
        # Y2 = true labels for X2, of shape (n2, )
        # Y2_approx = estimated labels for X2, one-hot encoded, of shape (n2, num_classes)
        """
            Calculates prediction accuracy of estimated labels
        """
        
        # One-hot encode true labels
        n_test = len(Y2)
        Y2 = np.reshape(Y2, (n_test,1))
        ohe = OneHotEncoder()
        Y2 = ohe.fit_transform(Y2)
        Y2 = Y2.toarray()
        
        acc = np.sum(Y2 == Y2_approx, axis=1)/Y2.shape[1]  # todo: this is 1 - hamming loss != accuracy
        acc = np.sum(acc)/n_test
        
        return acc


def accuracy(Y2, Y2_approx):
    n_test = len(Y2)
    Y2 = np.asarray(Y2)
    Y2_pred = np.argmax(Y2_approx, axis=1)
    acc = np.sum(Y2 == Y2_pred) / n_test
    return acc


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.datasets import load_digits
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    
    
    # Load data
    # data = load_breast_cancer()
    data = load_digits()
    X = data.data
    #pca = PCA(n_components=50) # Preprocess data so n_features <=50
    #X = pca.fit_transform(X).astype(np.float32, copy=False)
    Y = data.target
    (n, d) = X.shape
    indices = list(range(n))
    
    no_dims=2
    perplexity=30.0
    rho=0.5
    
    # Split data into train/test sets
    X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(X, Y, indices, test_size=0.2, random_state=42)
    
    # Size of test set
    n2 = X_test.shape[0]

    
    St_SNE = St_SNE(no_dims=no_dims, perplexity=perplexity, rho=rho)
    
    
    # Obtain embedding, tSt-SNE
    Z1 = St_SNE.transform(X_train, Y_train)
    Z2 = St_SNE.obtain_Z2_tSNE(X_train, X_test, Z1)
    Z=np.zeros((n,no_dims))
    Z[indices_train,:] = Z1
    Z[indices_test,:] = Z2

    # Plot embedding
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    plt.scatter(Z[:,0], Z[:,1], c=Y, cmap=cmap)
    plt.title('St-SNE')
    plt.show()
    
    Y2 = St_SNE.obtain_Y2_OptimizeOverEachLabel(Z1, Z2, Y_train)
    acc1 = St_SNE.score(Y_test, Y2)
    acc11 = accuracy(Y_test, Y2)
    
    Y2 = St_SNE.obtain_Y2_KNN(Z1, Z2, Y_train, n_neighbors=5)
    acc2 = St_SNE.score(Y_test, Y2)
    acc12 = accuracy(Y_test, Y2)
    
    print(f"Accuracy of Y2 after optimizing KL(O|Q) over each label = {acc1}, {acc11}")
    print(f"Accuracy of Y2 using KNN = {acc2}, {acc12}")
    
    
    # Obtain embedding, Comment 3
    (Z1, Z2) = St_SNE.obtain_Z_ByComment3(X_train, X_test, Y_train)
    Z=np.zeros((n,no_dims))
    Z[indices_train,:] = Z1
    Z[indices_test,:] = Z2
    
    # Plot embedding
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    plt.scatter(Z[:,0], Z[:,1], c=Y, cmap=cmap)
    plt.title('St-SNE')
    plt.show()
    
    Y2 = St_SNE.obtain_Y2_OptimizeOverEachLabel(Z1, Z2, Y_train)
    acc1 = St_SNE.score(Y_test, Y2)
    acc11 = accuracy(Y_test, Y2)
    
    Y2 = St_SNE.obtain_Y2_KNN(Z1, Z2, Y_train, n_neighbors=5)
    acc2 = St_SNE.score(Y_test, Y2)
    acc12 = accuracy(Y_test, Y2)
    
    print(f"Accuracy of Y2 after optimizing KL(O|Q) over each label = {acc1}, {acc11}")
    print(f"Accuracy of Y2 using KNN = {acc2}, {acc12}")
