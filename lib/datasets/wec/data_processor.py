import numpy as np
from numpy import linalg as LA
import pandas as pd

def data_processing_wec(args):
    data_Adelaide = pd.read_csv('lib/datasets/wec/Adelaide_Data.csv', delimiter=',')
    data_Adelaide = data_Adelaide.dropna()
    data_Adelaide = data_Adelaide.to_numpy()
    data_Adelaide = data_Adelaide[:25000,:]

    data_Perth = pd.read_csv('lib/datasets/wec/Perth_Data.csv', delimiter=',')
    data_Perth = data_Perth.dropna()
    data_Perth = data_Perth.to_numpy()
    data_Perth = data_Perth[:25000,:]

    data_Sydney = pd.read_csv('lib/datasets/wec/Sydney_Data.csv', delimiter=',')
    data_Sydney = data_Sydney.dropna()
    data_Sydney = data_Sydney.to_numpy()
    data_Sydney = data_Sydney[:25000,:]

    data_Tasmania = pd.read_csv('lib/datasets/wec/Tasmania_Data.csv', delimiter=',')
    data_Tasmania = data_Tasmania.dropna()
    data_Tasmania = data_Tasmania.to_numpy()
    data_Tasmania = data_Tasmania[:25000,:]

    K = args.num_clients

    maj_n = int(.7*args.num_samples)
    min_n = int(.1*args.num_samples)

    X = {}
    Y = {}
    for i in range(int(K/4)):
        client_data = data_Adelaide[maj_n*i:maj_n*(i+1),:]
        client_data = np.concatenate((client_data, data_Perth[maj_n*int(K/4)+min_n*i:maj_n*int(K/4)+min_n*(i+1),:]), axis=0)
        client_data = np.concatenate((client_data, data_Sydney[maj_n*int(K/4)+min_n*i:maj_n*int(K/4)+min_n*(i+1),:]), axis=0)
        client_data = np.concatenate((client_data, data_Tasmania[maj_n*int(K/4)+min_n*i:maj_n*int(K/4)+min_n*(i+1),:]), axis=0)
        np.random.shuffle(client_data)
        X[i] = client_data[:,:-1]
        Y[i] = client_data[:,-1]

        client_data = data_Perth[maj_n*i:maj_n*(i+1),:]
        client_data = np.concatenate((client_data, data_Adelaide[maj_n*int(K/4)+min_n*i:maj_n*int(K/4)+min_n*(i+1),:]), axis=0)
        client_data = np.concatenate((client_data, \
                                      data_Sydney[(maj_n + min_n)*int(K/4)+min_n*i:(maj_n + min_n)*int(K/4)+min_n*(i+1),:]), axis=0)
        client_data = np.concatenate((client_data, \
                                      data_Tasmania[(maj_n + min_n)*int(K/4)+min_n*i:(maj_n + min_n)*int(K/4)+min_n*(i+1),:]), \
                                     axis=0)
        np.random.shuffle(client_data)
        X[int(K/4)+i] = client_data[:,:-1]
        Y[int(K/4)+i] = client_data[:,-1]

        client_data = data_Sydney[maj_n*i:maj_n*(i+1),:]
        client_data = np.concatenate((client_data, \
                                      data_Adelaide[(maj_n+min_n)*int(K/4)+min_n*i:(maj_n+min_n)*int(K/4)+min_n*(i+1),:]), axis=0)
        client_data = np.concatenate((client_data, \
                                      data_Perth[(maj_n+min_n)*int(K/4)+min_n*i:(maj_n+min_n)*int(K/4)+min_n*(i+1),:]), axis=0)
        client_data = np.concatenate((client_data, \
                                      data_Tasmania[(maj_n+2*min_n)*int(K/4)+min_n*i:(maj_n+2*min_n)*int(K/4)+min_n*(i+1),:]), \
                                     axis=0)
        np.random.shuffle(client_data)
        X[2*int(K/4)+i] = client_data[:,:-1]
        Y[2*int(K/4)+i] = client_data[:,-1]

        client_data = data_Tasmania[maj_n*i:maj_n*(i+1),:]
        client_data = np.concatenate((client_data, \
                                      data_Adelaide[(maj_n+2*min_n)*int(K/4)+min_n*i:(maj_n+2*min_n)*int(K/4)+min_n*(i+1),:]), \
                                     axis=0)
        client_data = np.concatenate((client_data, \
                                      data_Perth[(maj_n+2*min_n)*int(K/4)+min_n*i:(maj_n+2*min_n)*int(K/4)+min_n*(i+1),:]), \
                                     axis=0)
        client_data = np.concatenate((client_data, \
                                      data_Sydney[(maj_n+2*min_n)*int(K/4)+min_n*i:(maj_n+2*min_n)*int(K/4)+min_n*(i+1),:]), \
                                     axis=0)
        np.random.shuffle(client_data)
        X[3*int(K/4)+i] = client_data[:,:-1]
        Y[3*int(K/4)+i] = client_data[:,-1]

    X_norm_max_clients = np.zeros((1,K))
    Y_max_clients = np.zeros((1,K))
    Y_min_clients = np.zeros((1,K))
    for i in range(K):
        X_n = np.zeros((1,args.num_samples))
        for j in range(args.num_samples):
            X_n[0,j] = LA.norm(X[i][j,:])
        X_norm_max_clients[0,i] = np.max(X_n)
        Y_max_clients[0,i] = np.max(Y[i])
        Y_min_clients[0,i] = np.min(Y[i])
    X_norm_max = np.max(X_norm_max_clients)
    Y_max = np.max(Y_max_clients)
    Y_min = np.min(Y_min_clients)

    for i in range(K):
        for j in range(args.num_samples):
            X[i][j,:] = X[i][j,:]/X_norm_max
        Y[i] = (Y[i] - Y_min)/(Y_max - Y_min)
    return X, Y