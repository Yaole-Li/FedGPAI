import numpy as np
from numpy import linalg as LA
import pandas as pd

def data_processing_air(args):
    data_Aotizhongxin = pd.read_csv('lib/datasets/air/PRSA_Data_Aotizhongxin.csv', delimiter=',')
    data_Aotizhongxin = data_Aotizhongxin.dropna()
    wind_direction = data_Aotizhongxin.wd.unique()
    for i in range(len(wind_direction)):
        data_Aotizhongxin = data_Aotizhongxin.replace(wind_direction[i], i).infer_objects(copy=False)
    data_Aotizhongxin = data_Aotizhongxin.to_numpy()
    data_Aotizhongxin = data_Aotizhongxin[:,2:-1]
    data_Aotizhongxin = data_Aotizhongxin[:25000,:].astype('float32')

    data_Changping = pd.read_csv('lib/datasets/air/PRSA_Data_Changping.csv', delimiter=',')
    data_Changping = data_Changping.dropna()
    wind_direction = data_Changping.wd.unique()
    for i in range(len(wind_direction)):
        data_Changping = data_Changping.replace(wind_direction[i], i).infer_objects(copy=False)
    data_Changping = data_Changping.to_numpy()
    data_Changping = data_Changping[:,2:-1]
    data_Changping = data_Changping[:25000,:].astype('float32')

    data_Dingling = pd.read_csv('lib/datasets/air/PRSA_Data_Dingling.csv', delimiter=',')
    data_Dingling = data_Dingling.dropna()
    wind_direction = data_Dingling.wd.unique()
    for i in range(len(wind_direction)):
        data_Dingling = data_Dingling.replace(wind_direction[i], i).infer_objects(copy=False)
    data_Dingling = data_Dingling.to_numpy()
    data_Dingling = data_Dingling[:,2:-1]
    data_Dingling = data_Dingling[:25000,:].astype('float32')

    data_Dongsi = pd.read_csv('lib/datasets/air/PRSA_Data_Dongsi.csv', delimiter=',')
    data_Dongsi = data_Dongsi.dropna()
    wind_direction = data_Dongsi.wd.unique()
    for i in range(len(wind_direction)):
        data_Dongsi = data_Dongsi.replace(wind_direction[i], i).infer_objects(copy=False)
    data_Dongsi = data_Dongsi.to_numpy()
    data_Dongsi = data_Dongsi[:,2:-1]
    data_Dongsi = data_Dongsi[:25000,:].astype('float32')

    K = args.num_clients

    maj_n = int(.7*args.num_samples)
    min_n = int(.1*args.num_samples)

    X = {}
    Y = {}
    for i in range(int(K/4)):
        client_data = data_Aotizhongxin[maj_n*i:maj_n*(i+1),:]
        client_data = np.concatenate((client_data, data_Changping[maj_n*int(K/4)+min_n*i:maj_n*int(K/4)+min_n*(i+1),:]), axis=0)
        client_data = np.concatenate((client_data, data_Dingling[maj_n*int(K/4)+min_n*i:maj_n*int(K/4)+min_n*(i+1),:]), axis=0)
        client_data = np.concatenate((client_data, data_Dongsi[maj_n*int(K/4)+min_n*i:maj_n*int(K/4)+min_n*(i+1),:]), axis=0)
        np.random.shuffle(client_data)
        X[i] = np.concatenate((client_data[:,:6],client_data[:,7:]),axis=1)
        Y[i] = client_data[:,6]

        client_data = data_Changping[maj_n*i:maj_n*(i+1),:]
        client_data = np.concatenate((client_data, data_Aotizhongxin[maj_n*int(K/4)+min_n*i:maj_n*int(K/4)+min_n*(i+1),:]), axis=0)
        client_data = np.concatenate((client_data, \
                                      data_Dingling[(maj_n + min_n)*int(K/4)+min_n*i:(maj_n + min_n)*int(K/4)+min_n*(i+1),:]), \
                                     axis=0)
        client_data = np.concatenate((client_data, \
                                      data_Dongsi[(maj_n + min_n)*int(K/4)+min_n*i:(maj_n + min_n)*int(K/4)+min_n*(i+1),:]), axis=0)
        np.random.shuffle(client_data)
        X[int(K/4)+i] = np.concatenate((client_data[:,:6],client_data[:,7:]),axis=1)
        Y[int(K/4)+i] = client_data[:,6]

        client_data = data_Dingling[maj_n*i:maj_n*(i+1),:]
        client_data = \
        np.concatenate((client_data,data_Aotizhongxin[(maj_n+min_n)*int(K/4)+min_n*i:(maj_n+min_n)*int(K/4)+min_n*(i+1),:]), axis=0)
        client_data = \
        np.concatenate((client_data, data_Changping[(maj_n+min_n)*int(K/4)+min_n*i:(maj_n+min_n)*int(K/4)+min_n*(i+1),:]), axis=0)
        client_data = \
        np.concatenate((client_data, data_Dongsi[(maj_n+2*min_n)*int(K/4)+min_n*i:(maj_n+2*min_n)*int(K/4)+min_n*(i+1),:]), axis=0)
        np.random.shuffle(client_data)
        X[2*int(K/4)+i] = np.concatenate((client_data[:,:6],client_data[:,7:]),axis=1)
        Y[2*int(K/4)+i] = client_data[:,6]

        client_data = data_Dongsi[maj_n*i:maj_n*(i+1),:]
        client_data = np.concatenate((client_data, \
                                      data_Aotizhongxin[(maj_n+2*min_n)*int(K/4)+min_n*i:(maj_n+2*min_n)*int(K/4)+min_n*(i+1),:]), \
                                     axis=0)
        client_data = np.concatenate((client_data, \
                                      data_Changping[(maj_n+2*min_n)*int(K/4)+min_n*i:(maj_n+2*min_n)*int(K/4)+min_n*(i+1),:]), \
                                     axis=0)
        client_data = np.concatenate((client_data, \
                                      data_Dingling[(maj_n+2*min_n)*int(K/4)+min_n*i:(maj_n+2*min_n)*int(K/4)+min_n*(i+1),:]), \
                                     axis=0)
        np.random.shuffle(client_data)
        X[3*int(K/4)+i] = np.concatenate((client_data[:,:6],client_data[:,7:]),axis=1)
        Y[3*int(K/4)+i] = client_data[:,6]

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