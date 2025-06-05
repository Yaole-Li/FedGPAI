import numpy as np
import argparse
from lib.datasets.data_loader import data_loader
from lib.FedPOE.get_FedPOE import get_FedPOE

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default='Air', type=str)
parser.add_argument("--task", default='regression', type=str)
parser.add_argument("--num_clients", default=400, type=int)
parser.add_argument("--num_samples", default=250, type=int)
parser.add_argument("--num_random_features", default=100, type=int)
parser.add_argument("--regularizer", default=1e-6, type=float)

args = parser.parse_args()

X, Y = data_loader(args)
    
M, N = X[0].shape
K = args.num_clients
M*=K
    
gamma = []
num_rbf = 3

for i in range(num_rbf):
    gamma.append(10**(i-1))
gamma = np.array(gamma)

n_components = args.num_random_features
w = np.ones((K, np.prod(gamma.shape)))
w_loc = np.ones((K, np.prod(gamma.shape)))
a = np.ones((K,1))
b = np.ones((K,1))

args.eta = 1/np.sqrt(args.num_samples)

mse = np.zeros((args.num_samples,1))
m = np.zeros((K,args.num_samples,20))

for cc in range(0,20):
    # Generating Random Features
    ran_feature = np.zeros((N,n_components,gamma.shape[0]))
    for i in range(num_rbf):
        ran_feature[:,:,i] = np.random.randn(N,n_components)*np.sqrt(1/gamma[i])
    
    alg_loc, alg = get_FedPOE(ran_feature ,args)
    e = np.zeros((args.num_samples,K))
    for i in range(args.num_samples):
        agg_grad = []
        agg_kernel_indices = []
        for j in range(K):
            f_RF_fed, f_RF_p, X_features = alg.predict(X[j][i:i+1,:],w[j:j+1,:])
            w[j:j+1,:], local_grad = alg.local_update(f_RF_p, Y[j][i], w[j:j+1,:], X_features)
            f_RF_loc, f_RF_p, X_features = alg_loc[j].predict(X[j][i:i+1,:],w_loc[j:j+1,:])
            w_loc[j:j+1,:], local_grad_loc = alg_loc[j].local_update(f_RF_p, Y[j][i], w_loc[j:j+1,:], X_features)
            f_RF = (a[j,0]*f_RF_fed + b[j,0]*f_RF_loc)/(a[j,0]+b[j,0])
            l_fed, l_loc = (f_RF_fed-Y[j][i])**2, (f_RF_loc-Y[j][i])**2
            a[j,0]*=np.exp(-args.eta*l_fed)
            b[j,0]*=np.exp(-args.eta*l_loc)
            alg_loc[j].global_update([local_grad_loc])
            agg_grad.append(local_grad)
            m[j, i, cc] = (f_RF-Y[j][i])**2
            if i==0:
                e[i,j] = (f_RF-Y[j][0])**2
            else:
                e[i,j] = ( 1/(i+1) )*( (i*e[i-1,j])+((f_RF-Y[j][i])**2) )
        alg.global_update(agg_grad)
    mse = ( 1/(cc+1) )*( (cc*mse)+np.reshape(np.mean(e,axis=1),(-1,1)) )
    
print('MSE of Fed-POE is %s' %mse[-1])
print('Standard deviation of Fed-POE is %s' %np.std(np.mean(np.mean(m, axis=2), axis=1)))