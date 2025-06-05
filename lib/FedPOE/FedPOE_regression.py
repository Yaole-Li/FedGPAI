import numpy as np
from numpy import linalg as LA
            
class FedPOE_regression:
    def __init__(self, lam, rf_feature, eta, num_clients):
        self.lam = lam
        self.rf_feature = np.array(rf_feature)
        self.eta = eta
        self.theta = np.zeros((2*rf_feature.shape[1],rf_feature.shape[2]))
        self.num_kernels = rf_feature.shape[2]
        self.num_clients = num_clients
        
    def predict(self, X, w):
        M, N = X.shape
        a, n_components, b = self.rf_feature.shape
        X_f = np.zeros((b,n_components))
        X_features = np.zeros((b,2*n_components))
        f_RF_p = np.zeros((b,1))
        for j in range(0,b):
            X_f[j,:] = X.dot(self.rf_feature[:,:,j])
        X_features = (1/np.sqrt(n_components))*np.concatenate((np.sin(X_f),np.cos(X_f)),axis=1)
        for j in range(0,b):
            f_RF_p[j,0] = X_features[j,:].dot(self.theta[:,j])
        w_bar = w/np.sum(w)
        f_RF = w_bar.dot(f_RF_p)
        return f_RF, f_RF_p, X_features
    
    def local_update(self, f_RF_p, Y, w, X_features):
        b, n_components = X_features.shape
        l = np.zeros((1,self.num_kernels))
        local_grad = np.zeros((n_components, self.num_kernels))
        for j in range(self.num_kernels):
            l[0,j] = (f_RF_p[j,0]-Y)**2+self.lam*(LA.norm(self.theta[:,j])**2)
            w[0,j] = w[0,j]*np.exp(-self.eta*l[0,j])
            local_grad[:,j] = self.eta*( (2*(f_RF_p[j,0] -Y)*np.transpose(X_features[j,:]))+2*self.lam*self.theta[:,j] )
        return w, local_grad
            
    def global_update(self, agg_grad):
        theta_update = np.zeros(self.theta.shape)
        for i in range(len(agg_grad)):
            for j in range(self.num_kernels):
                theta_update[:,j]+=(agg_grad[i][:,j]/len(agg_grad))
        for i in range(self.num_kernels):
            self.theta[:,i]-=theta_update[:,i]