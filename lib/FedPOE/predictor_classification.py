import numpy as np
import tensorflow as tf
from tensorflow import keras

class predictor_classification:
    def __init__(self, learning_rate, num_models, period):
        self.w = [1]
        self.eta = learning_rate
        self.M = num_models
        self.period = period
        self.a = 1
        self.b = 1
        self.c = 1
        self.d = 1
    
    def model_selection(self):
        total = []
        s = 0
        for weight in self.w:
            s+=weight
            total.append(s)
        indices = []
        for __ in range(self.M):
            n = np.random.rand()*total[-1]
            l = 0
            r = len(self.w)
            while l<r:
                mid = (l+r)//2
                if n>total[mid]:
                    l = mid+1
                else:
                    r = mid
            if l not in indices:
                indices.append(l)
        return indices
    
    def prediction(self, X_new, Y_new, models, indices, t):
        total = sum(self.w)
        weights = []
        for i in indices:
            weights.append(self.w[i])
        total_weights = sum(weights)
        en_pre = (weights[0]/total_weights)*models[2](X_new).numpy()
        for i in range(1,len(weights)):
            en_pre+=(weights[i]/total_weights)*models[i+2](X_new).numpy()
        en_f = (self.a/(self.a+self.b))*models[0](X_new).numpy() + (self.b/(self.a+self.b))*models[1](X_new).numpy()
        out = (self.c/(self.c+self.d))*en_f + (self.d/(self.c+self.d))*en_pre
        prediction = np.argmax(out)
        y_true = tf.convert_to_tensor(Y_new, dtype=tf.int32)
        y_pred = tf.convert_to_tensor(out, dtype=tf.float32)
        y_en_pre = tf.convert_to_tensor(en_pre, dtype=tf.float32)
        y_en_f = tf.convert_to_tensor(en_f, dtype=tf.float32)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        loss_pred = loss_fn(y_true, y_pred).numpy()
        loss_en_pre = loss_fn(y_true, y_en_pre).numpy()
        loss_en_f = loss_fn(y_true, y_en_f).numpy()
        for i, ind in enumerate(indices):
            loss, acc = models[i+2].evaluate(X_new, Y_new, verbose=0)
            p = self.w[ind]/total
            loss/=(1-(1-p)**self.M)
            self.w[ind]*=np.exp(-self.eta*loss)
        loss_loc, acc_loc = models[0].evaluate(X_new, Y_new, verbose=0)
        self.a*=np.exp(-self.eta*loss_loc)
        loss_fed, acc_fed = models[1].evaluate(X_new, Y_new, verbose=0)
        self.b*=np.exp(-self.eta*loss_fed)
        self.c*=np.exp(-self.eta*loss_en_f)
        self.d*=np.exp(-self.eta*loss_en_pre)
        if t==0:
            self.X = X_new
            self.Y = Y_new
        else:
            self.X = np.append(self.X, X_new, axis=0)
            self.Y = np.append(self.Y, Y_new, axis=0)
        if t>0 and t%self.period==0:
            new_w = 1
            loss, acc = models[1].evaluate(self.X, self.Y, verbose=0)
            loss*=self.X.shape[0]
            new_w*=np.exp(-self.eta*loss)
            self.w.append(new_w)
        return prediction