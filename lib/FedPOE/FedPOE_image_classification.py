import numpy as np
import tensorflow as tf
from tensorflow import keras
        
class FedPOE_classification:
    def __init__(self, model, learning_rate, period):
        self.model = model
        self.lr = learning_rate
        cloned_model = tf.keras.models.clone_model(model)
        cloned_model.set_weights(model.get_weights())
        cloned_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.dic = [cloned_model]
        self.period = period
        
    def client_update(self, X_batch, Y_batch, num_epochs):
        local_model = self.model
        client_batch_size = np.prod(Y_batch.shape)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_batch, Y_batch))
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        optimizer = keras.optimizers.SGD(learning_rate=self.lr)
        for epoch in range(num_epochs):
            with tf.GradientTape() as tape:
                logits = local_model(X_batch, training=True)
                loss_value = loss_fn(Y_batch, logits)
            grads = tape.gradient(loss_value, local_model.trainable_weights[-2:])
            optimizer.apply_gradients(zip(grads, local_model.trainable_weights[-2:]))
        return local_model
    
    def update(self, aggregated_models, t):
        num_layers = len(self.model.get_weights())
        global_weights = []
        for layer in range(num_layers):
            layer_weights = np.array([local_model.get_weights()[layer] for local_model in aggregated_models])
            avg_layer_weights = np.mean(layer_weights, axis=0)
            global_weights.append(avg_layer_weights)
        self.model.set_weights(global_weights)
        if (t+1)%self.period==0:
            cloned_model = tf.keras.models.clone_model(self.model)
            cloned_model.set_weights(self.model.get_weights())
            cloned_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            self.dic.append(cloned_model)