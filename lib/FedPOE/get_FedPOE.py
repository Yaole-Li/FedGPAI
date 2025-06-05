import tensorflow as tf
from tensorflow import keras
from lib.FedPOE.FedPOE_image_classification import FedPOE_classification
from lib.FedPOE.FedPOE_regression import FedPOE_regression

def get_FedPOE(model, args):
    if args.task=="classification":
        model_fed = tf.keras.models.clone_model(model)
        model_fed.set_weights(model.get_weights())
        model_fed.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        federated = FedPOE_classification(model_fed, args.eta, args.period)
        
        local = []
        for i in range(args.num_clients):
            model_loc = tf.keras.models.clone_model(model)
            model_loc.set_weights(model.get_weights())
            model_loc.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            local.append(FedPOE_classification(model_loc, args.eta, args.period))
    elif args.task=="regression":
        federated = FedPOE_regression(args.regularizer, model, args.eta, args.num_clients)
        local = []
        for i in range(args.num_clients):
            local.append(FedPOE_regression(args.regularizer, model, args.eta, args.num_clients))
    return local, federated