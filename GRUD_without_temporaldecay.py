import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import warnings

### PYPOTS model code doesn't use any temporal decay



class GRUDModel(tf.keras.Model):
    def __init__(self, n_steps, n_features, rnn_hidden_size, n_classes):
        super(GRUDModel, self).__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        
        self.grud_layer = layers.GRU(self.rnn_hidden_size, return_sequences=False)
        self.dense_layer = layers.Dense(self.n_classes, activation='sigmoid' if n_classes == 1 else 'softmax')
        
    
    # def gamma(self, deltas):
    #     gamma = tf.exp(-tf.maximum(0.0, deltas))
    #     return gamma

    # def beta(self, deltas):
    #     beta = tf.exp(-tf.maximum(0.0, deltas))
    #     return beta
    
    
    def call(self, inputs, training=False):
        
        X, X_filledLOCF, missing_mask, deltas, empirical_mean = inputs   # inputs: actualdata,masking,delta,empirical mean
        x = self.grud_layer(X)
        x = self.dense_layer(x)
        
        return x


class GRUD:
    def __init__(self, n_steps, n_features, n_classes, rnn_hidden_size, batch_size=32, epochs=100):
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.epochs = epochs

        self.model = GRUDModel(n_steps, n_features, rnn_hidden_size, n_classes)
        loss_fn = 'binary_crossentropy' if n_classes == 1 else 'sparse_categorical_crossentropy'
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_fn, metrics=['accuracy'])

    def fit(self, train_data, train_labels, val_data=None, val_labels=None):
        if val_data is not None and val_labels is not None:
            self.model.fit(
                train_data, 
                train_labels, 
                validation_data=(val_data, val_labels),
                epochs=self.epochs, 
                batch_size=self.batch_size
            )
        else:
            self.model.fit(train_data, train_labels, epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, test_data):
        predictions = self.model.predict(test_data, batch_size=self.batch_size)
        return predictions  # probabilities of smaples belonging to class 0 and class 1

    def classify(self, test_data):
        predictions = self.predict(test_data)
        if self.n_classes == 1:
            return (predictions > 0.5).astype(int)
        else:
            return np.argmax(predictions, axis=1)

