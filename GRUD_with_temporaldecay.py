import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import warnings


## handles temporal decay based on delta
class TemporalDecay(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, diag=False, **kwargs):   #weight initialization
        super(TemporalDecay, self).__init__(**kwargs)
        self.diag = diag
        self.input_size = input_size
        self.output_size = output_size
        self.W = self.add_weight(shape=(output_size, input_size), initializer='uniform', trainable=True) # can use other initializers
        self.b = self.add_weight(shape=(output_size,), initializer='uniform', trainable=True)
        
        # print(f"Shape of self.W: {self.W.shape}")
        # print(f"Shape of self.b: {self.b.shape}")


        if self.diag:
            assert input_size == output_size
            self.m = tf.eye(input_size)   # diagonal matrix
            
     ## Applies the temporal decay to the input delta values and computes the decay factor gamma.
    def call(self, delta):        
        print(f"Shape of delta: {delta.shape}")
        
        print(f"Shape of self.W: {self.W.shape}")
        print(f"Shape of self.b: {self.b.shape}")
        
        if self.diag:
            gamma = tf.nn.relu(tf.matmul(delta, self.W * self.m) + self.b)
        else:
            gamma = tf.nn.relu(tf.matmul(delta, self.W) + self.b)
            
        print(f"Shape of gamma: {gamma.shape}")
        gamma = tf.exp(-gamma)
        return gamma
    

 


class GRUDModel(tf.keras.Model):
    
    ## Setting  up the GRU layer, dense layer, and temporal decay layers
    def __init__(self, n_steps, n_features, rnn_hidden_size, n_classes):
        super(GRUDModel, self).__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        
        # Define GRU layer(handle sequential data) and Dense layer(for classification)
        self.gru = layers.GRU(self.rnn_hidden_size, return_sequences=True, return_state=False)
        self.dense = layers.Dense(self.n_classes, activation='sigmoid' if n_classes == 1 else 'softmax')
        
        # Define Temporal Decay layers
        self.temporal_decay_h = TemporalDecay(n_features, rnn_hidden_size, diag=False)
        self.temporal_decay_x = TemporalDecay(n_features, n_features, diag=True)
        
    ## implement GRUD forward pass, compute tempral decay, update hidden states and classify
    def call(self, inputs, training=False):
        X, X_filledLOCF, missing_mask, deltas, empirical_mean = inputs
        
        # Compute temporal decay factors
        gamma_h = self.temporal_decay_h(deltas)
        gamma_x = self.temporal_decay_x(deltas)
        
        # Initialize hidden state
        hidden_state = tf.zeros((X.shape[0], self.rnn_hidden_size))
        
        representation_collector = []
        for t in range(self.n_steps):
            x = X[:, t, :]  # Values
            m = missing_mask[:, t, :]  # Mask
            d = deltas[:, t, :]  # Delta, time gap
            x_filledLOCF = X_filledLOCF[:, t, :]

            # Apply temporal decay
            hidden_state = hidden_state * gamma_h[:, t, :]
            representation_collector.append(hidden_state)

            x_h = gamma_x[:, t, :] * x_filledLOCF + (1 - gamma_x[:, t, :]) * empirical_mean
            x_replaced = m * x + (1 - m) * x_h
            data_input = tf.concat([x_replaced, hidden_state, m], axis=1)
            
            # Update hidden state with GRU
            hidden_state = self.gru(data_input, initial_state=hidden_state)

        representation_collector = tf.stack(representation_collector, axis=1)

        # Dense layer for classification
        logits = self.dense(hidden_state)
        
        return logits  # probabilities for classification



class GRUD:
    
    # initialize model, sets loss function and optimizer
    def __init__(self, n_steps, n_features, n_classes, rnn_hidden_size, batch_size=32, epochs=100):
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Print the values directly
        print(f"Value of n_steps: {n_steps}")
        print(f"Value of n_features: {n_features}")
        print(f"Value of rnn_hidden_size: {rnn_hidden_size}")
        print(f"Value of batch_size: {batch_size}")
        

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
        
        
        
        
        
        

