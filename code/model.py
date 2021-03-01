import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops

# building the model with the vectorize layer and the embedding layer
def define_model(vectorize_layer, embedding_layer):
    # Build the model
    cells = [tf.keras.layers.LSTMCell(256), tf.keras.layers.LSTMCell(64)]
    rnn = tf.keras.layers.RNN(cells)
    output_layer = tf.keras.layers.Dense(1)

    model = tf.keras.Sequential([vectorize_layer, embedding_layer, rnn, output_layer])

    return model
