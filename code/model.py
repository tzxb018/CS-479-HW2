import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops

# building the model with the vectorize layer and the embedding layer
def define_model(EMBEDDING_SIZE, MAX_TOKENS, MAX_SEQ_LEN):

    # adding an embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        MAX_TOKENS, EMBEDDING_SIZE, input_length=MAX_SEQ_LEN
    )
    cells = [tf.keras.layers.LSTMCell(256), tf.keras.layers.LSTMCell(64)]
    rnn = tf.keras.layers.RNN(cells)
    lstm_1 = tf.keras.layers.LSTM(100)
    attention = tf.keras.layers.Attention()
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    model = tf.keras.Sequential([embedding_layer, rnn, attention, output_layer])

    return model
