import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops

# building the model with the vectorize layer and the embedding layer
def define_model(EMBEDDING_SIZE, MAX_TOKENS, MAX_SEQ_LEN):

    # Input layer that has an attention layer
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention
    # Variable-length int sequences.
    query_input = tf.keras.Input(shape=(None,), dtype="float32")
    value_input = tf.keras.Input(shape=(None,), dtype="float32")

    # adding an embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        MAX_TOKENS, EMBEDDING_SIZE, input_length=MAX_SEQ_LEN
    )

    # Query embeddings of shape [batch_size, Tq, dimension].
    query_embeddings = embedding_layer(query_input)
    # Value embeddings of shape [batch_size, Tv, dimension].
    value_embeddings = embedding_layer(value_input)

    # CNN layer.
    cnn_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        # Use 'same' padding so outputs have the same shape as inputs.
        padding="same",
    )
    # Query encoding of shape [batch_size, Tq, filters].
    query_seq_encoding = cnn_layer(query_embeddings)
    # Value encoding of shape [batch_size, Tv, filters].
    value_seq_encoding = cnn_layer(value_embeddings)

    # Query-value attention of shape [batch_size, Tq, filters].
    query_value_attention_seq = tf.keras.layers.Attention()(
        [query_seq_encoding, value_seq_encoding]
    )

    # Reduce over the sequence axis to produce encodings of shape
    # [batch_size, filters].
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_seq_encoding)
    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
        query_value_attention_seq
    )

    # Concatenate query and document encodings to produce a DNN input layer.
    input_layer = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])

    cells = [tf.keras.layers.LSTMCell(256), tf.keras.layers.LSTMCell(64)]
    rnn = tf.keras.layers.RNN(cells)

    # lstm_1 = tf.keras.layers.LSTM(100)
    attention = tf.keras.layers.Attention(name="attention_weight")
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    model = tf.keras.Sequential([input_layer, rnn, attention, output_layer])

    return model
