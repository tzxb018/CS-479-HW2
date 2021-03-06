import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
from tensorflow.keras.layers import *


# building the model with the vectorize layer and the embedding layer
def define_model(EMBEDDING_SIZE, MAX_TOKENS, MAX_SEQ_LEN, DROPOUT_RATE):

    # adding an embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        MAX_TOKENS, EMBEDDING_SIZE, input_length=MAX_SEQ_LEN
    )

    # cells = [tf.keras.layers.LSTMCell(256), tf.keras.layers.LSTMCell(64)]
    # rnn = tf.keras.layers.RNN(cells)

    # attention_in = tf.keras.layers.LSTM(
    #     100, return_sequences=True, dropout=DROPOUT_RATE
    # )(embedding_layer)

    # attention_out = tf.keras.layers.Attention()(attention_in)

    lstm_1 = tf.keras.layers.LSTM(100)
    dropout = tf.keras.layers.Dropout(DROPOUT_RATE)
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    model = tf.keras.Sequential([embedding_layer, lstm_1, dropout, output_layer])

    return model


def define_rnn(MAX_TOKENS, MAX_SEQ_LEN, DROPOUT_RATE, REG_CONSTANT, BATCH_SIZE, ds):

    input_layer = tf.keras.Input(shape=(), dtype=tf.string)
    # Create TextVectorization layer
    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=MAX_TOKENS, output_mode="int", output_sequence_length=MAX_SEQ_LEN
    )

    # Use `adapt` to create a vocabulary mapping words to integers
    train_text = ds["train"].map(lambda x: x["text"])
    vectorize_layer.adapt(train_text)

    # Getting our dimensions
    VOCAB_SIZE = len(vectorize_layer.get_vocabulary())
    EMBEDDING_SIZE = int(np.sqrt(VOCAB_SIZE))
    print(
        "Vocab size is {} and is embedded into {} dimensions".format(
            VOCAB_SIZE, EMBEDDING_SIZE
        )
    )

    # Getting our embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        VOCAB_SIZE, EMBEDDING_SIZE, name="embedding"
    )

    # Creating sequential model for preprocessing (because I don't know how to do it functionally)
    preprocesssing_model = tf.keras.Sequential([vectorize_layer, embedding_layer])

    preproessing_output = preprocesssing_model(input_layer)

    # TRYING TO COMBINE preprocessing_model with actual model

    # LSMT layer
    lstm_out = tf.keras.layers.LSTM(
        100,
        return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.l2(REG_CONSTANT),
        recurrent_regularizer=tf.keras.regularizers.l2(REG_CONSTANT),
        bias_regularizer=tf.keras.regularizers.l2(REG_CONSTANT),
    )(
        preproessing_output
    )  # I don't know what to put here...
    lstm_out = tf.keras.layers.Dropout(DROPOUT_RATE)(lstm_out)

    # attention layer block (https://github.com/deepakrana47/RNN-attention-network/blob/master/sentiment_analyzer_keras.py)
    dim = int(lstm_out.shape[2])  # getting shape
    attention_layer = Dense(1, activation="tanh")(lstm_out)
    attention_layer = Flatten()(attention_layer)
    attention_layer = Activation("softmax")(attention_layer)
    attention_layer = RepeatVector(dim)(attention_layer)  # recurring layer
    attention_layer = Permute([2, 1])(attention_layer)
    attention_out = tf.keras.layers.concatenate([lstm_out, attention_layer])
    attention_out = Lambda(
        lambda xin: tf.keras.backend.sum(xin, axis=-2), output_shape=(dim,),
    )(attention_out)
    output = Dense(1, activation="sigmoid")(attention_out)

    model = tf.keras.Model(
        inputs=input_layer, outputs=output
    )  # WHAT SHOULD THE INPUT BE?

    return model


def define_sequential_rnn(MAX_TOKENS, MAX_SEQ_LEN, REG_CONSTANT, DROPOUT_RATE, ds):

    # Create TextVectorization layer
    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=MAX_TOKENS, output_mode="int", output_sequence_length=MAX_SEQ_LEN
    )

    # Use `adapt` to create a vocabulary mapping words to integers
    train_text = ds["train"].map(lambda x: x["text"])
    vectorize_layer.adapt(train_text)

    # Getting our dimensions
    VOCAB_SIZE = len(vectorize_layer.get_vocabulary())
    EMBEDDING_SIZE = int(np.sqrt(VOCAB_SIZE))
    print(
        "Vocab size is {} and is embedded into {} dimensions".format(
            VOCAB_SIZE, EMBEDDING_SIZE
        )
    )

    # Getting our embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        VOCAB_SIZE, EMBEDDING_SIZE, name="embedding"
    )

    cells = [tf.keras.layers.LSTMCell(256), tf.keras.layers.LSTMCell(64)]
    rnn = tf.keras.layers.RNN(cells)
    lstm = tf.keras.layers.LSTM(
        100,
        return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.l2(REG_CONSTANT),
        recurrent_regularizer=tf.keras.regularizers.l2(REG_CONSTANT),
        bias_regularizer=tf.keras.regularizers.l2(REG_CONSTANT),
    )
    drop = tf.keras.layers.Dropout(DROPOUT_RATE)
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    model = tf.keras.Sequential(
        [vectorize_layer, embedding_layer, rnn, drop, output_layer]
    )
    return model
