import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
import model
import util

# getting the data set
MAX_SEQ_LEN = 128
MAX_TOKENS = 5000
(x_train, y_train), (x_test, y_test) = util.getDataset(MAX_TOKENS, MAX_SEQ_LEN)


# getting our model
EMBEDDING_SIZE = 32
DROPOUT_RATE = 0.5
REG_CONSTANT = 0.01
model = model.define_rnn(
    EMBEDDING_SIZE, MAX_TOKENS, MAX_SEQ_LEN, DROPOUT_RATE, REG_CONSTANT
)

# compiling our model!
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=["accuracy"],
)

# setting up early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)

# training our model
BATCH_SIZE = 64
EPOCHS = 3
history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_split=0.1,
)

util.print_arch(model)

# saving our model
VERSION = 1
PATH = "./models/rnn_v" + str(VERSION)
# model.save(PATH)
tf.keras.Model.save(model, PATH)

# evaluating our model
evaluate = model.evaluate(x_test, y_test, verbose=0)
print("test accuracy: ", evaluate[1])

util.graph_one(history, VERSION)
