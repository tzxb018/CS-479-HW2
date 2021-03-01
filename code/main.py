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
model = model.define_model(EMBEDDING_SIZE, MAX_TOKENS, MAX_SEQ_LEN)

# compiling our model!
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=["accuracy"],
)

# training our model
BATCH_SIZE = 32
EPOCHS = 3
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

util.print_arch(model)

# evaluating our model
history = model.evaluate(x_test, y_test, verbose=0)
print("test accuracy: ", history[1])
