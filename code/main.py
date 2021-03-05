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
# (x_train, y_train), (x_test, y_test) = util.getDataset(MAX_TOKENS, MAX_SEQ_LEN)
ds = util.getDS()

# getting our model
DROPOUT_RATE = 0.5
REG_CONSTANT = 0.01
BATCH_SIZE = 64

model = model.define_rnn(
    MAX_TOKENS, MAX_SEQ_LEN, DROPOUT_RATE, REG_CONSTANT, BATCH_SIZE, ds
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
EPOCHS = 3
history = model.fit(
    ds["train"],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
)

util.print_arch(model)

# saving our model
VERSION = 1
PATH = "./models/rnn_v" + str(VERSION)
# model.save(PATH)
tf.keras.Model.save(model, PATH)

# evaluating our model
evaluate = model.evaluate(ds["test"], verbose=0)
print("test accuracy: ", evaluate[1])

util.graph_one(history, VERSION)

######################################################################################


# obtaining our model
# DROPOUT_RATE = 0.5
# REG_CONSTANT = 0.01
# model = model.define_sequential_rnn(
#     MAX_TOKENS, MAX_SEQ_LEN, REG_CONSTANT, DROPOUT_RATE, ds
# )

# # testing a forward pass
# util.forward_pass(ds, model)

# # setting up parameters for training
# loss_values = []
# accuracy_values = []
# BATCH_SIZE = 32
# EPOCHS = 1
# LEARNING_RATE = 0.001
# optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# # Loop through epochs of data
# for epoch in range(EPOCHS):
#     for batch in tqdm(ds["train"].batch(BATCH_SIZE)):
#         with tf.GradientTape() as tape:
#             # run network
#             text = batch["text"]
#             labels = batch["label"]
#             logits = model(text)

#             # calculate loss
#             loss = tf.keras.losses.binary_crossentropy(
#                 tf.expand_dims(batch["label"], -1), logits, from_logits=True
#             )
#         loss_values.append(loss)

#         # gradient update
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))

#         # calculate accuracy
#         predictions = tf.argmax(logits, axis=1)
#         accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
#         accuracy_values.append(accuracy)

# print(model.summary())

# # accuracy
# print("Accuracy:", np.mean(accuracy_values))

# loss_values_eval = []
# accuracy_values_eval = []
# # evaluating the model
# for batch in tqdm(ds["test"].batch(BATCH_SIZE)):
#     text = batch["text"]
#     labels = batch["label"]
#     logits = model(text)

#     # calculate loss
#     loss = tf.keras.losses.binary_crossentropy(
#         tf.expand_dims(batch["label"], -1), logits, from_logits=True
#     )
#     loss_values_eval.append(loss)
#     # calculate accuracy
#     predictions = tf.argmax(logits, axis=1)
#     accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
#     accuracy_values_eval.append(accuracy)

# # accuracy
# print("Evaluated acuracy:", np.mean(accuracy_values_eval))
