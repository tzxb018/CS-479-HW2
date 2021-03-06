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

# setting up k-fold cross validation
K = 30
model1_accuracies = []
model2_accuracies = []
model1_loss = []
model2_loss = []
error_diff = []
error_diff_estimation = 0
# selecting at random which K-iteration to graph for
K_graph = 15
for i in range(1, K):

    print("K-iteration", i)
    len_of_train = len(x_train)
    len_of_test = len(x_test)
    # partitioning data into K equal sized subsets
    # since dataset is already divided into training and testing, we will partition both the training and testing data seperately
    x_train_k = x_train[(len_of_train // K * (i - 1)) : (len_of_train // K * i)]
    y_train_k = y_train[(len_of_train // K * (i - 1)) : (len_of_train // K * i)]
    x_test_k = x_test[(len_of_test // K * (i - 1)) : (len_of_test // K * i)]
    y_test_k = y_test[(len_of_test // K * (i - 1)) : (len_of_test // K * i)]

    BATCH_SIZE = 64
    EPOCHS = 50
    EMBEDDING_SIZE = 32
    TYPE_OF_RNN = "LSTM"

    # train learning algorithm L1 on training set i to get hypo 1
    # getting our model
    DROPOUT_RATE = 0.5
    REG_CONSTANT = 0.01
    model1 = model.define_rnn(
        EMBEDDING_SIZE, MAX_TOKENS, MAX_SEQ_LEN, DROPOUT_RATE, REG_CONSTANT
    )

    # compiling our model!
    model1.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=["accuracy"],
    )

    # setting up early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)

    # training our model
    history = model1.fit(
        x_train_k,
        y_train_k,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
    )

    # util.print_arch(model)

    # saving our model
    PATH = (
        "rnn_"
        + str(TYPE_OF_RNN)
        + "_dr"
        + str(DROPOUT_RATE).replace(".", "x")
        + "_rc"
        + str(REG_CONSTANT).replace(".", "x")
    )
    # model.save(PATH)
    tf.keras.Model.save(model1, "./models/" + PATH)
    if i == K_graph:
        util.graph_one(history, PATH)

    # evaluating our model 1
    evaluate1 = model1.evaluate(x_test_k, y_test_k, verbose=0)
    print("test accuracy: ", evaluate1[1])
    print("loss: ", evaluate1[0])
    model1_accuracies.append(evaluate1[1])
    model1_loss.append(evaluate1[0])

    # *********************************************************************************
    # train learning algorithm L2 on training set i to get hypo 2
    # getting our model
    # train learning algorithm L1 on training set i to get hypo 1
    # getting our model
    DROPOUT_RATE = 0.5
    REG_CONSTANT = 0.01
    model2 = model.define_rnn(
        EMBEDDING_SIZE, MAX_TOKENS, MAX_SEQ_LEN, DROPOUT_RATE, REG_CONSTANT
    )

    # compiling our model!
    model2.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=["accuracy"],
    )

    # setting up early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)

    # training our model
    history = model2.fit(
        x_train_k,
        y_train_k,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
    )

    # util.print_arch(model)

    # saving our model
    PATH = (
        "rnn_"
        + str(TYPE_OF_RNN)
        + "_dr"
        + str(DROPOUT_RATE).replace(".", "x")
        + "_rc"
        + str(REG_CONSTANT).replace(".", "x")
    )
    # model.save(PATH)
    tf.keras.Model.save(model2, "./models/" + PATH)
    if i == K_graph:
        util.graph_one(history, PATH)

    # evaluating our model 2
    evaluate2 = model2.evaluate(x_test_k, y_test_k, verbose=0)
    print("test accuracy: ", evaluate2[1])
    print("loss: ", evaluate2[0])
    model1_accuracies.append(evaluate2[1])
    model1_loss.append(evaluate2[0])

    # finding the error difference in this k-iteration
    p_i = evaluate1[0] - evaluate2[0]
    error_diff.append(p_i)
    error_diff_estimation = error_diff_estimation + p_i

error_diff_estimation = error_diff_estimation / K
print("Error Difference Estaimation: " + str(error_diff_estimation))
