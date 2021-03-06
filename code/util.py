import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops


def getDataset(num_words, max_seq_len):

    # load the text dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=num_words
    )

    # padding the sequences
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_seq_len)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_seq_len)

    return (x_train, y_train), (x_test, y_test)


def print_arch(model):
    print(model.summary())
    with open("./output/report.txt", "w") as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + "\n"))


def graph_one(history, PATH):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig("./output/" + PATH + "_accuracy.png")
    plt.clf()

    plt.plot(history.history["val_loss"])
    plt.plot(history.history["loss"])
    plt.title("model loss version ")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(
        ["validation loss, loss"], loc="lower right",
    )
    plt.savefig("./output/" + PATH + "_loss.png")
    plt.clf()

