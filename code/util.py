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


def getDS():
    # load the text dataset
    ds = tfds.load("imdb_reviews")

    return ds


def forward_pass(ds, model):
    # test a forward pass
    for batch in ds["train"].batch(32):
        logits = model(batch["text"])
        loss = tf.keras.losses.binary_crossentropy(
            tf.expand_dims(batch["label"], -1), logits, from_logits=True
        )
        print(loss)
        break


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


def graph_two(history1, history2, dr1, dr2, cr1, cr2):
    plt.plot(history1.history["val_accuracy"])
    plt.plot(history2.history["val_accuracy"])
    plt.title("model accuracy comparison")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    lgd = plt.legend(
        [
            "Dropping Rate=" + str(dr1) + "Reg Constant=" + str(cr1),
            "Dropping Rate=" + str(dr2) + "Reg Constant=" + str(cr2),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
    )
    plt.savefig(
        "./output/COMPARISON_accuracy.png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    plt.clf()

    plt.plot(history1.history["val_loss"])
    plt.plot(history2.history["val_loss"])
    plt.title("model loss comparison")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    lgd = plt.legend(
        [
            "Dropping Rate=" + str(dr1) + " Reg Constant=" + str(cr1),
            "Dropping Rate=" + str(dr2) + " Reg Constant=" + str(cr2),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
    )
    plt.savefig(
        "./output/COMPARISON_loss.png", bbox_extra_artists=(lgd,), bbox_inches="tight"
    )
    plt.clf()


def get_confusion_matrix(PATH, test_x, test_y, DROPOUT_RATE, REG_CONSTANT):
    PATH = "./models/" + PATH
    model = tf.keras.models.load_model(PATH)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=["accuracy"],
    )
    evaluation = model.evaluate(test_x, test_y, verbose=0)
    print("Retrieved model from " + PATH)
    print("\nTest accuracy: ", evaluation[1])

    f = open("./output/evaluated_results.txt", "a")
    f.write(
        "Dropout rate "
        + str(DROPOUT_RATE)
        + " Reg constant "
        + str(REG_CONSTANT)
        + ": "
        + str(evaluation[1])
        + "\n"
    )
    f.close()

    y_pred = model.predict(test_x)
    y_pred = np.argmax(y_pred, axis=1)

    conf_mat = tf.math.confusion_matrix(test_y, y_pred, num_classes=2)
    plt.title("Confusion Matrix of Model")
    plt.imshow(conf_mat, cmap=plt.cm.jet, interpolation="nearest")
    plt.colorbar()
    plt.savefig("./output/" + PATH + ".png")
    plt.clf()
