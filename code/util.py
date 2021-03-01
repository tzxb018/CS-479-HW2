import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops


def getDataset():
    # load the text dataset
    ds = tfds.load("imdb_reviews")
    return ds


def getPrelimLayers():
    MAX_SEQ_LEN = 128
    MAX_TOKENS = 5000

    ds = getDataset()

    # Create TextVectorization layer
    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=MAX_TOKENS, output_mode="int", output_sequence_length=MAX_SEQ_LEN
    )

    # Use `adapt` to create a vocabulary mapping words to integers
    train_text = ds["train"].map(lambda x: x["text"])
    vectorize_layer.adapt(train_text)

    # Let's print out a batch to see what it looks like in text and in integers
    for batch in ds["train"].batch(1):
        text = batch["text"]
        print(list(zip(text.numpy(), vectorize_layer(text).numpy())))
        break

    VOCAB_SIZE = len(vectorize_layer.get_vocabulary())
    EMBEDDING_SIZE = int(np.sqrt(VOCAB_SIZE))
    print(
        "Vocab size is {} and is embedded into {} dimensions".format(
            VOCAB_SIZE, EMBEDDING_SIZE
        )
    )

    embedding_layer = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)

    return vectorize_layer, embedding_layer


def print_arch(model):
    print(model.summary())
    with open("./output/report.txt", "w") as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + "\n"))


def test_forward_pass(model, ds):
    # test a forward pass
    for batch in ds["train"].batch(32):
        logits = model(batch["text"])
        loss = tf.keras.losses.binary_crossentropy(
            tf.expand_dims(batch["label"], -1), logits, from_logits=True
        )
        print(loss)
        print_arch(model)
        break
