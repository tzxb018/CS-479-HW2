import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt  # to visualize data and draw plots
from tqdm import tqdm  # to track progress of loops
import model
import util

# getting the data set
ds = util.getDataset()

# getting the first two layers of our network from a helper function in util
vectorize_layer, embedding_layer = util.getPrelimLayers()

# getting our model
model = model.define_model(vectorize_layer, embedding_layer)

# testing a forward pass to get 
util.test_forward_pass(model,ds)

# compiling our model!

