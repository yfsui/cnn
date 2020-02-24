"""
Model definition for CNN sentiment training


"""

import os
import tensorflow as tf


def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling

    """
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Embedding, Convolution1D, GlobalMaxPooling1D

    cnn_model = Sequential()
    cnn_model.add(Embedding(input_length = config["padding_size"],
        input_dim = config["embeddings_dictionary_size"],
        output_dim = config["embeddings_vector_size"], 
        trainable = True))
    cnn_model.add(Convolution1D(filters=100,kernel_size=2,strides = 1, padding='valid',activation = 'relu'))
    cnn_model.add(GlobalMaxPooling1D())
    cnn_model.add(Dense(units=100, activation = 'relu'))
    cnn_model.add(Dense(units=1, activation = 'sigmoid'))

    cnn_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics =['accuracy'])

    return cnn_model

def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving

    """

    #model.save(os.path.join(output, "1"))
    #IDK why he wrote this
    model.save(output)

    print("Model successfully saved at: {}".format(output))
