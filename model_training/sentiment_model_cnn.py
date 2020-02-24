"""
Model definition for CNN sentiment training


"""

import os
import tensorflow as tf


def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling

    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Embedding, Convolution1D, GlobalMaxPool1D
    import tensorflow.keras as keras
    import numpy as np

    f = open(config["embeddings_path"],encoding='utf8')
    glove = f.readlines()[:config["embeddings_dictionary_size"]]
    f.close()

    embedding_matrix = np.zeros((config["embeddings_dictionary_size"], config["embeddings_vector_size"]))
    for i in range(config["embeddings_dictionary_size"]):
        if len(glove[i].split()[1:]) != config["embeddings_vector_size"]:
            continue
        embedding_matrix[i] = np.asarray(glove[i].split()[1:], dtype='float32')

    cnn_model = Sequential()
    cnn_model.add(Embedding(weights=[embedding_matrix], input_length = config["padding_size"],input_dim = config["embeddings_dictionary_size"],output_dim = config["embeddings_vector_size"], trainable = True))
    cnn_model.add(Convolution1D(filters=100,kernel_size=2,strides = 1, padding='valid',activation = 'relu'))
    cnn_model.add(GlobalMaxPool1D())
    cnn_model.add(Dense(units=100, activation = 'relu'))
    cnn_model.add(Dense(units=1, activation = 'sigmoid'))
    Adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    cnn_model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics =['accuracy'])

    return cnn_model

def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving

    """

    #model.save(os.path.join(output, "1"))
    #IDK why he wrote this
    model.save(output)

    print("Model successfully saved at: {}".format(output))
