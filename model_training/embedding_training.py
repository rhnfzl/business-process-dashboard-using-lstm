# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:23:55 2018
@author: Manuel Camargo
"""
import os
import random
import itertools
import math
import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, Dot, Reshape, Multiply, Concatenate

from support_modules import support as sup


def training_model(parameters, log, ac_index, index_ac, rl_index, index_rl, label_index, index_label):
    """Main method of the embedding training module.
    Args:
        parameters (dict): parameters for training the embeddeding network.
        timeformat (str): event-log date-time format.
        no_loops (boolean): remove loops fom the event-log (optional).
    """
    # Define the number of dimensions as the 4th root of the # of categories
    dim_number = math.ceil(
        len(list(itertools.product(*[list(ac_index.items()),
                                     list(rl_index.items()),
                                     list(label_index.items())])))**0.25)

    print("Dimension Number :", dim_number)

    ac_weights, rl_weights, label_weights = train_embedded(log,
                                            ac_index, rl_index, label_index, dim_number)

    if not os.path.exists(os.path.join('input_files', 'embedded_matix')):
        os.makedirs(os.path.join('input_files', 'embedded_matix'))

    sup.create_file_from_list(
        reformat_matrix(index_ac, ac_weights),
        os.path.join(os.path.join('input_files', 'embedded_matix'),
                     'ac_' + parameters['file_name'].split('.')[0]+'.emb'))
    sup.create_file_from_list(
        reformat_matrix(index_rl, rl_weights),
        os.path.join(os.path.join('input_files', 'embedded_matix'),
                     'rl_' + parameters['file_name'].split('.')[0]+'.emb'))

    sup.create_file_from_list(
        reformat_matrix(index_label, label_weights),
        os.path.join(os.path.join('input_files', 'embedded_matix'),
                     'label_' + parameters['file_name'].split('.')[0]+'.emb'))


# =============================================================================
# Pre-processing: embedded dimension
# =============================================================================

def train_embedded(log_df, ac_index, rl_index, label_index, dim_number):
    """Carry out the training of the embeddings"""
    # Iterate through each book
    pairs = list()
    for i in range(0, len(log_df)):
        # Iterate through the links in the book
        pairs.append((ac_index[log_df.iloc[i]['task']],
                      rl_index[log_df.iloc[i]['role']],
                      label_index[log_df.iloc[i]['label']]))

    model = ac_rl_embedding_model(ac_index, rl_index, label_index, dim_number)
    model.summary()

    n_positive = 1024
    gen = generate_batch(pairs, ac_index, rl_index, label_index,
                         n_positive, negative_ratio=2)

    print("Bacth Generator :", gen)
    # Train
    model.fit(gen, epochs=100,
                        steps_per_epoch=len(pairs) // n_positive,
                        verbose=2)

    # Extract embeddings
    ac_layer = model.get_layer('activity_embedding')
    rl_layer = model.get_layer('role_embedding')
    label_layer = model.get_layer('label_embedding')

    ac_weights = ac_layer.get_weights()[0]
    rl_weights = rl_layer.get_weights()[0]
    label_weights = label_layer.get_weights()[0]

    return ac_weights, rl_weights, label_weights


def generate_batch(pairs, ac_index, rl_index, label_index, n_positive=50,
                   negative_ratio=1.0):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 4)) #changed from size 3 to 4
    print("Batch Size :", batch_size, "& :", batch)
    pairs_set = set(pairs)
    activities = list(ac_index.keys())
    roles = list(rl_index.keys())
    labels = list(label_index.keys())

    # This creates a generator
    while True:
        # randomly choose positive examples
        idx = 0
        for idx, (activity, role, label) in enumerate(random.sample(pairs,
                                                             n_positive)):

            batch[idx, :] = (activity, role, label, 1)
        # Increment idx by 1
        idx += 1

        # Add negative examples until reach batch size
        while idx < batch_size:
            # random selection
            random_ac = random.randrange(len(activities))
            random_rl = random.randrange(len(roles))
            random_label = random.randrange(len(labels))



            # Check to make sure this is not a positive example
            if (random_ac, random_rl, random_label) not in pairs_set:

                # Add to batch and increment index,  0 due classification task
                batch[idx, :] = (random_ac, random_rl, random_label, 0)
                idx += 1

        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'activity': batch[:, 0], 'role': batch[:, 1], 'label': batch[:, 2]},  batch[:, 3]


def ac_rl_embedding_model(ac_index, rl_index, label_index, embedding_size):
    """Model to embed activities and roles using the functional API"""

    # Both inputs are 1-dimensional
    activity = Input(name='activity', shape=[1])
    role = Input(name='role', shape=[1])
    label = Input(name='label', shape=[1])

    # Embedding the activity (shape will be (None, 1, embedding_size))
    activity_embedding = Embedding(name='activity_embedding',
                                   input_dim=len(ac_index),
                                   output_dim=embedding_size)(activity)

    # Embedding the role (shape will be (None, 1, embedding_size))
    role_embedding = Embedding(name='role_embedding',
                               input_dim=len(rl_index),
                               output_dim=embedding_size)(role)


    # Embedding the label (shape will be (None, 1, embedding_size))

    label_embedding = Embedding(name='label_embedding',
                               input_dim=len(label_index),
                               output_dim=embedding_size)(label)

    # print("Activity Index : ", len(ac_index), "& : ", ac_index, "Input :", activity, "Embedding :", activity_embedding,
    #       "Embedding Shape :", activity_embedding.shape)
    # print("Role Index : ", len(rl_index), "& : ", rl_index, "Input :", role, "Embedding :", role_embedding,
    #       "Embedding Shape :", role_embedding.shape)
    # print("Label Index : ", len(label_index), "& : ", label_index, "Input :", label, "Embedding :", label_embedding,
    #       "Embedding Shape :", label_embedding.shape)

    merged = Dot(name='dot_product',
                 normalize=True, axes=2)([activity_embedding, role_embedding])


    #the dot product of two vectors, a and b represents the projection of a to b and the resulting scalar is the length of that projection
    merged = Dot(name='dot_product_label',
                 normalize=True, axes=1)([merged, label_embedding])

    # Loss function is mean squared error
    model = Model(inputs=[activity, role, label], outputs=merged)
    model.compile(optimizer='Adam', loss='mse')

    return model

# =============================================================================
# Support
# =============================================================================


def reformat_matrix(index, weigths):
    """Reformating of the embedded matrix for exporting.
    Args:
        index: index of activities or roles.
        weigths: matrix of calculated coordinates.
    Returns:
        matrix with indexes.
    """
    matrix = list()
    for i, _ in enumerate(index):
        data = [i, index[i]]
        data.extend(weigths[i])
        matrix.append(data)
    return matrix