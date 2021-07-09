# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:15:12 2019

@author: Manuel Camargo
"""
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Concatenate
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Nadam, Adam, SGD, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.layers import Reshape

from support_modules.callbacks import time_callback as tc
from support_modules.callbacks import clean_models_callback as cm


def _training_model(vec, ac_weights, rl_weights, label_weights, output_folder, args):
    """Example function with types documented in the docstring.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """

    print('Build model - concatenated')
    print(args)
# =============================================================================
#     Input layer
# =============================================================================
    print("***ac_input Inputs vec*** :", vec['prefixes']['activities'])
    print("***rl_input Inputs vec*** :",vec['prefixes']['roles'])
    print("***label_input Inputs vec*** :", vec['prefixes']['label'])

    ac_input = Input(shape=(vec['prefixes']['activities'].shape[1], ), name='ac_input')
    rl_input = Input(shape=(vec['prefixes']['roles'].shape[1], ), name='rl_input')
    label_input = Input(shape=(vec['prefixes']['label'].shape[1], ), name='label_input')
    t_input = Input(shape=(vec['prefixes']['times'].shape[1],
                           vec['prefixes']['times'].shape[2]), name='t_input')
    print("***ac_input Inputs*** :", ac_input)
    print("***rl_input Inputs*** :", rl_input)
    print("***label_input Inputs*** :", label_input)
    print("***t_input Inputs*** :", t_input)

#=============================================================================
#    Embedding layer for categorical attributes
# =============================================================================
    print("AC Weight Value", ac_weights)
    print("AC Weight", ac_weights.shape[0],"&", ac_weights.shape[1])
    print("RL Weight Value", rl_weights)
    print("RL Weight", rl_weights.shape[0],"&", rl_weights.shape[1])
    print("LB Weight Value", label_weights)
    print("LB Weight", label_weights.shape[0],"&", label_weights.shape[1])
    print("LB INPUT Length :", vec['prefixes']['label'].shape[1])
    print("LB INPUT Length Values :", vec['prefixes']['label'])
    print("LB Input :", label_input)
    print("LB OUT Length Values :", vec['next_evt']['label'], type(vec['next_evt']['label']))
    print("LB OUT Length sShape :", vec['next_evt']['label'].shape[1])
    ac_embedding = Embedding(ac_weights.shape[0],
                             ac_weights.shape[1],
                             weights=[ac_weights],
                             input_length=vec['prefixes']['activities'].shape[1],
                             trainable=False, name='ac_embedding')(ac_input)

    rl_embedding = Embedding(rl_weights.shape[0],
                             rl_weights.shape[1],
                             weights=[rl_weights],
                             input_length=vec['prefixes']['roles'].shape[1],
                             trainable=False, name='rl_embedding')(rl_input)


    label_embedding = Embedding(label_weights.shape[0],
                             label_weights.shape[1],
                             weights=[label_weights],
                             input_length=vec['prefixes']['label'].shape[1],
                             trainable=True, name='label_embedding')(label_input)

    # label_embedding = Embedding(name='label_embedding',
    #                            input_dim=len(label_index),
    #                            output_dim=embedding_size)(label)

# =============================================================================
#    Layer 1
# =============================================================================
    concatenate = Concatenate(name='concatenated', axis=2)([ac_embedding, rl_embedding, label_embedding, t_input])

    if args['lstm_act'] is not None:
        l1_c1 = LSTM(args['l_size'],
                     activation=args['lstm_act'],
                     kernel_initializer='glorot_uniform',
                     return_sequences=True,
                     dropout=0.2,
                     implementation=args['imp'])(concatenate)
    else:
        l1_c1 = LSTM(args['l_size'],
                     kernel_initializer='glorot_uniform',
                     return_sequences=True,
                     dropout=0.2,
                     implementation=args['imp'])(concatenate)

# =============================================================================
#    Batch Normalization Layer
# =============================================================================
    batch1 = BatchNormalization()(l1_c1)

# =============================================================================
# The layer specialized in prediction
# =============================================================================
    l2_c1 = LSTM(args['l_size'],
                 kernel_initializer='glorot_uniform',
                 return_sequences=False,
                 dropout=0.2,
                 implementation=args['imp'])(batch1)

#   The layer specialized in role prediction
    l2_c2 = LSTM(args['l_size'],
                 kernel_initializer='glorot_uniform',
                 return_sequences=False,
                 dropout=0.2,
                 implementation=args['imp'])(batch1)

#   The layer specialized in label prediction
    l2_c3 = LSTM(args['l_size'],
                kernel_initializer='glorot_uniform',
                return_sequences=False,
                dropout=0.2,
                implementation=args['imp'])(batch1)

    if args['lstm_act'] is not None:
        l2_3 = LSTM(args['l_size'],
                    activation=args['lstm_act'],
                    kernel_initializer='glorot_uniform',
                    return_sequences=False,
                    dropout=0.2,
                    implementation=args['imp'])(batch1)
    else:
        l2_3 = LSTM(args['l_size'],
                    kernel_initializer='glorot_uniform',
                    return_sequences=False,
                    dropout=0.2,
                    implementation=args['imp'])(batch1)

# =============================================================================
# Output Layer
# =============================================================================
    act_output = Dense(ac_weights.shape[0],
                       activation='softmax',
                       kernel_initializer='glorot_uniform',
                       name='act_output')(l2_c1)

    role_output = Dense(rl_weights.shape[0],
                        activation='softmax',
                        kernel_initializer='glorot_uniform',
                        name='role_output')(l2_c2)

    label_output = Dense(label_weights.shape[0],
                        activation='softmax',
                        kernel_initializer='glorot_uniform',
                        name='label_output')(l2_c3)

    if ('dense_act' in args) and (args['dense_act'] is not None):
        time_output = Dense(vec['next_evt']['times'].shape[1],
                            activation=args['dense_act'],
                            kernel_initializer='glorot_uniform',
                            name='time_output')(l2_3)
    else:
        time_output = Dense(vec['next_evt']['times'].shape[1],
                            kernel_initializer='glorot_uniform',
                            name='time_output')(l2_3)
    model = Model(inputs=[ac_input, rl_input, label_input, t_input],
                  outputs=[act_output, role_output, label_output, time_output])

    if args['optim'] == 'Nadam':
        opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    elif args['optim'] == 'Adam':
        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif args['optim'] == 'SGD':
        opt = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
    elif args['optim'] == 'Adagrad':
        opt = Adagrad(learning_rate=0.01)

    model.compile(loss={'act_output': 'categorical_crossentropy',
                        'role_output': 'categorical_crossentropy',
                        'label_output': 'categorical_crossentropy',
                        'time_output': 'mae'}, optimizer=opt)

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    cb = tc.TimingCallback(output_folder)
    clean_models = cm.CleanSavedModelsCallback(output_folder, 2)

    # Output file
    output_file_path = os.path.join(output_folder,
                                    'model_' + str(args['model_type']) +
                                    '_{epoch:02d}-{val_loss:.2f}.h5')

    # Saving
    model_checkpoint = ModelCheckpoint(output_file_path,
                                       monitor='val_loss',
                                       verbose=0,
                                       save_best_only=True, #saves when the model is considered the "best" and the latest best model according to the quantity monitored will not be overwritten.
                                       save_weights_only=False,
                                       mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.5,
                                   patience=10,
                                   verbose=0,
                                   mode='auto',
                                   min_delta=0.0001,
                                   cooldown=0,
                                   min_lr=0)
    #To automatically calculate the batch size if the batch size is set to default i.e 0
    if args['batch_size'] == 0:
        batch_size = vec['prefixes']['activities'].shape[1]
    else:
        batch_size = args['batch_size']
    print("Batch Size : ", batch_size)
    #label_o = Reshape(target_shape=[2])(vec['next_evt']['label'])
    #print("Input Activities :", vec['prefixes']['activities'])
    #print("Input Roles :", vec['prefixes']['roles'])
    #print("Input Prefixes Times :", vec['prefixes']['times'])
    #print("Input Next Event Times :", vec['next_evt']['times'])
    model.fit({'ac_input': vec['prefixes']['activities'],
               'rl_input': vec['prefixes']['roles'],
               'label_input': vec['prefixes']['label'],
               't_input': vec['prefixes']['times']},
              {'act_output': vec['next_evt']['activities'],
               'role_output': vec['next_evt']['roles'],
               'label_output': vec['next_evt']['label'],
               'time_output': vec['next_evt']['times']},
              validation_split=0.2,
              verbose=2,
              callbacks=[early_stopping, model_checkpoint,
                         lr_reducer, cb, clean_models],
              batch_size=batch_size,
              epochs=args['epochs'])
