# -*- coding: utf-8 -*-
"""
@author: Manuel Camargo
@co-author: Rehan Fazal
"""
import os
import streamlit as st
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from keras.layers import Input, Embedding, Concatenate, Dot
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Nadam, Adam, SGD, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from support_modules.callbacks import time_callback as tc
from support_modules.callbacks import clean_models_callback as cm

def _training_model(vec, ac_weights, rl_weights, output_folder, args):
    """Example function with types documented in the docstring.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """
# =============================================================================
#     Input layer
# =============================================================================

    ac_input = Input(shape=(vec['prefixes']['activities'].shape[1], ), name='ac_input')
    rl_input = Input(shape=(vec['prefixes']['roles'].shape[1], ), name='rl_input')
    t_input = Input(shape=(vec['prefixes']['times'].shape[1],
                           vec['prefixes']['times'].shape[2]), name='t_input')

#=============================================================================
#    Embedding layer for categorical attributes
# =============================================================================

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

# =============================================================================
#    Layer 1
# =============================================================================

    concatenate = Concatenate(name='concatenated', axis=2)([ac_embedding, rl_embedding, t_input])


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

    if ('dense_act' in args) and (args['dense_act'] is not None):
        time_output = Dense(vec['next_evt']['times'].shape[1],
                            activation=args['dense_act'],
                            kernel_initializer='glorot_uniform',
                            name='time_output')(l2_3)
    else:
        time_output = Dense(vec['next_evt']['times'].shape[1],
                            kernel_initializer='glorot_uniform',
                            name='time_output')(l2_3)
    model = Model(inputs=[ac_input, rl_input, t_input],
                  outputs=[act_output, role_output, time_output])

    if args['optim'] == 'Nadam':
        opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    elif args['optim'] == 'Adam':
        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False) #Hyperparameter Opt with amsgrad=True/False
                                                                #   AMSGrad is an extension to the Adam version of
                                                                #   gradient descent that attempts to improve the
                                                                #   convergence properties of the algorithm, avoiding
                                                                #   large abrupt changes in the learning rate for each input variable
    elif args['optim'] == 'SGD':
        opt = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
    elif args['optim'] == 'Adagrad':
        opt = Adagrad(learning_rate=0.01)

    model.compile(loss={'act_output': 'categorical_crossentropy',
                        'role_output': 'categorical_crossentropy',
                        'time_output': 'mae'}, optimizer=opt, metrics=['accuracy'])

    model.summary()

    model_history_file_path = os.path.join(output_folder, "parameters", "model_history_log.csv")
    csv_logger = CSVLogger(model_history_file_path, append=True)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
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

    history = model.fit({'ac_input': vec['prefixes']['activities'],
                   'rl_input': vec['prefixes']['roles'],
                   't_input': vec['prefixes']['times']},
                {'act_output': vec['next_evt']['activities'],
                 'role_output': vec['next_evt']['roles'],
                 'time_output': vec['next_evt']['times']},
                  validation_split=0.2,
                  verbose=2,
                  callbacks=[early_stopping, model_checkpoint, lr_reducer, cb, clean_models, csv_logger],
                  batch_size=batch_size,
                  epochs=args['epochs'])

    with st.container():

        st.subheader("Fully Shared Model Performance")

        fcol1, fcol2, fcol3 = st.columns([2, 2, 2])

        with fcol1:
            fig1 = plt.figure()
            plt.plot(history.history['act_output_loss'], label='train')
            plt.plot(history.history['val_act_output_loss'], label='validation')
            plt.plot(history.history['act_output_accuracy'], label='acc')
            plt.title('Activity Loss')
            plt.ylabel('loss/accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation', 'acc'], loc='upper left')
            # plt.show()
            st.write(fig1);

        with fcol2:
            fig2 = plt.figure()
            plt.plot(history.history['role_output_loss'], label='train')
            plt.plot(history.history['val_role_output_loss'], label='validation')
            plt.plot(history.history['act_output_accuracy'], label='acc')
            plt.title('Role Loss')
            plt.ylabel('loss/accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation', 'acc'], loc='upper left')
            st.write(fig2);

        with fcol3:
            fig3 = plt.figure()
            plt.plot(history.history['time_output_loss'], label='train')
            plt.plot(history.history['val_time_output_loss'], label='validation')
            plt.plot(history.history['act_output_accuracy'], label='acc')
            plt.title('Time Loss')
            plt.ylabel('loss/accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation', 'acc'], loc='upper left')
            st.write(fig3);

    fig4 = plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    st.write(fig4);