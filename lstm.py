# -*- coding: utf-8 -*-
"""
@author: Manuel Camargo
"""

#---Workaround for Not creating XLA devices, tf_xla_enable_xla_devices not set
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#---

import sys
import getopt

from model_prediction import model_predictor as pr
from model_training import model_trainer as tr

#---Workaround for "tensorflow.python.framework.errors_impl.UnknownError: Fail to find the dnn implementation."
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
session = InteractiveSession(config=config)
#-----

def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-o': 'one_timestamp', '-a': 'activity',
              '-f': 'file_name', '-i': 'imp', '-l': 'lstm_act',
              '-d': 'dense_act', '-p': 'optim', '-n': 'norm_method',
              '-m': 'model_type', '-z': 'n_size', '-y': 'l_size',
              '-c': 'folder', '-b': 'model_file', '-x': 'is_single_exec',
              '-t': 'max_trace_size', '-e': 'splits', '-g': 'sub_group',
              '-v': 'variant', '-r': 'rep'}
    try:
        return switch[opt]
    except:
        raise Exception('Invalid option ' + opt)


# --setup--
def main(argv):
    """Main aplication method"""
    parameters = dict()
    column_names = {'Case ID': 'caseid',
                    'Activity': 'task',
                    'lifecycle:transition': 'event_type', #---#
                    'Resource': 'user'}
    parameters['one_timestamp'] = True # Only one timestamp for each activity i.e the start and end time will be same
    # Similarity btw the resources profile execution (Song e.t. all)
    parameters['rp_sim'] = 0.85
    parameters['batch_size'] = 128 # Usually 16/32/64/128/256
    parameters['epochs'] = 200 #v1 200, for embedded training it's 100.
    # Parameters setting manual fixed or catched by console
    '''
        **Concept**
        One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE. Since one epoch is too big to feed to the computer at once we divide it in several smaller batches.
        Batch size is total number of training examples present in a single batch and Iterations is the number of batches needed to complete one epoch.
        e.g : For 2000 training examples we can divide the dataset of 2000 examples into batches of 500 then it will take 4 iterations to complete 1 epoch. Where Batch Size is 500 and Iterations is 4, for 1 complete epoch.
        Conclusion : Lower the Batch size higher will be the Epoch Value
    '''
    if not argv:
        # Type of LSTM task -> training, pred_log
        # pred_sfx, predict_next
        parameters['activity'] = 'training' #Change Here
        # Event-log reading parameters
        parameters['read_options'] = {
            'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
            #'timeformat': '%d-%m-%Y %H:%M',
            #'timeformat': '%H:%M.%S',
            'column_names': column_names,
            'one_timestamp': parameters['one_timestamp'],
            'ns_include': True}
        # General training parameters
        if parameters['activity'] in ['training']:
            # Event-log parameters
            parameters['file_name'] = 'sepsis_cases_1.csv' #Change Here
            # Specific model training parameters
            if parameters['activity'] == 'training':
                parameters['imp'] = 2  # keras lstm implementation 1 cpu,2 gpu
                parameters['lstm_act'] = 'tanh' # optimization function Keras, None in v1, 'relu' in v1.1
                parameters['dense_act'] = 'sigmoid'  # optimization function Keras, used at output layer for time opt: linear or sigmoid
                parameters['optim'] = 'Nadam'  # optimization function Keras
                parameters['norm_method'] = 'lognorm'  # max, lognorm
                # Model types --> shared_cat, specialized, concatenated, 
                #                 shared_cat_gru, specialized_gru, concatenated_gru
                parameters['model_type'] = 'concatenated'
                parameters['n_size'] = 10  # n-gram sizeA
                parameters['l_size'] = 50  # LSTM layer sizes
                # Generation parameters
        elif parameters['activity'] in ['pred_log', 'pred_sfx', 'predict_next']:
            parameters['folder'] = '20210428_68D85C7D_E0B5_47F8_92BC_79732E5F58C5'
            parameters['model_file'] = 'model_concatenated_10-1.50.h5'
            parameters['is_single_exec'] = True  # single or batch execution
            parameters['variant'] = 'random_choice'  # random_choice, arg_max for variants and repetitions to be tested
            parameters['rep'] = 1
        else:
            raise ValueError(parameters['activity'])
    else:
        # Catch parameters by console
        try:
            opts, _ = getopt.getopt(
                argv,
                "ho:a:f:i:l:d:p:n:m:z:y:c:b:x:t:e:v:r:",
                ['one_timestamp=', 'activity=',
                 'file_name=', 'imp=', 'lstm_act=',
                 'dense_act=', 'optim=', 'norm_method=',
                 'model_type=', 'n_size=', 'l_size=',
                 'folder=', 'model_file=', 'is_single_exec=',
                 'max_trace_size=', 'splits=', 'sub_group=',
                 'variant=', 'rep='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if arg in ['None', 'none']:
                    parameters[key] = None
                elif key in ['is_single_exec', 'one_timestamp']:
                    parameters[key] = arg in ['True', 'true', 1]
                elif key in ['imp', 'n_size', 'l_size',
                             'max_trace_size','splits', 'rep']:
                    parameters[key] = int(arg)
                else:
                    parameters[key] = arg
            parameters['read_options'] = {'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
                                          'column_names': column_names,
                                          'one_timestamp':
                                              parameters['one_timestamp'],
                                              'ns_include': True}
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
#   Execution
    if parameters['activity'] == 'training':
        print(parameters)
        trainer = tr.ModelTrainer(parameters)
        print(trainer.output, trainer.model, sep=' ')
    elif parameters['activity'] in ['predict_next', 'pred_sfx', 'pred_log']:
        print(parameters)
        print(parameters['folder'])
        print(parameters['model_file'])
        predictor = pr.ModelPredictor(parameters)
        print(predictor.acc)
if __name__ == "__main__":
    main(sys.argv[1:])
